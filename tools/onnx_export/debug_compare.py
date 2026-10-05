"""逐层对拍: 官方 decode_s1/decode_s2  vs  增量实现。

用于定位 prefill / step_s1 / step_s2 中哪一步与官方实现不一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent / "third_party" / "Kronos"
sys.path.insert(0, str(REPO))

from model import Kronos, KronosTokenizer  # noqa: E402

from kronos_onnx import (  # noqa: E402
    KronosPrefillONNX,
    KronosStepS1ONNX,
    KronosStepS2ONNX,
    TokenizerEncodeONNX,
)

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a - b).abs().max().item()
    flag = "OK " if d < 1e-4 else "BAD"
    print(f"  [{flag}] {name:<24} max|diff| = {d:.6e}")
    return d


def dep_k(model, ctx: torch.Tensor) -> torch.Tensor:
    """官方口径的依赖层 key 投影 (未施加 RoPE)。"""
    cross = model.dep_layer.cross_attn
    b, t, _ = ctx.shape
    return cross.k_proj(ctx).reshape(b, t, cross.n_heads, cross.head_dim).transpose(1, 2)


def main() -> None:
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION).eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION).eval()

    torch.manual_seed(0)
    L, P = 32, 4
    x = torch.randn(1, L, 6).clamp(-5, 5)
    x_stamp = torch.zeros(1, L, 5)
    x_stamp[..., 1] = torch.randint(0, 24, (1, L))
    x_stamp[..., 2] = torch.randint(0, 7, (1, L))
    x_stamp[..., 3] = torch.randint(1, 29, (1, L))
    x_stamp[..., 4] = torch.randint(1, 13, (1, L))
    y_stamp = x_stamp[:, :P, :].clone()
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

    enc = TokenizerEncodeONNX(tok, model.s1_bits, model.s2_bits).eval()
    prefill = KronosPrefillONNX(model).eval()
    step1 = KronosStepS1ONNX(model).eval()
    step2 = KronosStepS2ONNX(model).eval()

    with torch.no_grad():
        s1_ids, s2_ids = enc(x)
        tok_s1, tok_s2 = tok.encode(x, half=True)
        print("== tokenizer 编码 ==")
        report("s1_ids", s1_ids, tok_s1)
        report("s2_ids", s2_ids, tok_s2)
        print(f"  首个 token: mine=({s1_ids[0,0].item()},{s2_ids[0,0].item()}) torch=({tok_s1[0,0].item()},{tok_s2[0,0].item()})")

        print("\n== i=0: prefill vs decode_s1(整个 prompt) ==")
        o_s1, o_ctx = model.decode_s1(s1_ids, s2_ids, x_stamp)
        m_s1l, m_ctx, t_k, t_v, d_k, d_v = prefill(s1_ids, s2_ids, x_stamp)
        report("s1_logits[last]", o_s1[:, -1, :], m_s1l)
        report("context[last]", o_ctx[:, -1, :], m_ctx)
        # 依赖层缓存应当覆盖位置 0..L-2 (第 0 步会由 step_s2 追加位置 L-1)
        report("d_k cache (0..L-2)", dep_k(model, o_ctx[:, :-1, :]), d_k)

        gen1 = torch.argmax(m_s1l, dim=-1, keepdim=True)
        print(f"  贪心 s1_0 = {gen1.item()}")

        print("\n== i=0: step_s2 vs decode_s2 ==")
        o_s2 = model.decode_s2(o_ctx, gen1)
        m_s2, d_k2, d_v2 = step2(gen1, m_ctx, d_k, d_v)
        report("s2_logits[last]", o_s2[:, -1, :], m_s2)
        gen2 = torch.argmax(m_s2, dim=-1, keepdim=True)
        print(f"  贪心 s2_0 = {gen2.item()}")

        print("\n== i=1: step_s1 vs decode_s1(prompt + gen0) ==")
        full_s1 = torch.cat([s1_ids, gen1], dim=1)
        full_s2 = torch.cat([s2_ids, gen2], dim=1)
        o_s1b, o_ctxb = model.decode_s1(full_s1, full_s2, full_stamp[:, : L + 1, :])
        m_s1b, m_ctxb, t_k2, t_v2 = step1(
            gen1, gen2, full_stamp[:, L : L + 1, :], torch.full((1, 1), L, dtype=torch.int64), t_k, t_v
        )
        report("s1_logits[last]", o_s1b[:, -1, :], m_s1b)
        report("context[last]", o_ctxb[:, -1, :], m_ctxb)
        print("  对比历史位置的 context (应当完全一致, 说明 cache 未污染):")
        report("context[0:L]", o_ctxb[:, :L, :], o_ctx)

        print("\n== i=1: step_s2 vs decode_s2 ==")
        o_s2b = model.decode_s2(o_ctxb, torch.argmax(m_s1b, -1, keepdim=True))
        gen1b = torch.argmax(m_s1b, dim=-1, keepdim=True)
        m_s2b, _, _ = step2(gen1b, m_ctxb, d_k2, d_v2)
        report("s2_logits[last]", o_s2b[:, -1, :], m_s2b)


if __name__ == "__main__":
    main()
