"""把 Kronos-small + Kronos-Tokenizer-base 导出为 Android 端可用的 ONNX 图。

用法::

    python export_onnx.py                      # 导出到 android/app/src/main/assets/kronos
    python export_onnx.py --output-dir out     # 指定输出目录
    python export_onnx.py --precision fp16     # 额外产出 fp16 版本 (体积减半)
    python export_onnx.py --self-check         # 只做等价性自检, 不导出

产出文件
--------
======================  ==========================================================
文件                     说明
======================  ==========================================================
``tokenizer_encode.onnx``   ``x[B,L,6]`` -> ``s1_ids[B,L]``, ``s2_ids[B,L]``
``tokenizer_decode.onnx``   ``s1_ids[B,T]``, ``s2_ids[B,T]`` -> ``y[B,T,6]``
``kronos_prefill.onnx``     预填充, 产出 KV cache
``kronos_step_s1.onnx``     单步 s1 解码
``kronos_step_s2.onnx``     单步 s2 解码 (依赖已采样的 s1)
``kronos_manifest.json``    超参数与张量签名, 供 App 读取
======================  ==========================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
KRONOS_REPO = REPO_ROOT / "third_party" / "Kronos"

# 允许 `from model import ...`
sys.path.insert(0, str(KRONOS_REPO))

from model import Kronos, KronosTokenizer  # noqa: E402

from kronos_onnx import (  # noqa: E402
    IncrementalReferenceDecoder,
    KronosDecodeONNX,
    KronosStepS2ONNX,
    TokenizerDecodeONNX,
    TokenizerEncodeONNX,
)

# 与 tests/test_kronos_regression.py 一致的固定 revision, 保证可复现
MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CONTEXT = 512
OPSET = 17

DEFAULT_OUTPUT = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "kronos"


def _log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def load_models(device: str = "cpu"):
    _log(f"加载 tokenizer: {TOKENIZER_ID}@{TOKENIZER_REVISION[:8]}")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    _log(f"加载模型: {MODEL_ID}@{MODEL_REVISION[:8]}")
    model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.eval().to(device)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"Kronos 参数量: {n_params / 1e6:.1f}M")
    return tokenizer, model


# --------------------------------------------------------------------------------------
# 自检: 增量解码 vs 官方实现
# --------------------------------------------------------------------------------------


@torch.no_grad()
def self_check(model, tokenizer, device="cpu", lookback: int = 128, pred_len: int = 8) -> None:
    """用官方 ``auto_regressive_inference`` 与增量解码对拍。

    使用 ``top_k=1`` (贪心) 保证两条路径给出完全相同的 token 序列。
    """
    from model.kronos import auto_regressive_inference  # type: ignore

    torch.manual_seed(0)
    b, l = 1, lookback
    x_norm = torch.randn(b, l, 6) * 0.8
    x_norm = x_norm.clamp(-5, 5)
    x_stamp = torch.zeros(b, l, 5)
    x_stamp[..., 1] = torch.randint(0, 24, (b, l))
    x_stamp[..., 2] = torch.randint(0, 7, (b, l))
    x_stamp[..., 3] = torch.randint(1, 29, (b, l))
    x_stamp[..., 4] = torch.randint(1, 13, (b, l))
    y_stamp = x_stamp[:, :pred_len, :].clone()

    kw = dict(max_context=MAX_CONTEXT, pred_len=pred_len, clip=5, T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False)
    official = auto_regressive_inference(
        tokenizer, model, x_norm.clone(), x_stamp.clone(), y_stamp.clone(), **kw
    )
    official = official[:, -pred_len:, :]

    dec = IncrementalReferenceDecoder(model, tokenizer, max_context=MAX_CONTEXT)
    incremental = dec.generate(x_norm, x_stamp, y_stamp, pred_len, temperature=1.0, top_k=1, top_p=1.0, sample_count=1)

    # official 是 numpy, incremental 是 tensor
    import numpy as _np

    official = _np.asarray(official, dtype=_np.float32)
    incremental = incremental.detach().cpu().numpy().astype(_np.float32)

    diff = float(_np.abs(official - incremental).max())
    rel = float((_np.abs(official - incremental) / (_np.abs(official) + 1e-6)).max())
    _log(f"  official   close[:4] = {official[0, :4, 3]}")
    _log(f"  incremental close[:4] = {incremental[0, :4, 3]}")
    _log(f"自检 max|Δ|={diff:.3e}  max rel={rel:.3e}")
    if diff > 1e-4:
        raise SystemExit(f"自检失败: 增量解码与官方实现差异过大 ({diff:.3e})")
    _log("自检通过 ✅")


# --------------------------------------------------------------------------------------
# ONNX 导出
# --------------------------------------------------------------------------------------


def _export(module, args, path: Path, input_names, output_names, dynamic_axes, precision: str, dtype=torch.float32):
    module.eval()
    export_module = module
    if precision == "fp16":
        export_module = module.half()
        args = tuple(a.half() if a.is_floating_point() else a for a in args)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    _log(f"  写出 {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def export_all(tokenizer, model, out_dir: Path, precision: str = "fp32") -> dict:
    precision = precision.lower()
    suffix = "" if precision == "fp32" else ".fp16"
    dtype = torch.float32 if precision == "fp32" else torch.float16

    n_layers = model.n_layers
    n_heads = model.n_heads
    head_dim = model.d_model // model.n_heads
    dep_heads = model.dep_layer.cross_attn.n_heads
    dep_head_dim = model.dep_layer.cross_attn.head_dim
    d_model = model.d_model
    s1_vocab = 2 ** model.s1_bits

    # ---------------- tokenizer encode ----------------
    enc = TokenizerEncodeONNX(tokenizer, model.s1_bits, model.s2_bits)
    _export(
        enc,
        (torch.randn(1, 64, 6, dtype=torch.float32),),
        out_dir / f"tokenizer_encode{suffix}.onnx",
        ["x"],
        ["s1_ids", "s2_ids"],
        {"x": {0: "batch", 1: "seq"}, "s1_ids": {0: "batch", 1: "seq"}, "s2_ids": {0: "batch", 1: "seq"}},
        precision,
        dtype,
    )

    # ---------------- tokenizer decode ----------------
    dec = TokenizerDecodeONNX(tokenizer, model.s1_bits, model.s2_bits)
    _export(
        dec,
        (torch.zeros(1, 8, dtype=torch.int64), torch.zeros(1, 8, dtype=torch.int64)),
        out_dir / f"tokenizer_decode{suffix}.onnx",
        ["s1_ids", "s2_ids"],
        ["y"],
        {"s1_ids": {0: "batch", 1: "seq"}, "s2_ids": {0: "batch", 1: "seq"}, "y": {0: "batch", 1: "seq"}},
        precision,
        dtype,
    )

    # ---------------- 统一解码图 (prefill 与增量步共用) ----------------
    #
    # KV cache 使用**预分配定长**布局: 容量固定为 max_context, 由 `past_len` 指明有效长度。
    # 这样图中 cache 的形状是静态的, 端上可以复用一块 direct buffer 做到零拷贝/零分配。
    cache_cap = MAX_CONTEXT
    decode = KronosDecodeONNX(model)
    _export(
        decode,
        (
            torch.zeros(1, 8, dtype=torch.int64),
            torch.zeros(1, 8, dtype=torch.int64),
            torch.zeros(1, 8, 5, dtype=torch.float32),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(n_layers, 1, n_heads, cache_cap, head_dim, dtype=torch.float32),
            torch.zeros(n_layers, 1, n_heads, cache_cap, head_dim, dtype=torch.float32),
        ),
        out_dir / f"kronos_decode{suffix}.onnx",
        ["s1_ids", "s2_ids", "stamp", "past_len", "t_k", "t_v"],
        ["s1_logits", "context", "t_k_new", "t_v_new", "d_k_new", "d_v_new"],
        {
            "s1_ids": {0: "batch", 1: "seq"},
            "s2_ids": {0: "batch", 1: "seq"},
            "stamp": {0: "batch", 1: "seq"},
            "t_k": {1: "batch"},
            "t_v": {1: "batch"},
            "t_k_new": {1: "batch", 3: "seq"},
            "t_v_new": {1: "batch", 3: "seq"},
            "d_k_new": {0: "batch", 2: "seq"},
            "d_v_new": {0: "batch", 2: "seq"},
            "s1_logits": {0: "batch"},
            "context": {0: "batch"},
        },
        precision,
        dtype,
    )

    # ---------------- step_s2 ----------------
    step2 = KronosStepS2ONNX(model)
    _export(
        step2,
        (
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, d_model, dtype=torch.float32),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dep_heads, cache_cap, dep_head_dim, dtype=torch.float32),
            torch.zeros(1, dep_heads, cache_cap, dep_head_dim, dtype=torch.float32),
        ),
        out_dir / f"kronos_step_s2{suffix}.onnx",
        ["s1_id", "context", "valid_len", "d_k", "d_v"],
        ["s2_logits"],
        {
            "s1_id": {0: "batch"},
            "context": {0: "batch"},
            "d_k": {0: "batch"},
            "d_v": {0: "batch"},
            "s2_logits": {0: "batch"},
        },
        precision,
        dtype,
    )

    manifest = {
        "formatVersion": 1,
        "precision": precision,
        "modelId": MODEL_ID,
        "tokenizerId": TOKENIZER_ID,
        "modelRevision": MODEL_REVISION,
        "tokenizerRevision": TOKENIZER_REVISION,
        "maxContext": MAX_CONTEXT,
        "cacheCapacity": MAX_CONTEXT,
        "opset": OPSET,
        "s1Bits": model.s1_bits,
        "s2Bits": model.s2_bits,
        "s1VocabSize": s1_vocab,
        "s2VocabSize": 2 ** model.s2_bits,
        "codebookDim": tokenizer.codebook_dim,
        "dModel": d_model,
        "nLayers": n_layers,
        "nHeads": n_heads,
        "headDim": head_dim,
        "depNHeads": dep_heads,
        "depHeadDim": dep_head_dim,
        "clip": 5.0,
        "files": {
            "tokenizerEncode": f"tokenizer_encode{suffix}.onnx",
            "tokenizerDecode": f"tokenizer_decode{suffix}.onnx",
            "decode": f"kronos_decode{suffix}.onnx",
            "stepS2": f"kronos_step_s2{suffix}.onnx",
        },
    }
    return manifest


def write_manifest(out_dir: Path, manifest: dict, name: str = "kronos_manifest.json") -> None:
    path = out_dir / name
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"  写出 {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Kronos ONNX 模型")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precision", choices=["fp32", "fp16", "both"], default="fp32")
    parser.add_argument("--self-check", action="store_true", help="只做增量解码与官方实现的等价性自检")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer, model = load_models(args.device)

    if args.self_check:
        self_check(model, tokenizer, device=args.device)
        return

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清理历史版本的产物 (早期拆分为 prefill / step_s1 两张图的命名)
    for stale in ("kronos_prefill", "kronos_step_s1"):
        for f in out_dir.glob(f"{stale}*.onnx"):
            _log(f"  清理旧文件 {f.name}")
            f.unlink()

    precisions = ["fp32", "fp16"] if args.precision == "both" else [args.precision]
    for precision in precisions:
        _log(f"=== 导出 {precision} ===")
        t0 = time.time()
        manifest = export_all(tokenizer, model, out_dir, precision=precision)
        write_manifest(out_dir, manifest, "kronos_manifest.json" if precision == "fp32" else "kronos_manifest.fp16.json")
        _log(f"{precision} 完成, 耗时 {time.time() - t0:.1f}s")

    _log("全部导出完成 ✅")
    _log(f"输出目录: {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
