"""用官方仓库自带的金标准数据验证导出的 ONNX 图。

``third_party/Kronos/tests/data/`` 里有一组官方回归测试数据:

* ``regression_input.csv``      -- 输入 K 线 (5 分钟频, 含 open/high/low/close/volume/amount)
* ``regression_output_256.csv`` -- 用 ``lookback=256, pred_len=8, top_k=1`` 得到的期望输出
* ``regression_output_512.csv`` -- ``lookback=512`` 版本

本脚本在**纯 numpy + onnxruntime** 上复刻 App 将要执行的控制流, 并与金标准逐值对比。
``lookback=512`` 违反 ``lookback + pred_len <= max_context`` 约束 (官方实现会滑动窗口,
增量解码不再等价), 因此默认只验证 256 版本; 512 版本会作为「已知差异」单独报告。

用法::

    python verify_onnx.py                       # 默认验证 256
    python verify_onnx.py --lookback 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = REPO_ROOT / "third_party" / "Kronos" / "tests" / "data"
DEFAULT_MODEL_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "kronos"

FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]


# --------------------------------------------------------------------------------------
# 采样 (复刻 model/kronos.py 的 top_k_top_p_filtering + sample_from_logits)
# --------------------------------------------------------------------------------------


def _top_k_top_p_filter(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    logits = logits.astype(np.float32).copy()
    if top_k > 0:
        k = min(max(int(top_k), 1), logits.shape[-1])
        kth = np.sort(logits, axis=-1)[..., -k][..., None]
        logits[logits < kth] = -np.inf
        return logits
    if top_p < 1.0:
        idx = np.argsort(-logits, axis=-1)
        sorted_logits = np.take_along_axis(logits, idx, axis=-1)
        cumulative = np.cumsum(_softmax(sorted_logits, axis=-1), axis=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].copy()
        remove[..., 0] = False
        scatter = np.zeros_like(remove)
        np.put_along_axis(scatter, idx, remove, axis=-1)
        logits[scatter] = -np.inf
    return logits


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def sample(logits: np.ndarray, temperature: float, top_k: int, top_p: float, rng: np.random.Generator) -> np.ndarray:
    logits = logits / temperature
    if top_k > 0 or top_p < 1.0:
        logits = _top_k_top_p_filter(logits, top_k, top_p)
    probs = _softmax(logits, axis=-1)
    if top_k == 1:
        return np.argmax(probs, axis=-1, keepdims=True).astype(np.int64)
    out = np.empty((probs.shape[0], 1), dtype=np.int64)
    for i in range(probs.shape[0]):
        out[i, 0] = rng.choice(probs.shape[1], p=probs[i] / probs[i].sum())
    return out


# --------------------------------------------------------------------------------------
# 时间戳 (复刻 model/kronos.py::calc_time_stamps)
# --------------------------------------------------------------------------------------


def calc_time_stamps(ts: pd.Series) -> np.ndarray:
    idx = pd.DatetimeIndex(ts)
    return np.stack(
        [idx.minute, idx.hour, idx.weekday, idx.day, idx.month], axis=-1
    ).astype(np.float32)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


class OnnxKronos:
    """在 onnxruntime 上运行 Kronos 增量解码。"""

    def __init__(self, model_dir: Path, precision: str = "fp32"):
        suffix = "" if precision == "fp32" else ".fp16"
        manifest_path = model_dir / ("kronos_manifest.json" if precision == "fp32" else "kronos_manifest.fp16.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("precision") != precision:
            raise SystemExit(f"manifest 精度为 {self.manifest.get('precision')}, 与请求的 {precision} 不符")

        files = self.manifest["files"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 模型带动态轴 (batch / seq), 必须关闭内存模式复用, 否则切换 batch 时
        # ORT 会尝试复用形状不匹配的缓冲区并直接抛错。
        opts.enable_mem_pattern = False
        self.enc = ort.InferenceSession(str(model_dir / files["tokenizerEncode"]), opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(model_dir / files["tokenizerDecode"]), opts, providers=["CPUExecutionProvider"])
        self.decode = ort.InferenceSession(str(model_dir / files["decode"]), opts, providers=["CPUExecutionProvider"])
        self.step2 = ort.InferenceSession(str(model_dir / files["stepS2"]), opts, providers=["CPUExecutionProvider"])

        self.max_context = int(self.manifest["maxContext"])

    # -- 编码 -----------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``x``: [1, L, 6] -> s1_ids, s2_ids 各 [1, L]"""
        s1, s2 = self.enc.run(None, {"x": x.astype(np.float32)})
        return s1.astype(np.int64), s2.astype(np.int64)

    def decode_tokens(self, s1_ids: np.ndarray, s2_ids: np.ndarray) -> np.ndarray:
        """``tokenizer_decode`` 图: token 序列 -> 6 维 K 线。"""
        (y,) = self.dec.run(None, {"s1_ids": s1_ids.astype(np.int64), "s2_ids": s2_ids.astype(np.int64)})
        return y

    # -- 自回归生成 ------------------------------------------------------------------

    def generate(
        self,
        x: np.ndarray,
        x_stamp: np.ndarray,
        y_stamp: np.ndarray,
        pred_len: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        seed: int = 123,
    ) -> np.ndarray:
        """返回 ``[sample_count, pred_len, 6]`` (已反归一化交给调用方)。

        实现要点
        --------
        1. **batch 恒为 1**。ONNX Runtime 对符号 batch 维度 + 缓冲区复用的处理存在
           限制 (切换 batch 会直接抛错), 且批处理并不减少 FLOPs, 因此这里逐路径解码。
        2. **预填充只做一次**。预填充是确定性的 (只依赖 prompt 的 token 序列), 且占总
           计算量约 80%, 所以各采样路径共享同一份预填充结果 —— 只有采样过程不同。
        3. 路径**顺序执行**并复用同一块工作缓存 (原地重置), 内存占用与路径数无关。
        """
        lookback = x.shape[1]
        if lookback + pred_len > self.max_context:
            raise SystemExit(
                f"lookback({lookback}) + pred_len({pred_len}) > max_context({self.max_context}); "
                "该配置会触发官方实现的滑动窗口, 增量解码不再等价"
            )

        m = self.manifest
        nl, nh, hd = int(m["nLayers"]), int(m["nHeads"]), int(m["headDim"])
        dnh, dhd = int(m["depNHeads"]), int(m["depHeadDim"])
        cap = int(m["cacheCapacity"])

        s1_ids, s2_ids = self.encode(x)  # [1, L] (tokenizer 确定性)
        full_stamp = np.concatenate([x_stamp, y_stamp], axis=1)  # [1, L+P, 5]

        def new_caches():
            return (
                np.zeros((nl, 1, nh, cap, hd), np.float32),
                np.zeros((nl, 1, nh, cap, hd), np.float32),
                np.zeros((1, dnh, cap, dhd), np.float32),
                np.zeros((1, dnh, cap, dhd), np.float32),
            )

        def decode(s1_in, s2_in, stamp_in, valid, t_k, t_v, d_k, d_v):
            """在 `valid` 处写入新位置, 返回最后位置的 s1 logits 与 context。"""
            out = self.decode.run(
                None,
                {
                    "s1_ids": s1_in.astype(np.int64),
                    "s2_ids": s2_in.astype(np.int64),
                    "stamp": stamp_in.astype(np.float32),
                    "past_len": np.array([valid], np.int64),
                    "t_k": t_k,
                    "t_v": t_v,
                },
            )
            logits, ctx, k_new, v_new, dk_new, dv_new = out
            span = s1_in.shape[1]
            t_k[:, :, :, valid : valid + span, :] = k_new
            t_v[:, :, :, valid : valid + span, :] = v_new
            d_k[:, :, valid : valid + span, :] = dk_new
            d_v[:, :, valid : valid + span, :] = dv_new
            return logits, ctx

        def s2_head(s1_in, ctx, valid, d_k, d_v):
            (logits,) = self.step2.run(
                None,
                {
                    "s1_id": s1_in.astype(np.int64),
                    "context": ctx,
                    "valid_len": np.array([valid], np.int64),
                    "d_k": d_k,
                    "d_v": d_v,
                },
            )
            return logits

        # ---------- 预填充 (所有采样路径共享) ----------
        base_tk, base_tv, base_dk, base_dv = new_caches()
        logits0, ctx0 = decode(s1_ids, s2_ids, full_stamp[:, :lookback, :], 0, base_tk, base_tv, base_dk, base_dv)

        # ---------- 逐路径解码 (复用同一块工作缓存) ----------
        t_k, t_v, d_k, d_v = new_caches()
        results = []
        for path in range(sample_count):
            rng = np.random.default_rng(seed + path)
            np.copyto(t_k, base_tk)
            np.copyto(t_v, base_tv)
            np.copyto(d_k, base_dk)
            np.copyto(d_v, base_dv)
            valid = lookback

            s1_cur = sample(logits0, temperature, top_k, top_p, rng)
            s2_cur = sample(s2_head(s1_cur, ctx0, valid, d_k, d_v), temperature, top_k, top_p, rng)
            gen1 = [s1_cur.copy()]
            gen2 = [s2_cur.copy()]

            for i in range(1, pred_len):
                stamp = full_stamp[:, lookback + i - 1 : lookback + i, :]
                logits, ctx = decode(gen1[-1], gen2[-1], stamp, valid, t_k, t_v, d_k, d_v)
                valid += 1
                s1_cur = sample(logits, temperature, top_k, top_p, rng)
                s2_cur = sample(s2_head(s1_cur, ctx, valid, d_k, d_v), temperature, top_k, top_p, rng)
                gen1.append(s1_cur.copy())
                gen2.append(s2_cur.copy())

            full_s1 = np.concatenate([s1_ids] + gen1, axis=1)
            full_s2 = np.concatenate([s2_ids] + gen2, axis=1)
            results.append(self.decode_tokens(full_s1, full_s2)[0])

        return np.stack([r[-pred_len:] for r in results], axis=0)


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------


def run_bench(args) -> None:
    """合成数据基准: 评估不同 lookback / pred_len / sample_count 下的耗时与步数。"""
    import time

    engine = OnnxKronos(args.model_dir, precision=args.precision)
    rng = np.random.default_rng(0)

    print(f"{'lookback':>9} {'pred_len':>9} {'samples':>8} {'tokens':>8} {'seconds':>9} {'ms/step':>9}")
    for lookback, pred_len, samples in [
        (192, 48, 1),
        (256, 60, 1),
        (256, 60, 5),
        (384, 96, 1),
        (448, 64, 5),
    ]:
        if lookback + pred_len > engine.max_context:
            continue
        ts = 1_700_000_000_000 + np.arange(lookback + pred_len) * 300_000
        base = 100.0 + np.cumsum(rng.normal(0, 0.2, lookback + pred_len))
        ohlc = np.stack([base, base + 0.3, base - 0.3, base + 0.05], axis=-1).astype(np.float32)
        vol = np.abs(rng.normal(1000, 100, lookback + pred_len)).astype(np.float32)
        amt = (vol * ohlc[:, 3]).astype(np.float32)
        feats = np.concatenate([ohlc, vol[:, None], amt[:, None]], axis=-1)

        x = feats[:lookback][None, :, :]
        mean, std = x[0].mean(axis=0), x[0].std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)

        stamps = np.stack(
            [
                (ts // 60_000 % 60),
                (ts // 3_600_000 % 24),
                ((ts // 86_400_000 + 4) % 7),
                (ts // 86_400_000) % 32,
                (ts // 2_629_800_000) % 13,
            ],
            axis=-1,
        ).astype(np.float32)
        x_stamp = stamps[:lookback][None, :, :]
        y_stamp = stamps[lookback : lookback + pred_len][None, :, :]

        t0 = time.perf_counter()
        engine.generate(x_norm, x_stamp, y_stamp, pred_len, temperature=1.0, top_k=0, top_p=0.9, sample_count=samples)
        dt = time.perf_counter() - t0
        tokens = samples * pred_len
        print(f"{lookback:>9} {pred_len:>9} {samples:>8} {tokens:>8} {dt:>9.2f} {dt / pred_len * 1000:>9.1f}")

    print("\n注: 桌面 CPU 测得的数值仅供量级参考; 手机端大致会慢 3~8 倍。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--lookback", type=int, default=256, choices=[256, 512])
    parser.add_argument("--pred-len", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--bench", action="store_true", help="跑性能基准 (用合成数据, 不做金标准对比)")
    args = parser.parse_args()

    if args.bench:
        run_bench(args)
        return

    df = pd.read_csv(DATA_DIR / "regression_input.csv", parse_dates=["timestamps"])
    expected_df = pd.read_csv(DATA_DIR / f"regression_output_{args.lookback}.csv", parse_dates=["timestamps"])

    ctx = df.iloc[: args.lookback].copy()
    x_feat = ctx[FEATURE_NAMES].to_numpy(dtype=np.float32)
    x_ts = ctx["timestamps"].reset_index(drop=True)
    y_ts = df["timestamps"].iloc[args.lookback : args.lookback + args.pred_len].reset_index(drop=True)
    expected = expected_df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    x_mean = x_feat.mean(axis=0)
    x_std = x_feat.std(axis=0)
    x_norm = np.clip((x_feat - x_mean) / (x_std + 1e-5), -5.0, 5.0)[None, :, :]

    x_stamp = calc_time_stamps(x_ts)[None, :, :]
    y_stamp = calc_time_stamps(y_ts)[None, :, :]

    engine = OnnxKronos(args.model_dir, precision=args.precision)
    preds = engine.generate(
        x_norm,
        x_stamp,
        y_stamp,
        pred_len=args.pred_len,
        temperature=1.0,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_count=1,
    )[0]
    preds = preds * (x_std + 1e-5) + x_mean

    abs_diff = np.abs(preds - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-9)

    print(f"lookback={args.lookback} pred_len={args.pred_len} precision={args.precision}")
    print("  逐列最大绝对 / 相对误差:")
    for i, name in enumerate(FEATURE_NAMES):
        print(
            f"    {name:<8} |Δ|max={abs_diff[:, i].max():.6e}   relΔmax={rel_diff[:, i].max():.3e}"
            f"   |value|~{np.abs(expected[:, i]).mean():.4g}"
        )

    table = pd.DataFrame(
        {
            "gold_close": expected[:, 3],
            "pred_close": preds[:, 3],
            "diff": abs_diff[:, 3],
        }
    )
    print(table.to_string())

    # 与官方 tests/test_kronos_regression.py 一致: 使用相对容差
    max_rel = rel_diff.max()
    if max_rel > args.tolerance:
        raise SystemExit(f"❌ 最大相对误差 {max_rel:.3e} 超出容差 {args.tolerance}")
    print(f"✅ ONNX 与官方 PyTorch 金标准一致 (max rel = {max_rel:.3e}, 容差 {args.tolerance})")


if __name__ == "__main__":
    main()
