# tools/onnx_export

把 **Kronos-small + Kronos-Tokenizer-base** 从 PyTorch 导出为 Android 端可用的 ONNX 图。

## 为什么需要这一层

官方 `KronosPredictor` 在自回归解码的每一步都会把**整个上下文窗口**重新前向一次：

```python
s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
```

以 `lookback=512, pred_len=120` 计算，总计算量约 **1.7 TFLOPs** —— 在手机 CPU 上需要数分钟，
完全不可用。

本目录把同一套权重重新组织为「**一次预填充 + 每步单 token 增量解码**」：

| 阶段 | 计算量 |
|---|---|
| 预填充 (lookback=256) | ≈ 5.9 GMACs |
| 每个解码步 | ≈ 21 MMACs |
| `pred_len=60` 合计 | ≈ 7.2 GMACs |

即 **约 1/50 的计算量**，并且预填充只做一次、所有采样路径共享。

## 产出的四张图

| 文件 | 输入 → 输出 |
|---|---|
| `tokenizer_encode.onnx` | `x[1,L,6]` → `s1_ids[1,L]`, `s2_ids[1,L]` |
| `tokenizer_decode.onnx` | `s1_ids[1,T]`, `s2_ids[1,T]` → `y[1,T,6]` |
| `kronos_decode.onnx` | `s1_ids[1,T]`, `s2_ids[1,T]`, `stamp[1,T,5]`, `past_len[1]`, `t_k[8,1,8,512,64]`, `t_v[...]` → `s1_logits[1,1024]`, `context[1,512]`, `t_k_new[8,1,8,T,64]`, `t_v_new`, `d_k_new[1,4,T,128]`, `d_v_new` |
| `kronos_step_s2.onnx` | `s1_id[1,1]`, `context[1,512]`, `valid_len[1]`, `d_k[1,4,512,128]`, `d_v` → `s2_logits[1,1024]` |

`kronos_decode` 一张图同时覆盖两种调用形态：

- **预填充**：`T = lookback`、`past_len = 0`（缓存全为零，被掩码屏蔽）
- **增量步**：`T = 1`、`past_len ≥ lookback`

之所以能共用，是因为因果掩码写作 `keep[j] = j <= past_len + i`：
`T = 1` 时该式恒成立（退化为"全部可见"），`past_len = 0, T = L` 时正好是标准三角因果掩码。

## 使用方法

```powershell
# 1. 准备环境（需要 Python 3.10 ~ 3.12，不支持 3.13+）
python -m venv tools\.venv
tools\.venv\Scripts\python.exe -m pip install -r tools\onnx_export\requirements.txt

# 2. 等价性自检（会下载模型，约 115MB）
tools\.venv\Scripts\python.exe tools\onnx_export\export_onnx.py --self-check

# 3. 导出到 Android assets
tools\.venv\Scripts\python.exe tools\onnx_export\export_onnx.py

# 4. 用官方金标准数据验证
tools\.venv\Scripts\python.exe tools\onnx_export\verify_onnx.py --lookback 256

# 5. 性能基准
tools\.venv\Scripts\python.exe tools\onnx_export\verify_onnx.py --bench
```

### 网络受限时的镜像配置

Python 的 `requests` 走某些本地代理会出现 `SSLEOFError`。在国内环境建议：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"      # 禁用 Xet 存储，否则会在 cas-bridge.xethub.hf.co 超时
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
```

## 验证结果

导出脚本带两级验证：

1. **`--self-check`**：把增量解码与官方 `auto_regressive_inference` 对拍（贪心采样保证可复现）。
   当前结果为 **`max|Δ| = 0.000e+00`**，即浮点逐位一致。

2. **`verify_onnx.py`**：用官方仓库自带的金标准
   （`third_party/Kronos/tests/data/regression_output_256.csv`）做端到端验证，
   结果 `max rel = 9.2e-07`（官方测试自身的容差是 `rtol=1e-5`）。

## 移植过程中必须复刻的 6 个细节

这些是逐位等价的关键，改动 ONNX 导出时请务必保留：

1. **`emb_s1` 不能乘 `sqrt(d_model)`**。缩放只发生在 `HierarchicalEmbedding.forward` 内部；
   `Kronos.forward` 与 `decode_s2` 里的 `sibling_embed = self.embedding.emb_s1(...)` 是不缩放的。

2. **KV cache 必须存「未施加 RoPE」的 key**，每次使用时按各自的相对位置补上 RoPE。

3. **缓存输入/输出必须传完整序列**，而不是只传新增的那一份。

4. **依赖层跨注意力的 RoPE 怪癖**：`MultiHeadCrossAttentionWithRoPE` 用 `q.shape[-2]` 构造
   cos/sin 表，再同时作用到 q 和 k 上。`decode_s2` 中 `q_len` 恒为 1，广播后等价于
   **所有 key 都在位置 0 被旋转**。必须原样复刻。

5. **`bitwise_and` 需要 opset 18**。为了留在 opset 17（移动端兼容性更好），改用
   `floor(idx / 2^i) % 2`。

6. **ONNX Runtime 未实现 bool 类型的 `Where`**（opset 16）。掩码改用 `And`/`Or`/`Not`。

## 引擎侧的额外约束

- **`lookback + pred_len <= 512`**。超出时官方实现会启用滑动窗口，而滑动会改变已缓存
  token 的相对位置编码，增量解码将不再等价。App 侧用
  `KronosManifest.effectiveLookback()` 自动截断（保留最近的数据）。

- **batch 恒为 1**。ONNX Runtime 在符号 batch 维度上切换形状时会因缓冲区复用而直接抛错，
  而且批处理并不减少 FLOPs。代之以「预填充一次 + 多路径顺序解码」。

## 文件说明

| 文件 | 作用 |
|---|---|
| `kronos_onnx.py` | 包装模块：四张图的定义、RoPE/注意力/RMSNorm 的手写复刻 |
| `export_onnx.py` | 导出脚本，`--self-check` 做等价性对拍 |
| `verify_onnx.py` | 金标准验证 + `--bench` 性能基准 |
| `debug_compare.py` | 逐层对拍，用于定位数值差异 |
| `requirements.txt` | 依赖清单 |
