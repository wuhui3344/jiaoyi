# Kronos 金融分析助手

基于 [Kronos](https://github.com/shiyu-coder/Kronos)（首个面向金融 K 线的开源基础模型，AAAI 2026）
打造的 **Android 端金融分析应用**。模型推理**完全在设备本地完成**，行情数据不会离开手机。

- 模型：`NeoQuasar/Kronos-small` (24.7M) + `NeoQuasar/Kronos-Tokenizer-base` (4.0M)
- 推理：ONNX Runtime for Android，CPU FP32
- 行情：Binance（加密货币）、东方财富（A 股 / 港股 / 国内期货）、Yahoo Finance（美股）

---

## 这个项目做了什么

官方推理实现 (`KronosPredictor`) 在自回归解码的**每一步**都会把整个上下文窗口重新前向一次。
以 `lookback=512, pred_len=120` 计算约 **1.7 TFLOPs** —— 手机上需要数分钟，不可用。

本项目把同一套权重改写为「**一次预填充 + 每步单 token 增量解码**」，并导出为 ONNX：

| | 官方实现 | 本项目 |
|---|---|---|
| lookback=256, pred_len=60 单路径 | ≈ 36 GMACs | ≈ 7.2 GMACs |
| 手机端可行性 | ✗ | ✓（实测桌面 2.0s，手机约 6~15s） |

改写后的实现与官方 PyTorch 结果**浮点逐位一致**（`max|Δ| = 0.000e+00`），
并用官方仓库自带的金标准回归数据完成端到端验证（`max rel = 9.2e-07`，官方容差 `1e-5`）。

---

## 目录结构

```
jiaoyi/
├── tools/onnx_export/          # PyTorch → ONNX 导出工具链
│   ├── kronos_onnx.py          #   四张 ONNX 图的定义（含 RoPE/注意力的手写复刻）
│   ├── export_onnx.py          #   导出脚本 + 与官方实现的等价性自检
│   ├── verify_onnx.py          #   金标准验证 + 性能基准
│   ├── debug_compare.py        #   逐层对拍，定位数值差异
│   └── README.md               #   ⚠️ 移植细节与踩坑记录，改导出前必读
│
├── third_party/Kronos/         # shiyu-coder/Kronos 源码（浅克隆，用于导出与验证）
│
└── android/                    # Android 应用
    ├── gradle/libs.versions.toml
    └── app/src/main/
        ├── assets/kronos/      # 👈 导出的 ONNX 模型放这里（约 117MB，不入库）
        └── java/com/kronos/assistant/
            ├── inference/      # ONNX Runtime 封装、KV cache、采样器
            ├── domain/         # 领域模型
            ├── data/           # 行情数据源 + 自选持久化
            ├── analysis/       # 编排：行情 → 推理 → 分位聚合
            ├── indicator/      # MA / BOLL / MACD / RSI / KDJ
            ├── backtest/       # 滚动回测
            ├── insight/        # 文字研判（本地规则引擎）
            └── ui/             # Compose 界面 + 自绘 K 线图
```

---

## 快速开始

### 1. 导出模型（必需）

```powershell
# 用 Python 3.10~3.12（不支持 3.13+，PyTorch 尚无对应 wheel）
python -m venv tools\.venv
tools\.venv\Scripts\python.exe -m pip install -r tools\onnx_export\requirements.txt

# 先自检，确认增量解码与官方实现逐位一致
tools\.venv\Scripts\python.exe tools\onnx_export\export_onnx.py --self-check

# 导出到 android/app/src/main/assets/kronos/
tools\.venv\Scripts\python.exe tools\onnx_export\export_onnx.py

# 可选：金标准验证与性能基准
tools\.venv\Scripts\python.exe tools\onnx_export\verify_onnx.py --lookback 256
tools\.venv\Scripts\python.exe tools\onnx_export\verify_onnx.py --bench
```

国内环境需要镜像加速，详见 `tools/onnx_export/README.md`。

### 2. 构建 APK

需要 **Android SDK**（compileSdk 35 / build-tools）+ **JDK 17+**。

```powershell
cd android

# 首次需要指定 SDK 路径（Android Studio 打开工程会自动生成）
"your-sdk-dir" | Out-File -Encoding ascii local.properties
# 文件内容应为: sdk.dir=D\:\\Android\\Sdk

.\gradlew.bat assembleDebug        # 产物: app\build\outputs\apk\debug\
.\gradlew.bat assembleRelease
```

也可以直接 **用 Android Studio 打开 `android/` 目录**，Gradle Sync 后点 Run。

> `minSdk = 26`（ONNX Runtime 1.20 的要求），`targetSdk = 35`。
> 默认只打包 `arm64-v8a` 与 `armeabi-v7a`。

---

## 功能

**分析页**
- 自绘 K 线图：历史 K 线 + 预测 K 线 + 中位路径 + 10%~90% 置信带 + 全部采样路径叠加
- 成交量副图，以及与主图槽位严格对齐的指标副图
- 参数可调：历史长度、预测步数、采样路径数、温度、Top-P
- 预测明细表（含每个预测点的分位数）

**技术指标**
- 主图叠加：MA(5/10/20/60)、BOLL(20, 2)
- 副图：MACD(12,26,9)、RSI(14)、KDJ(9)

**文字研判**（本地规则引擎，不调用外部 LLM）
- 方向判定 + 置信度（综合路径一致性、信噪比、区间宽度）
- 要点：期望涨跌幅、路径分歧度、预测波动 vs 历史波动、MA20 相对位置、区间位置
- 风险提示：分歧大于期望、高波动、采样路径过少、历史长度不足

**回测**
- 滚动预测 → 按预测收益方向持有/空仓 → 扣除双边手续费
- 净值曲线 vs 买入持有，输出总收益/年化/最大回撤/夏普/胜率/持仓占比
- 窗口数量自动限制，保证手机上可完成

**自选**
- DataStore 持久化，支持加密货币 / A 股 / 国内期货 / 美股 / 港股
- 国内期货按交易所（SHFE / DCE / CZCE / INE / CFFEX）+ 合约代码添加

---

## 关键实现约定

### 1. 上下文约束：`lookback + pred_len <= 512`

超出时官方实现会启用滑动窗口，而滑动会改变已缓存 token 的相对位置编码，
增量解码将不再与官方等价。App 侧用 `KronosManifest.effectiveLookback()` 自动截断
（保留**最近**的数据，这在信号上也更合理）。UI 会提示已截断。

### 2. batch 恒为 1，预填充只做一次

ONNX Runtime 在符号 batch 维度上切换形状时，会因缓冲区复用而直接抛错；
而且批处理并不减少 FLOPs。因此改为：

- 预填充执行一次并快照 KV cache（预填充是确定性的，且占总计算量约 80%）
- 每条采样路径复制快照后独立解码，复用同一块工作缓存

实测 5 条路径只比 1 条慢 4×（而非 5×）。

### 3. 定长 KV cache + 零分配

缓存容量固定为 `maxContext`，有效长度由 `past_len` 输入指明，掩码在图中完成。
端上使用 direct `FloatBuffer` 预分配并原地复用，整个解码过程**不产生任何堆分配**。

### 4. 详细移植细节

ONNX 导出过程中必须复刻的 6 个非显而易见的行为（RoPE 缓存语义、依赖层的 RoPE 广播怪癖、
`emb_s1` 不缩放、opset 兼容性等）见 **`tools/onnx_export/README.md`**。

---

## 已知限制

- **不支持滑动窗口**，即单次预测的上下文上限为 512 根 K 线。
- **东方财富无 4 小时周期**，请求 4h 时会退回 1h 数据。
- **Yahoo Finance 不返回成交额**，按 `volume × 典型价` 估算（与官方预测器在缺少
  `amount` 列时的处理一致）。
- APK 体积较大（模型约 117MB）。如需减小，可导出 fp16 版本
  （`--precision fp16`，约减半，但需要重新做精度验证）。
- 回测为**演示级**：未建模滑点、冲击成本、涨跌停与停牌。

---

## 免责声明

本工具的输出是统计模型的**抽样预测结果**，不构成任何投资建议。
金融市场价格具有高度随机性，请独立判断并自行承担风险。

## 许可

- Kronos 模型与代码：[MIT License](https://github.com/shiyu-coder/Kronos/blob/master/LICENSE)
- 本仓库代码：MIT

## 引用

```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```
