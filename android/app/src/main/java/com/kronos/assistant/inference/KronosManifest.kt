package com.kronos.assistant.inference

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

/**
 * `kronos_manifest.json` 的映射。
 *
 * 该文件由 `tools/onnx_export/export_onnx.py` 产出，与 ONNX 图一同放在
 * `assets/kronos/` 下。App **不硬编码任何超参数**，全部从这里读取，
 * 这样重新导出模型（例如换用 Kronos-mini 或 finetune 版本）无需改动 Kotlin 代码。
 */
@Serializable
data class KronosManifest(
    val formatVersion: Int,
    val precision: String,
    val modelId: String,
    val tokenizerId: String,
    val modelRevision: String,
    val tokenizerRevision: String,
    /** 模型能处理的最大上下文长度（Kronos-small 为 512）。 */
    val maxContext: Int,
    /** KV cache 的预分配容量，必须与导出时一致。 */
    val cacheCapacity: Int,
    val opset: Int,
    val s1Bits: Int,
    val s2Bits: Int,
    val s1VocabSize: Int,
    val s2VocabSize: Int,
    val codebookDim: Int,
    val dModel: Int,
    val nLayers: Int,
    val nHeads: Int,
    val headDim: Int,
    val depNHeads: Int,
    val depHeadDim: Int,
    /** 预测器对输入做归一化时的截断范围。 */
    val clip: Float,
    val files: ModelFiles,
) {
    /** 主 Transformer KV cache 的元素个数（batch 恒为 1）。 */
    val transformerCacheElements: Int get() = nLayers * nHeads * cacheCapacity * headDim

    /** 依赖层 KV cache 的元素个数。 */
    val depCacheElements: Int get() = depNHeads * cacheCapacity * depHeadDim

    /**
     * 单次预测允许的最大历史长度。
     *
     * 官方实现在 `lookback + pred_len > maxContext` 时会启用滑动窗口，而滑动会改变
     * 已缓存 token 的相对位置编码，使增量解码不再与官方实现等价。因此这里强制约束，
     * 超出的部分从最早的 K 线开始丢弃（保留最近的数据）。
     */
    fun effectiveLookback(requested: Int, predLen: Int): Int =
        requested.coerceIn(MIN_LOOKBACK, maxContext - predLen)

    companion object {
        const val MIN_LOOKBACK = 32

        private val json = Json { ignoreUnknownKeys = true }

        fun parse(text: String): KronosManifest = json.decodeFromString(text)
    }
}

@Serializable
data class ModelFiles(
    @SerialName("tokenizerEncode") val tokenizerEncode: String,
    @SerialName("tokenizerDecode") val tokenizerDecode: String,
    @SerialName("decode") val decode: String,
    @SerialName("stepS2") val stepS2: String,
)
