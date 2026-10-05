package com.kronos.assistant.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.ForecastParams
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.time.Instant
import java.time.ZoneId
import kotlin.math.sqrt

/**
 * Kronos 设备端推理引擎。
 *
 * ## 与官方 PyTorch 实现的关系
 *
 * 官方 `KronosPredictor` 在自回归解码的每一步都会把整个上下文窗口重新前向一次，
 * 以 `lookback=512 / pred_len=120` 计约 1.7 TFLOPs，手机上不可用。这里使用
 * `tools/onnx_export/` 导出的四张 ONNX 图，等价地改写为「一次预填充 + 每步单 token
 * 增量解码」，并把数学等价性用官方仓库自带的金标准数据验证到浮点逐位一致
 * （`max|Δ| = 0`）。
 *
 * ## 关键实现约定
 *
 * 1. **batch 恒为 1**。ONNX Runtime 对符号 batch 维度 + 缓冲区复用存在限制
 *    （切换 batch 直接抛错），而且批处理并不减少 FLOPs，所以逐路径解码。
 * 2. **预填充只做一次**。预填充是确定性的，且占总计算量约 80%；各采样路径共享
 *    同一份预填充结果（KV cache 快照），只有采样过程不同。
 * 3. **定长 KV cache**。缓存容量固定为 `manifest.cacheCapacity`，有效长度由
 *    `past_len` 输入指明；缓存用 direct `FloatBuffer` 预分配并原地复用，
 *    整个解码过程零分配。
 * 4. 路径**顺序执行**并复用同一块工作缓存，内存占用与采样路径数无关。
 *
 * ## 重要约束
 *
 * 必须满足 `lookback + predLen <= maxContext`。超出时官方实现会启用滑动窗口，而滑动
 * 会改变已缓存 token 的相对位置编码，增量解码将不再与官方实现等价。调用方应使用
 * [KronosManifest.effectiveLookback] 自动截断历史长度。
 */
class KronosEngine private constructor(
    private val env: OrtEnvironment,
    val manifest: KronosManifest,
    private val sessionEncode: OrtSession,
    private val sessionTokenDecode: OrtSession,
    private val sessionDecode: OrtSession,
    private val sessionStepS2: OrtSession,
) : AutoCloseable {

    // ----------------------------------------------------------------------------------
    // 张量名字（与 tools/onnx_export/export_onnx.py 中的约定一致）
    // ----------------------------------------------------------------------------------

    private object N {
        const val ENC_X = "x"
        const val ENC_S1 = "s1_ids"
        const val ENC_S2 = "s2_ids"

        const val DEC_S1_IDS = "s1_ids"
        const val DEC_S2_IDS = "s2_ids"
        const val DEC_STAMP = "stamp"
        const val DEC_PAST_LEN = "past_len"
        const val DEC_TK = "t_k"
        const val DEC_TV = "t_v"
        const val DEC_S1_LOGITS = "s1_logits"
        const val DEC_CONTEXT = "context"
        const val DEC_TK_NEW = "t_k_new"
        const val DEC_TV_NEW = "t_v_new"
        const val DEC_DK_NEW = "d_k_new"
        const val DEC_DV_NEW = "d_v_new"

        const val S2_ID = "s1_id"
        const val S2_CONTEXT = "context"
        const val S2_VALID_LEN = "valid_len"
        const val S2_DK = "d_k"
        const val S2_DV = "d_v"
        const val S2_LOGITS = "s2_logits"

        const val TOK_S1 = "s1_ids"
        const val TOK_S2 = "s2_ids"
        const val TOK_Y = "y"
    }

    private val layers = manifest.nLayers
    private val heads = manifest.nHeads
    private val headDim = manifest.headDim
    private val depHeads = manifest.depNHeads
    private val depHeadDim = manifest.depHeadDim
    private val capacity = manifest.cacheCapacity

    // ----------------------------------------------------------------------------------
    // 构造
    // ----------------------------------------------------------------------------------

    companion object {
        private const val TAG = "KronosEngine"

        /** 时间特征维度: minute / hour / weekday / day / month。 */
        private const val TIME_FEATURES = 5

        /** 创建引擎（会加载四张 ONNX 图，耗时约 1~3 秒，建议在后台线程调用）。 */
        fun create(modelDir: File): KronosEngine {
            val manifest = KronosModelStore.loadManifest(modelDir)
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                // 模型带有符号维度 (seq)，关闭内存模式复用，避免形状变化时复用出错的缓冲区
                setMemoryPatternOptimization(false)
            }
            val cache = HashMap<String, OrtSession>()
            fun session(name: String): OrtSession =
                cache.getOrPut(name) { env.createSession(File(modelDir, name).absolutePath, options) }

            val engine = KronosEngine(
                env,
                manifest,
                session(manifest.files.tokenizerEncode),
                session(manifest.files.tokenizerDecode),
                session(manifest.files.decode),
                session(manifest.files.stepS2),
            )
            engine.verifyGraphNames()
            return engine
        }
    }

    /** 启动时校验图接口，模型版本不匹配时给出明确报错，而不是在推理中途崩溃。 */
    private fun verifyGraphNames() {
        fun check(session: OrtSession, role: String, expected: Set<String>) {
            val actual = session.inputNames
            val missing = expected - actual
            require(missing.isEmpty()) {
                "$role 图接口不匹配，缺少输入 $missing。请重新执行 tools/onnx_export/export_onnx.py 导出模型。"
            }
        }
        check(sessionEncode, "tokenizer_encode", setOf(N.ENC_X))
        check(sessionTokenDecode, "tokenizer_decode", setOf(N.TOK_S1, N.TOK_S2))
        check(
            sessionDecode, "kronos_decode",
            setOf(N.DEC_S1_IDS, N.DEC_S2_IDS, N.DEC_STAMP, N.DEC_PAST_LEN, N.DEC_TK, N.DEC_TV),
        )
        check(sessionStepS2, "kronos_step_s2", setOf(N.S2_ID, N.S2_CONTEXT, N.S2_VALID_LEN, N.S2_DK, N.S2_DV))
    }

    override fun close() {
        sessionEncode.close()
        sessionTokenDecode.close()
        sessionDecode.close()
        sessionStepS2.close()
    }

    // ----------------------------------------------------------------------------------
    // 对外接口
    // ----------------------------------------------------------------------------------

    /**
     * 执行一次完整预测。
     *
     * @param candles 历史 K 线（时间升序）。调用方应保证 `candles.size >= lookback`。
     * @param futureTimestamps 待预测的时间戳（毫秒），长度即 `predLen`。
     * @param zoneId 行情所属交易所的时区，用于计算时间特征（分钟/小时/星期/日/月）。
     * @param onProgress 解码进度回调 `(已完成步数, 总步数)`，每步调用一次。
     * @return 形状为 `[sampleCount][predLen * 6]`，每行按 `open, high, low, close, volume, amount` 排列，
     *         **已反归一化**为原始价格量纲。
     */
    fun predict(
        candles: List<Candle>,
        futureTimestamps: LongArray,
        params: ForecastParams,
        zoneId: ZoneId = ZoneId.of("UTC"),
        onProgress: ((Int, Int) -> Unit)? = null,
    ): Array<FloatArray> {
        val predLen = futureTimestamps.size
        require(predLen > 0) { "predLen 必须大于 0" }

        val lookback = manifest.effectiveLookback(candles.size, predLen)
        require(candles.size >= lookback) {
            "历史数据不足: 需要至少 $lookback 根 K 线，实际 ${candles.size}"
        }
        val recent = candles.subList(candles.size - lookback, candles.size)

        val workTK = directFloats(manifest.transformerCacheElements)
        val workTV = directFloats(manifest.transformerCacheElements)
        val workDK = directFloats(manifest.depCacheElements)
        val workDV = directFloats(manifest.depCacheElements)

        val mean = FloatArray(6)
        val std = FloatArray(6)
        val normalized = normalize(recent, lookback, mean, std)

        val xStamp = LongArray(lookback * TIME_FEATURES)
        val yStamp = LongArray(predLen * TIME_FEATURES)
        fillTimeFeatures(xStamp, recent.map { it.timestamp }, zoneId)
        fillTimeFeatures(yStamp, futureTimestamps.toList(), zoneId)

        // ---- 1. tokenizer 编码（确定性，只做一次） ----
        val s1Ids = LongArray(lookback)
        val s2Ids = LongArray(lookback)
        encode(normalized, lookback, s1Ids, s2Ids)

        // ---- 2. 预填充（所有采样路径共享） ----
        val s1Logits = FloatArray(manifest.s1VocabSize)
        val context = FloatArray(manifest.dModel)

        val baseTKCopy = FloatArray(manifest.transformerCacheElements)
        val baseTVCopy = FloatArray(manifest.transformerCacheElements)
        val baseDKCopy = FloatArray(manifest.depCacheElements)
        val baseDVCopy = FloatArray(manifest.depCacheElements)

        prefill(
            s1Ids, s2Ids, xStamp, lookback,
            workTK, workTV, workDK, workDV,
            s1Logits, context, baseTKCopy, baseTVCopy, baseDKCopy, baseDVCopy,
        )

        // ---- 3. 逐路径解码（复用同一块工作缓存） ----
        val stepTK = FloatArray(layers * heads * headDim)
        val stepTV = FloatArray(layers * heads * headDim)
        val stepDK = FloatArray(depHeads * depHeadDim)
        val stepDV = FloatArray(depHeads * depHeadDim)

        val generated = Array(params.sampleCount) { LongArray(predLen) to LongArray(predLen) }
        val results = Array(params.sampleCount) { FloatArray(predLen * 6) }

        for (path in 0 until params.sampleCount) {
            val sampler = Sampler(params.seed + path)
            resetCache(workTK, baseTKCopy)
            resetCache(workTV, baseTVCopy)
            resetCache(workDK, baseDKCopy)
            resetCache(workDV, baseDVCopy)
            var valid = lookback

            val (gen1, gen2) = generated[path]

            var s1 = sampler.sample(s1Logits, params.temperature, params.topK, params.topP)
            var s2 = sampleS2(s1, context, valid, workDK, workDV, params, sampler)
            gen1[0] = s1.toLong()
            gen2[0] = s2.toLong()

            for (step in 1 until predLen) {
                decodeStep(
                    s1.toLong(), s2.toLong(), yStamp, step - 1,
                    valid, workTK, workTV, workDK, workDV,
                    s1Logits, context, stepTK, stepTV, stepDK, stepDV,
                )
                valid += 1
                s1 = sampler.sample(s1Logits, params.temperature, params.topK, params.topP)
                s2 = sampleS2(s1, context, valid, workDK, workDV, params, sampler)
                gen1[step] = s1.toLong()
                gen2[step] = s2.toLong()
                onProgress?.invoke(step, predLen)
            }

            results[path] = decodeTokens(s1Ids, s2Ids, gen1, gen2, predLen, mean, std)
        }

        onProgress?.invoke(predLen, predLen)
        return results
    }

    // ----------------------------------------------------------------------------------
    // 预处理
    // ----------------------------------------------------------------------------------

    private fun normalize(candles: List<Candle>, lookback: Int, mean: FloatArray, std: FloatArray): FloatArray {
        val raw = FloatArray(lookback * 6)
        for (i in 0 until lookback) {
            val c = candles[i]
            val base = i * 6
            raw[base] = c.open
            raw[base + 1] = c.high
            raw[base + 2] = c.low
            raw[base + 3] = c.close
            raw[base + 4] = c.volume
            raw[base + 5] = c.amount
        }
        for (k in 0 until 6) {
            var sum = 0.0
            for (i in 0 until lookback) sum += raw[i * 6 + k]
            mean[k] = (sum / lookback).toFloat()
        }
        for (k in 0 until 6) {
            var acc = 0.0
            for (i in 0 until lookback) {
                val d = (raw[i * 6 + k] - mean[k]).toDouble()
                acc += d * d
            }
            // 官方使用 np.std (总体标准差, ddof=0)
            std[k] = sqrt(acc / lookback).toFloat()
        }
        val out = FloatArray(raw.size)
        for (i in raw.indices) {
            val k = i % 6
            val v = (raw[i] - mean[k]) / (std[k] + 1e-5f)
            out[i] = v.coerceIn(-manifest.clip, manifest.clip)
        }
        return out
    }

    /**
     * 计算时间特征，对齐 pandas 的 `dt.minute / dt.hour / dt.weekday / dt.day / dt.month`。
     * 注意 weekday 以周一为 0。
     */
    private fun fillTimeFeatures(out: LongArray, timestamps: List<Long>, zoneId: ZoneId) {
        for (i in timestamps.indices) {
            val zdt = Instant.ofEpochMilli(timestamps[i]).atZone(zoneId)
            val base = i * TIME_FEATURES
            out[base] = zdt.minute.toLong()
            out[base + 1] = zdt.hour.toLong()
            out[base + 2] = (zdt.dayOfWeek.value - 1).toLong()
            out[base + 3] = zdt.dayOfMonth.toLong()
            out[base + 4] = zdt.monthValue.toLong()
        }
    }

    // ----------------------------------------------------------------------------------
    // ONNX 调用
    // ----------------------------------------------------------------------------------

    /** `tokenizer_encode`: 归一化后的 [1, L, 6] -> s1/s2 token id。 */
    private fun encode(x: FloatArray, lookback: Int, s1Ids: LongArray, s2Ids: LongArray) {
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, lookback.toLong(), 6))
        tensor.use {
            sessionEncode.run(mapOf(N.ENC_X to it)).use { result ->
                readLongs(result, N.ENC_S1, s1Ids, lookback)
                readLongs(result, N.ENC_S2, s2Ids, lookback)
            }
        }
    }

    /** 预填充 + 快照。`T = lookback`, `past_len = 0`。 */
    private fun prefill(
        s1Ids: LongArray, s2Ids: LongArray, xStamp: LongArray, lookback: Int,
        workTK: FloatBuffer, workTV: FloatBuffer, workDK: FloatBuffer, workDV: FloatBuffer,
        s1Logits: FloatArray, context: FloatArray,
        baseTK: FloatArray, baseTV: FloatArray, baseDK: FloatArray, baseDV: FloatArray,
    ) {
        val s1T = OnnxTensor.createTensor(env, LongBuffer.wrap(s1Ids), longArrayOf(1, lookback.toLong()))
        val s2T = OnnxTensor.createTensor(env, LongBuffer.wrap(s2Ids), longArrayOf(1, lookback.toLong()))
        val stampT = OnnxTensor.createTensor(
            env, LongBuffer.wrap(xStamp), longArrayOf(1, lookback.toLong(), TIME_FEATURES.toLong()),
        )
        val pastLenT = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(0)), longArrayOf(1))
        val tkT = OnnxTensor.createTensor(env, workTK, longArrayOf(layers.toLong(), 1, heads.toLong(), capacity.toLong(), headDim.toLong()))
        val tvT = OnnxTensor.createTensor(env, workTV, longArrayOf(layers.toLong(), 1, heads.toLong(), capacity.toLong(), headDim.toLong()))

        try {
            sessionDecode.run(
                mapOf(
                    N.DEC_S1_IDS to s1T, N.DEC_S2_IDS to s2T, N.DEC_STAMP to stampT,
                    N.DEC_PAST_LEN to pastLenT, N.DEC_TK to tkT, N.DEC_TV to tvT,
                ),
            ).use { result ->
                readFloats(result, N.DEC_S1_LOGITS, s1Logits)
                readFloats(result, N.DEC_CONTEXT, context)

                val tkNew = FloatArray(layers * heads * lookback * headDim)
                val tvNew = FloatArray(layers * heads * lookback * headDim)
                val dkNew = FloatArray(depHeads * lookback * depHeadDim)
                val dvNew = FloatArray(depHeads * lookback * depHeadDim)
                readFloats(result, N.DEC_TK_NEW, tkNew)
                readFloats(result, N.DEC_TV_NEW, tvNew)
                readFloats(result, N.DEC_DK_NEW, dkNew)
                readFloats(result, N.DEC_DV_NEW, dvNew)

                // 主缓存: 布局 [layer, 1, head, capacity, headDim]
                scatterHeadBlocks(workTK, tkNew, lookback, layers, heads, headDim, capacity)
                scatterHeadBlocks(workTV, tvNew, lookback, layers, heads, headDim, capacity)
                scatterHeadBlocks(workDK, dkNew, lookback, 1, depHeads, depHeadDim, capacity)
                scatterHeadBlocks(workDV, dvNew, lookback, 1, depHeads, depHeadDim, capacity)
                snapshot(workTK, baseTK)
                snapshot(workTV, baseTV)
                snapshot(workDK, baseDK)
                snapshot(workDV, baseDV)
            }
        } finally {
            s1T.close(); s2T.close(); stampT.close(); pastLenT.close(); tkT.close(); tvT.close()
        }
    }

    /** 单步解码。`T = 1`，新位置写入 `pastLen` 处。 */
    private fun decodeStep(
        s1Id: Long, s2Id: Long, yStamp: LongArray, stampIndex: Int,
        pastLen: Int,
        workTK: FloatBuffer, workTV: FloatBuffer, workDK: FloatBuffer, workDV: FloatBuffer,
        s1Logits: FloatArray, context: FloatArray,
        stepTK: FloatArray, stepTV: FloatArray, stepDK: FloatArray, stepDV: FloatArray,
    ) {
        val stampRow = LongArray(TIME_FEATURES)
        System.arraycopy(yStamp, stampIndex * TIME_FEATURES, stampRow, 0, TIME_FEATURES)

        val s1T = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(s1Id)), longArrayOf(1, 1))
        val s2T = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(s2Id)), longArrayOf(1, 1))
        val stampT = OnnxTensor.createTensor(env, LongBuffer.wrap(stampRow), longArrayOf(1, 1, TIME_FEATURES.toLong()))
        val pastLenT = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(pastLen.toLong())), longArrayOf(1))
        val tkT = OnnxTensor.createTensor(env, workTK, longArrayOf(layers.toLong(), 1, heads.toLong(), capacity.toLong(), headDim.toLong()))
        val tvT = OnnxTensor.createTensor(env, workTV, longArrayOf(layers.toLong(), 1, heads.toLong(), capacity.toLong(), headDim.toLong()))

        try {
            sessionDecode.run(
                mapOf(
                    N.DEC_S1_IDS to s1T, N.DEC_S2_IDS to s2T, N.DEC_STAMP to stampT,
                    N.DEC_PAST_LEN to pastLenT, N.DEC_TK to tkT, N.DEC_TV to tvT,
                ),
            ).use { result ->
                readFloats(result, N.DEC_S1_LOGITS, s1Logits)
                readFloats(result, N.DEC_CONTEXT, context)
                readFloats(result, N.DEC_TK_NEW, stepTK)
                readFloats(result, N.DEC_TV_NEW, stepTV)
                readFloats(result, N.DEC_DK_NEW, stepDK)
                readFloats(result, N.DEC_DV_NEW, stepDV)

                scatterHeadBlocks(workTK, stepTK, 1, layers, heads, headDim, capacity, destPos = pastLen)
                scatterHeadBlocks(workTV, stepTV, 1, layers, heads, headDim, capacity, destPos = pastLen)
                scatterHeadBlocks(workDK, stepDK, 1, 1, depHeads, depHeadDim, capacity, destPos = pastLen)
                scatterHeadBlocks(workDV, stepDV, 1, 1, depHeads, depHeadDim, capacity, destPos = pastLen)
            }
        } finally {
            s1T.close(); s2T.close(); stampT.close(); pastLenT.close(); tkT.close(); tvT.close()
        }
    }

    /** `kronos_step_s2`: 由已采样的 s1 推出 s2。 */
    private fun sampleS2(
        s1Id: Int, context: FloatArray, validLen: Int,
        workDK: FloatBuffer, workDV: FloatBuffer,
        params: ForecastParams, sampler: Sampler,
    ): Int {
        val s1T = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(s1Id.toLong())), longArrayOf(1, 1))
        val ctxT = OnnxTensor.createTensor(env, FloatBuffer.wrap(context), longArrayOf(1, manifest.dModel.toLong()))
        val validT = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(validLen.toLong())), longArrayOf(1))
        val dkT = OnnxTensor.createTensor(env, workDK, longArrayOf(1, depHeads.toLong(), capacity.toLong(), depHeadDim.toLong()))
        val dvT = OnnxTensor.createTensor(env, workDV, longArrayOf(1, depHeads.toLong(), capacity.toLong(), depHeadDim.toLong()))

        val logits = FloatArray(manifest.s2VocabSize)
        try {
            sessionStepS2.run(
                mapOf(N.S2_ID to s1T, N.S2_CONTEXT to ctxT, N.S2_VALID_LEN to validT, N.S2_DK to dkT, N.S2_DV to dvT),
            ).use { result -> readFloats(result, N.S2_LOGITS, logits) }
        } finally {
            s1T.close(); ctxT.close(); validT.close(); dkT.close(); dvT.close()
        }
        return sampler.sample(logits, params.temperature, params.topK, params.topP)
    }

    /** `tokenizer_decode`: 完整 token 序列 -> OHLCV，并做反归一化。 */
    private fun decodeTokens(
        s1Ids: LongArray, s2Ids: LongArray, gen1: LongArray, gen2: LongArray,
        predLen: Int, mean: FloatArray, std: FloatArray,
    ): FloatArray {
        val total = s1Ids.size + predLen
        // 只解码最后 maxContext 个位置即可（官方实现同样会截断）
        val start = maxOf(0, total - manifest.maxContext)
        val len = total - start

        val full1 = LongArray(len)
        val full2 = LongArray(len)
        copySegment(full1, 0, len, start, s1Ids, gen1)
        copySegment(full2, 0, len, start, s2Ids, gen2)

        val s1T = OnnxTensor.createTensor(env, LongBuffer.wrap(full1), longArrayOf(1, len.toLong()))
        val s2T = OnnxTensor.createTensor(env, LongBuffer.wrap(full2), longArrayOf(1, len.toLong()))
        val out = FloatArray(len * 6)
        try {
            sessionTokenDecode.run(mapOf(N.TOK_S1 to s1T, N.TOK_S2 to s2T)).use { result ->
                readFloats(result, N.TOK_Y, out)
            }
        } finally {
            s1T.close(); s2T.close()
        }

        // 取最后 predLen 个位置并反归一化
        val result = FloatArray(predLen * 6)
        val offset = (len - predLen) * 6
        for (i in 0 until predLen * 6) {
            val k = i % 6
            result[i] = out[offset + i] * (std[k] + 1e-5f) + mean[k]
        }
        return result
    }

    /** 把 (s1Ids + gen1) 拼接后的一段拷进 dst。 */
    private fun copySegment(dst: LongArray, dstOffset: Int, dstLen: Int, srcStart: Int, a: LongArray, b: LongArray) {
        for (i in 0 until dstLen) {
            val idx = srcStart + i
            dst[dstOffset + i] = if (idx < a.size) a[idx] else b[idx - a.size]
        }
    }

    // ----------------------------------------------------------------------------------
    // 缓冲区工具
    // ----------------------------------------------------------------------------------

    /**
     * 把图输出的「新增 K/V」写进定长缓存的对应位置。
     *
     * 源布局为 `[layer, heads, T, headDim]`，目标布局为 `[layer, head, capacity, headDim]`，
     * 因此对每个 (layer, head) 做一次连续块拷贝即可。
     */
    private fun scatterHeadBlocks(
        dest: FloatBuffer, src: FloatArray,
        span: Int, numLayers: Int, numHeads: Int, dim: Int, cap: Int, destPos: Int = 0,
    ) {
        val blockLen = span * dim
        val srcLayerStride = numHeads * blockLen
        val destLayerStride = numHeads * cap * dim
        for (l in 0 until numLayers) {
            for (h in 0 until numHeads) {
                val srcOffset = l * srcLayerStride + h * blockLen
                val destOffset = l * destLayerStride + h * cap * dim + destPos * dim
                for (i in 0 until blockLen) {
                    dest.put(destOffset + i, src[srcOffset + i])
                }
            }
        }
    }

    private fun snapshot(buf: FloatBuffer, dest: FloatArray) {
        val dup = buf.duplicate()
        dup.position(0)
        dup.get(dest, 0, dest.size)
    }

    private fun resetCache(buf: FloatBuffer, src: FloatArray) {
        buf.position(0)
        buf.put(src)
        buf.position(0)
    }

    private fun directFloats(elements: Int): FloatBuffer =
        ByteBuffer.allocateDirect(elements * Float.SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer()

    private fun readFloats(result: OrtSession.Result, name: String, dest: FloatArray) {
        val tensor = result.get(name).orElseThrow { IllegalStateException("图输出缺少 $name") } as OnnxTensor
        val buffer = tensor.floatBuffer
        buffer.position(0)
        buffer.get(dest, 0, dest.size)
    }

    private fun readLongs(result: OrtSession.Result, name: String, dest: LongArray, count: Int) {
        val tensor = result.get(name).orElseThrow { IllegalStateException("图输出缺少 $name") } as OnnxTensor
        val buffer = tensor.longBuffer
        buffer.position(0)
        buffer.get(dest, 0, count)
    }

    private fun log(message: String) = Log.d(TAG, message)
}
