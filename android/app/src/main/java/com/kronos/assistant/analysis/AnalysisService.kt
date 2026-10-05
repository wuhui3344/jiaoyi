package com.kronos.assistant.analysis

import com.kronos.assistant.data.MarketDataRepository
import com.kronos.assistant.data.nextTimestamps
import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.CandleSeries
import com.kronos.assistant.domain.ForecastParams
import com.kronos.assistant.domain.ForecastPoint
import com.kronos.assistant.domain.ForecastResult
import com.kronos.assistant.domain.IndicatorBundle
import com.kronos.assistant.domain.IndicatorConfig
import com.kronos.assistant.domain.Insight
import com.kronos.assistant.domain.Market
import com.kronos.assistant.domain.SymbolInfo
import com.kronos.assistant.domain.Timeframe
import com.kronos.assistant.indicator.Indicators
import com.kronos.assistant.inference.KronosEngine
import com.kronos.assistant.insight.InsightEngine
import java.time.ZoneId

/**
 * 一次分析的完整产物：数据 + 预测 + 指标 + 研判。
 */
data class AnalysisReport(
    val forecast: ForecastResult,
    val indicators: IndicatorBundle,
    val insight: Insight,
    /** 预测区间的置信带，供图表直接绘制（已含历史段的对齐占位）。 */
    val confidenceBand: ConfidenceBand,
) {
    val symbol: SymbolInfo get() = forecast.symbol
    val timeframe: Timeframe get() = forecast.timeframe
    val history: List<Candle> get() = forecast.history
    val points: List<ForecastPoint> get() = forecast.points
}

/** 预测段的置信带（10%~90% 分位）。 */
data class ConfidenceBand(
    val timestamps: LongArray,
    val low: FloatArray,
    val high: FloatArray,
)

/**
 * 把「行情数据」与「Kronos 推理」串起来的用例层。
 *
 * 关键职责：
 * 1. **自动截断历史长度**，保证 `lookback + predLen <= maxContext`。超出时官方实现会
 *    启用滑动窗口，增量解码将不再与官方等价，因此这里保留最近的数据。
 * 2. **多路径分位聚合**：把 `sampleCount` 条采样路径转换为 10%/90% 分位与标准差，
 *    作为不确定度的可视化依据。
 * 3. 顺带计算技术指标与文字研判。
 */
class AnalysisService(
    private val repository: MarketDataRepository,
    private val engineProvider: suspend () -> KronosEngine,
) {

    suspend fun analyze(
        symbol: SymbolInfo,
        timeframe: Timeframe,
        requestedLookback: Int,
        predLen: Int,
        sampleCount: Int,
        temperature: Float = 1.0f,
        topK: Int = 0,
        topP: Float = 0.9f,
        seed: Long = System.currentTimeMillis(),
        indicatorConfig: IndicatorConfig = IndicatorConfig(),
        onStage: (String) -> Unit = {},
        onStep: (Int, Int) -> Unit = { _, _ -> },
    ): AnalysisReport {
        onStage("正在获取行情数据…")
        val engine = engineProvider()
        val lookback = engine.manifest.effectiveLookback(requestedLookback, predLen)

        // 多取一些，便于指标计算与图表展示（历史段画满 lookback，前面再留一段）
        val fetchLimit = (lookback + 120).coerceAtLeast(200)
        val candles = repository.candles(symbol, timeframe, fetchLimit)
        val zone = repository.timezoneFor(symbol)

        onStage("正在生成预测…")
        val lastTs = candles.last().timestamp
        val future = nextTimestamps(lastTs, timeframe, predLen, symbol.market, zone)

        val params = ForecastParams(
            lookback = lookback,
            predLen = predLen,
            sampleCount = sampleCount,
            temperature = temperature,
            topK = topK,
            topP = topP,
            seed = seed,
        )

        val history = candles.takeLast(lookback)
        val raw = engine.predict(history, future, params, zone, onStep)
        val forecast = buildForecast(symbol, timeframe, history, future, raw, params)

        onStage("正在计算指标…")
        val indicators = Indicators.compute(forecast.history, indicatorConfig)

        onStage("正在生成研判…")
        val periodsPerYear = periodsPerYear(timeframe)
        val histVol = Indicators.realizedVolatility(forecast.history, 60, periodsPerYear)
        val insight = InsightEngine.analyze(forecast, histVol)

        val band = ConfidenceBand(
            timestamps = future,
            low = FloatArray(predLen) { forecast.points[it].closeP10 },
            high = FloatArray(predLen) { forecast.points[it].closeP90 },
        )

        return AnalysisReport(forecast, indicators, insight, band)
    }

    /** 只需要行情时使用（图表预加载、自选列表刷新）。 */
    fun candles(symbol: SymbolInfo, timeframe: Timeframe, limit: Int): CandleSeries =
        CandleSeries(symbol, timeframe, repository.candles(symbol, timeframe, limit))

    // ----------------------------------------------------------------------------------

    private fun buildForecast(
        symbol: SymbolInfo,
        timeframe: Timeframe,
        history: List<Candle>,
        futureTimestamps: LongArray,
        raw: Array<FloatArray>,
        params: ForecastParams,
    ): ForecastResult {
        val predLen = params.predLen
        val sampleCount = raw.size

        // 均值与分位
        val meanClose = FloatArray(predLen)
        val p10 = FloatArray(predLen)
        val p90 = FloatArray(predLen)
        val stdClose = FloatArray(predLen)
        val column = FloatArray(sampleCount)

        for (i in 0 until predLen) {
            var sum = 0.0
            for (s in 0 until sampleCount) {
                val v = raw[s][i * 6 + 3]
                column[s] = v
                sum += v
            }
            val mean = (sum / sampleCount).toFloat()
            meanClose[i] = mean
            java.util.Arrays.sort(column)
            p10[i] = percentile(column, 0.10f)
            p90[i] = percentile(column, 0.90f)
            var acc = 0.0
            for (v in column) acc += (v - mean) * (v - mean)
            stdClose[i] = kotlin.math.sqrt(acc / sampleCount).toFloat()
        }

        val points = List(predLen) { i ->
            ForecastPoint(
                timestamp = futureTimestamps[i],
                open = meanOf(raw, i, 0),
                high = meanOf(raw, i, 1),
                low = meanOf(raw, i, 2),
                close = meanClose[i],
                volume = meanOf(raw, i, 4),
                amount = meanOf(raw, i, 5),
                closeP10 = p10[i],
                closeP90 = p90[i],
                closeStd = stdClose[i],
            )
        }

        val paths = List(sampleCount) { s ->
            List(predLen) { i ->
                ForecastPoint(
                    timestamp = futureTimestamps[i],
                    open = raw[s][i * 6],
                    high = raw[s][i * 6 + 1],
                    low = raw[s][i * 6 + 2],
                    close = raw[s][i * 6 + 3],
                    volume = raw[s][i * 6 + 4],
                    amount = raw[s][i * 6 + 5],
                    closeP10 = raw[s][i * 6 + 3],
                    closeP90 = raw[s][i * 6 + 3],
                    closeStd = 0f,
                )
            }
        }

        return ForecastResult(
            symbol = symbol,
            timeframe = timeframe,
            history = history,
            points = points,
            paths = paths,
            params = params,
            elapsedMillis = 0,
        )
    }

    private fun meanOf(raw: Array<FloatArray>, step: Int, channel: Int): Float {
        var sum = 0.0
        for (path in raw) sum += path[step * 6 + channel]
        return (sum / raw.size).toFloat()
    }

    /** 线性插值分位数（与 numpy 默认的 linear 插值一致）。 */
    private fun percentile(sorted: FloatArray, q: Float): Float {
        if (sorted.isEmpty()) return 0f
        if (sorted.size == 1) return sorted[0]
        val pos = q * (sorted.size - 1)
        val lo = kotlin.math.floor(pos).toInt()
        val hi = kotlin.math.ceil(pos).toInt()
        if (lo == hi) return sorted[lo]
        val weight = pos - lo
        return sorted[lo] * (1 - weight) + sorted[hi] * weight
    }

    private fun periodsPerYear(timeframe: Timeframe): Double = when {
        timeframe.minutes >= 1440 -> 365.0 / (timeframe.minutes / 1440.0)
        else -> 365.0 * 24 * 60 / timeframe.minutes
    }
}

/** 便于在预览/测试中构造空报告。 */
fun emptyReport(symbol: SymbolInfo, timeframe: Timeframe): AnalysisReport? = null
