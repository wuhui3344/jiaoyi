package com.kronos.assistant.insight

import com.kronos.assistant.domain.Direction
import com.kronos.assistant.domain.ForecastPoint
import com.kronos.assistant.domain.ForecastResult
import com.kronos.assistant.domain.Insight
import com.kronos.assistant.indicator.Indicators
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 文字研判引擎。
 *
 * 完全**本地规则驱动**，不调用任何外部 LLM：
 * 1. 统计所有采样路径的终点分布，得到期望收益与分歧度；
 * 2. 结合路径一致性（多少比例看涨）、预测区间宽度、历史波动率与关键技术位；
 * 3. 输出结构化研判 + 要点 + 风险提示。
 *
 * 这样做的原因：APK 定位是「设备端离线推理」，引入在线 LLM 会破坏隐私与可用性假设。
 * 若需要大模型润色，可在 [Insight] 的基础上接入可选的 LLM Provider。
 */
object InsightEngine {

    /** 各条路径终点相对最后收盘价的收益率。 */
    fun pathReturns(result: ForecastResult): FloatArray {
        val last = result.lastClose
        if (last <= 0f) return FloatArray(result.paths.size)
        return FloatArray(result.paths.size) { i ->
            val path = result.paths[i]
            if (path.isEmpty()) 0f else (path.last().close - last) / last
        }
    }

    fun analyze(result: ForecastResult, historicalVolatility: Float): Insight {
        val last = result.lastClose
        val returns = pathReturns(result)
        val meanReturn = if (returns.isEmpty()) result.expectedReturn else returns.average().toFloat()
        val dispersion = stdDev(returns)

        val upCount = returns.count { it > 0 }
        val direction = Direction.of(meanReturn)
        val agreement = if (returns.isEmpty()) 0.5f else {
            when (direction) {
                Direction.UP -> upCount.toFloat() / returns.size
                Direction.DOWN -> (returns.size - upCount).toFloat() / returns.size
                Direction.NEUTRAL -> 1f - abs(upCount.toFloat() / returns.size - 0.5f) * 2f
            }
        }

        // 预测区间宽度（相对值）作为模型自身的不确定度
        val intervalWidth = result.points.lastOrNull()?.let { p ->
            if (last <= 0f) 0f else (p.closeP90 - p.closeP10) / last
        } ?: 0f

        // 信噪比: |期望收益| / 路径分歧，越大说明信号越一致
        val ratio = if (dispersion < 1e-6f) (if (abs(meanReturn) > 1e-6f) 3f else 0f) else abs(meanReturn) / dispersion
        val confidence = (0.45f * agreement + 0.35f * (ratio / 3f).coerceIn(0f, 1f) + 0.20f * (1f - intervalWidth.coerceIn(0f, 1f)))
            .coerceIn(0f, 1f)

        val predictedVol = volatilityOfPaths(result)

        val bullets = buildBullets(result, meanReturn, agreement, intervalWidth, historicalVolatility, predictedVol)
        val risks = buildRisks(result, meanReturn, dispersion, historicalVolatility, predictedVol)

        return Insight(
            direction = direction,
            confidence = confidence,
            expectedReturn = meanReturn,
            volatility = predictedVol,
            summary = buildSummary(result, direction, meanReturn, confidence, agreement),
            bullets = bullets,
            risks = risks,
        )
    }

    // ----------------------------------------------------------------------------------

    private fun buildSummary(
        result: ForecastResult,
        direction: Direction,
        meanReturn: Float,
        confidence: Float,
        agreement: Float,
    ): String {
        val steps = result.points.size
        val name = result.symbol.name
        val tf = result.timeframe.label
        val pct = meanReturn * 100f
        val horizon = horizonText(result)
        val agreementPct = (agreement * 100).toInt()
        val confidenceText = when {
            confidence >= 0.7f -> "信号一致性较高"
            confidence >= 0.5f -> "信号中等"
            else -> "信号偏弱"
        }
        val directionText = when (direction) {
            Direction.UP -> "模型整体预期上行"
            Direction.DOWN -> "模型整体预期下行"
            Direction.NEUTRAL -> "模型判断偏向震荡"
        }
        return "$name $tf 未来 $steps 根 K 线（约 $horizon）: $directionText，" +
            "各采样路径平均涨跌幅 ${formatPercent(pct)}，$agreementPct% 的路径方向一致，$confidenceText。"
    }

    private fun buildBullets(
        result: ForecastResult,
        meanReturn: Float,
        agreement: Float,
        intervalWidth: Float,
        histVol: Float,
        predVol: Float,
    ): List<String> {
        val last = result.lastClose
        val bullets = mutableListOf<String>()

        result.points.lastOrNull()?.let { p ->
            bullets += "预测末点收盘价 ${"%.4f".format(p.close)}，区间 ${"%.4f".format(p.closeP10)} ~ ${"%.4f".format(p.closeP90)}"
        }
        bullets += "路径分歧度 ${formatPercent(intervalWidth * 100f / 2f)}（半个 10%~90% 区间宽度），越小说明各路径越一致"

        // 预测波动 vs 历史波动
        if (histVol > 1e-6f) {
            val ratio = predVol / histVol
            val verdict = when {
                ratio > 1.25f -> "模型预期波动显著高于近期水平，注意单边行情或数据异动"
                ratio < 0.75f -> "模型预期波动低于近期水平，可能进入缩量整理"
                else -> "模型预期波动与近期水平相当"
            }
            bullets += verdict
        }

        // 关键技术位
        val ma20 = Indicators.sma(result.history.map { it.close }, 20).lastOrNull()
        if (ma20 != null && last > 0f) {
            val diff = (last - ma20) / ma20
            bullets += if (diff >= 0f) {
                "当前价位于 MA20 上方 ${formatPercent(diff * 100f)}，中期均线构成支撑"
            } else {
                "当前价位于 MA20 下方 ${formatPercent(-diff * 100f)}，中期均线构成压力"
            }
        }

        val recentHigh = result.history.takeLast(60).maxOfOrNull { it.high }
        val recentLow = result.history.takeLast(60).minOfOrNull { it.low }
        if (recentHigh != null && recentLow != null && last > 0f) {
            val pos = ((last - recentLow) / (recentHigh - recentLow)).coerceIn(0f, 1f)
            bullets += "近 60 根 K 线区间位置 ${(pos * 100).toInt()}%（0% = 区间低点，100% = 区间高点）"
        }

        // 方向一致性
        val up = pathReturns(result).count { it > 0 }
        bullets += "看涨路径 $up / ${result.paths.size}，看跌路径 ${result.paths.size - up} / ${result.paths.size}"

        return bullets
    }

    private fun buildRisks(
        result: ForecastResult,
        meanReturn: Float,
        dispersion: Float,
        histVol: Float,
        predVol: Float,
    ): List<String> {
        val risks = mutableListOf<String>()
        if (abs(meanReturn) < dispersion) {
            risks += "期望收益的绝对值小于路径分歧度，方向性判断并不稳健，不宜作为单一决策依据"
        }
        if (histVol > 0.8f) {
            risks += "近期历史波动率偏高（年化 ${(histVol * 100).toInt()}%），止损与仓位控制尤为重要"
        }
        if (predVol > histVol * 1.5f && histVol > 1e-6f) {
            risks += "模型预测波动远高于近期实现波动，存在模型外推失准的可能"
        }
        val sampleCount = result.params.sampleCount
        if (sampleCount < 3) {
            risks += "本次仅使用 $sampleCount 条采样路径，置信区间估计不够稳定，建议提高到 5 条以上"
        }
        val lookback = result.params.lookback
        if (lookback < 128) {
            risks += "历史长度仅 $lookback 根，上下文偏短，模型对中长期结构的把握有限"
        }
        risks += "Kronos 是统计预测模型，无法预知政策、财报、突发事件等外生冲击"
        return risks
    }

    /** 用各路径逐点收益率的离散程度折算年化波动。 */
    private fun volatilityOfPaths(result: ForecastResult): Float {
        val last = result.lastClose
        if (last <= 0f || result.paths.isEmpty()) return 0f
        val logReturns = ArrayList<Double>()
        val steps = result.points.size
        if (steps < 1) return 0f
        for (path in result.paths) {
            for (i in 1 until path.size) {
                val prev = path[i - 1].close.toDouble()
                val cur = path[i].close.toDouble()
                if (prev > 1e-9 && cur > 1e-9) logReturns += kotlin.math.ln(cur / prev)
            }
        }
        if (logReturns.size < 2) return 0f
        val mean = logReturns.average()
        val variance = logReturns.sumOf { (it - mean) * (it - mean) } / (logReturns.size - 1)
        val perPeriod = sqrt(variance)
        // 按周期折算到年化
        val periodsPerYear = when {
            result.timeframe.minutes >= 1440 -> 365.0 / (result.timeframe.minutes / 1440.0)
            else -> 365.0 * 24 * 60 / result.timeframe.minutes
        }
        return (perPeriod * sqrt(periodsPerYear)).toFloat()
    }

    private fun horizonText(result: ForecastResult): String {
        val totalMinutes = result.timeframe.minutes.toLong() * result.points.size
        return when {
            totalMinutes < 60 -> "$totalMinutes 分钟"
            totalMinutes < 60 * 24 -> "${totalMinutes / 60} 小时"
            totalMinutes < 60 * 24 * 7 -> "${totalMinutes / (60 * 24)} 天"
            else -> "${totalMinutes / (60 * 24 * 7)} 周"
        }
    }

    private fun stdDev(values: FloatArray): Float {
        if (values.size < 2) return 0f
        val mean = values.average()
        var acc = 0.0
        for (v in values) acc += (v - mean) * (v - mean)
        return sqrt(acc / values.size).toFloat()
    }

    private fun formatPercent(value: Float): String {
        val sign = if (value >= 0) "+" else ""
        return "$sign${"%.2f".format(value)}%"
    }

    /** 供 UI 直接使用的便捷方法。 */
    fun analyze(result: ForecastResult): Insight =
        analyze(result, Indicators.realizedVolatility(result.history, 60, periodsPerYearFor(result)))

    private fun periodsPerYearFor(result: ForecastResult): Double = when {
        result.timeframe.minutes >= 1440 -> 365.0 / (result.timeframe.minutes / 1440.0)
        else -> 365.0 * 24 * 60 / result.timeframe.minutes
    }

    /** 由预测点构造 [ForecastPoint] 的分位信息已在引擎侧完成；这里只做展示用聚合。 */
    fun directionLabel(direction: Direction): String = direction.label
}
