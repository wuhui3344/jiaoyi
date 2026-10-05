package com.kronos.assistant.indicator

import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.IndicatorBundle
import com.kronos.assistant.domain.IndicatorConfig
import com.kronos.assistant.domain.IndicatorPanel
import com.kronos.assistant.domain.IndicatorSeries
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 技术指标计算。
 *
 * 全部为纯函数、无状态；输入 K 线按时间升序，输出与输入等长，
 * 暖机期（数据不足）用 `null` 占位，绘图时直接跳过。
 *
 * 颜色使用 ARGB（`Long`），避免依赖 Compose 的 `Color`，便于单元测试。
 */
object Indicators {

    fun compute(candles: List<Candle>, config: IndicatorConfig): IndicatorBundle {
        val closes = candles.map { it.close }
        val overlays = ArrayList<IndicatorSeries>()
        val panels = ArrayList<IndicatorPanel>()

        if (config.showMa) {
            config.maPeriods.forEachIndexed { index, period ->
                overlays += IndicatorSeries(
                    name = "MA$period",
                    colorArgb = MA_COLORS[index % MA_COLORS.size],
                    values = sma(closes, period),
                )
            }
        }

        if (config.showBoll) {
            val (mid, upper, lower) = bollinger(closes, config.bollPeriod, config.bollK)
            overlays += IndicatorSeries("BOLL", 0xFF64B5F6, mid)
            overlays += IndicatorSeries("UB", 0xFF4DB6AC, upper, dashed = true)
            overlays += IndicatorSeries("LB", 0xFF4DB6AC, lower, dashed = true)
        }

        if (config.showMacd) {
            val (diff, dea, hist) = macd(closes, config.macdFast, config.macdSlow, config.macdSignal)
            panels += IndicatorPanel(
                name = "MACD(${config.macdFast},${config.macdSlow},${config.macdSignal})",
                series = listOf(
                    IndicatorSeries("DIF", 0xFFFFB74D, diff),
                    IndicatorSeries("DEA", 0xFF64B5F6, dea),
                ),
                histogram = hist,
                histogramName = "MACD",
                zeroLine = true,
            )
        }

        if (config.showRsi) {
            panels += IndicatorPanel(
                name = "RSI(${config.rsiPeriod})",
                series = listOf(IndicatorSeries("RSI", 0xFFBA68C8, rsi(closes, config.rsiPeriod))),
                zeroLine = false,
                fixedRange = 0f..100f,
            )
        }

        val (k, d, j) = kdj(candles, config.kdjPeriod)
        panels += IndicatorPanel(
            name = "KDJ(${config.kdjPeriod})",
            series = listOf(
                IndicatorSeries("K", 0xFFFFD54F, k),
                IndicatorSeries("D", 0xFF4FC3F7, d),
                IndicatorSeries("J", 0xFFF06292, j),
            ),
            zeroLine = false,
        )

        return IndicatorBundle(overlays, panels)
    }

    // ----------------------------------------------------------------------------------
    // 基础均线
    // ----------------------------------------------------------------------------------

    /** 简单移动平均。 */
    fun sma(values: List<Float>, window: Int): List<Float?> {
        if (window <= 0 || values.size < window) return List(values.size) { null }
        val out = arrayOfNulls<Float>(values.size)
        var sum = 0.0
        for (i in values.indices) {
            sum += values[i]
            if (i >= window) sum -= values[i - window]
            if (i >= window - 1) out[i] = (sum / window).toFloat()
        }
        return out.toList()
    }

    /** 指数移动平均。`alpha = 2 / (window + 1)`，与行情软件默认一致。 */
    fun ema(values: List<Float>, window: Int): List<Float?> {
        if (window <= 0 || values.isEmpty()) return List(values.size) { null }
        val out = arrayOfNulls<Float>(values.size)
        val alpha = 2.0 / (window + 1)
        var prev = 0.0
        var sum = 0.0
        for (i in values.indices) {
            if (i < window) {
                sum += values[i]
                if (i == window - 1) {
                    prev = sum / window
                    out[i] = prev.toFloat()
                }
            } else {
                prev = alpha * values[i] + (1 - alpha) * prev
                out[i] = prev.toFloat()
            }
        }
        return out.toList()
    }

    /** 滚动总体标准差。 */
    fun rollingStd(values: List<Float>, window: Int): List<Float?> {
        if (window <= 1 || values.size < window) return List(values.size) { null }
        val out = arrayOfNulls<Float>(values.size)
        var sum = 0.0
        var sumSq = 0.0
        for (i in values.indices) {
            val v = values[i].toDouble()
            sum += v
            sumSq += v * v
            if (i >= window) {
                val drop = values[i - window].toDouble()
                sum -= drop
                sumSq -= drop * drop
            }
            if (i >= window - 1) {
                val mean = sum / window
                val variance = (sumSq / window - mean * mean).coerceAtLeast(0.0)
                out[i] = sqrt(variance).toFloat()
            }
        }
        return out.toList()
    }

    /** 布林带: (中轨, 上轨, 下轨)。 */
    fun bollinger(values: List<Float>, window: Int, k: Float): Triple<List<Float?>, List<Float?>, List<Float?>> {
        val mid = sma(values, window)
        val sd = rollingStd(values, window)
        val upper = mid.mapIndexed { i, m -> if (m == null || sd[i] == null) null else m + k * sd[i]!! }
        val lower = mid.mapIndexed { i, m -> if (m == null || sd[i] == null) null else m - k * sd[i]!! }
        return Triple(mid, upper, lower)
    }

    // ----------------------------------------------------------------------------------
    // 振荡指标
    // ----------------------------------------------------------------------------------

    /** MACD: (DIF, DEA, 柱状体)。官方行情软件口径。 */
    fun macd(
        values: List<Float>,
        fast: Int = 12,
        slow: Int = 26,
        signal: Int = 9,
    ): Triple<List<Float?>, List<Float?>, List<Float?>> {
        val emaFast = ema(values, fast)
        val emaSlow = ema(values, slow)
        val dif = emaFast.mapIndexed { i, f ->
            val s = emaSlow[i]
            if (f == null || s == null) null else f - s
        }
        // DEA 是 DIF 的 EMA，但需要在 DIF 有效之后才开始累积
        val firstValid = dif.indexOfFirst { it != null }
        val dea = arrayOfNulls<Float>(values.size)
        if (firstValid >= 0) {
            val alpha = 2.0 / (signal + 1)
            var prev = 0.0
            var count = 0
            for (i in firstValid until values.size) {
                val v = dif[i]?.toDouble() ?: continue
                if (count < signal) {
                    prev += v
                    count++
                    if (count == signal) {
                        prev /= signal
                        dea[i] = prev.toFloat()
                    }
                } else {
                    prev = alpha * v + (1 - alpha) * prev
                    dea[i] = prev.toFloat()
                }
            }
        }
        val hist = dif.mapIndexed { i, d ->
            val e = dea[i]
            if (d == null || e == null) null else (d - e) * 2f
        }
        return Triple(dif, dea.toList(), hist)
    }

    /** RSI，使用 Wilder 平滑。 */
    fun rsi(values: List<Float>, window: Int = 14): List<Float?> {
        val n = values.size
        if (n <= window) return List(n) { null }
        val out = arrayOfNulls<Float>(n)
        var gain = 0.0
        var loss = 0.0
        for (i in 1..window) {
            val delta = (values[i] - values[i - 1]).toDouble()
            if (delta >= 0) gain += delta else loss -= delta
        }
        var avgGain = gain / window
        var avgLoss = loss / window
        out[window] = rsiValue(avgGain, avgLoss)
        for (i in window + 1 until n) {
            val delta = (values[i] - values[i - 1]).toDouble()
            val g = if (delta > 0) delta else 0.0
            val l = if (delta < 0) -delta else 0.0
            avgGain = (avgGain * (window - 1) + g) / window
            avgLoss = (avgLoss * (window - 1) + l) / window
            out[i] = rsiValue(avgGain, avgLoss)
        }
        return out.toList()
    }

    private fun rsiValue(avgGain: Double, avgLoss: Double): Float {
        if (avgLoss <= 1e-12) return 100f
        val rs = avgGain / avgLoss
        return (100.0 - 100.0 / (1.0 + rs)).toFloat()
    }

    /** KDJ: (K, D, J)。 */
    fun kdj(candles: List<Candle>, window: Int = 9): Triple<List<Float?>, List<Float?>, List<Float?>> {
        val n = candles.size
        val k = arrayOfNulls<Float>(n)
        val d = arrayOfNulls<Float>(n)
        val j = arrayOfNulls<Float>(n)
        var prevK = 50.0
        var prevD = 50.0
        for (i in candles.indices) {
            val from = maxOf(0, i - window + 1)
            var highest = Float.NEGATIVE_INFINITY
            var lowest = Float.POSITIVE_INFINITY
            for (t in from..i) {
                if (candles[t].high > highest) highest = candles[t].high
                if (candles[t].low < lowest) lowest = candles[t].low
            }
            val close = candles[i].close
            val range = (highest - lowest).toDouble()
            val rsv = if (range <= 1e-9) 50.0 else (close - lowest) / range * 100.0
            prevK = prevK * 2.0 / 3.0 + rsv / 3.0
            prevD = prevD * 2.0 / 3.0 + prevK / 3.0
            k[i] = prevK.toFloat()
            d[i] = prevD.toFloat()
            j[i] = (3 * prevK - 2 * prevD).toFloat()
        }
        return Triple(k.toList(), d.toList(), j.toList())
    }

    // ----------------------------------------------------------------------------------
    // 用于文字研判的摘要
    // ----------------------------------------------------------------------------------

    /** 对最近 [window] 根 K 线计算历史年化波动率（按周期折算）。 */
    fun realizedVolatility(candles: List<Candle>, window: Int = 60, periodsPerYear: Double = 365.0): Float {
        if (candles.size < 3) return 0f
        val recent = candles.takeLast(window.coerceAtLeast(3))
        val returns = ArrayList<Double>(recent.size - 1)
        for (i in 1 until recent.size) {
            val prev = recent[i - 1].close.toDouble()
            val cur = recent[i].close.toDouble()
            if (prev > 1e-9 && cur > 1e-9) returns += kotlin.math.ln(cur / prev)
        }
        if (returns.size < 2) return 0f
        val mean = returns.average()
        val variance = returns.sumOf { (it - mean) * (it - mean) } / (returns.size - 1)
        return (sqrt(variance) * sqrt(periodsPerYear)).toFloat()
    }

    /** 相对末根收盘价的累计涨跌幅，用于展示区间表现。 */
    fun changePercent(candles: List<Candle>, window: Int): Float {
        if (candles.size < 2) return 0f
        val recent = candles.takeLast(window.coerceAtLeast(2))
        val first = recent.first().close
        val last = recent.last().close
        if (abs(first) < 1e-9) return 0f
        return (last - first) / first
    }

    private val MA_COLORS = longArrayOf(
        0xFFFFB74D, // MA5  橙
        0xFF64B5F6, // MA10 蓝
        0xFFBA68C8, // MA20 紫
        0xFF4DD0E1, // MA60 青
        0xFFAED581, // 备用
    )
}
