package com.kronos.assistant.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.ForecastPoint
import com.kronos.assistant.domain.IndicatorPanel
import com.kronos.assistant.domain.IndicatorSeries
import com.kronos.assistant.ui.theme.ChartColors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** 图表上所有需要绘制的数据。 */
data class ChartData(
    val history: List<Candle>,
    val forecast: List<ForecastPoint>,
    /** 置信带上下沿，与 [forecast] 一一对应。 */
    val bandLow: List<Float>? = null,
    val bandHigh: List<Float>? = null,
    val overlays: List<IndicatorSeries> = emptyList(),
    /** 每根预测路径的收盘价序列，用于画"意大利面"图。 */
    val paths: List<List<Float>> = emptyList(),
)

/**
 * 主图：K 线 + 技术指标叠加 + 预测区间。
 *
 * 设计取舍：
 * - **不用第三方图表库**，全部用 Compose Canvas 手绘。原因是要叠加「历史 K 线 + 多路径
 *   预测扇形 + 置信带」这种非标准组合，现成库反而更难贴合。
 * - 横轴按"槽位"等宽排布（而不是按时间），这样周末/休市不会在图上留下空洞，
 *   与主流行情软件的表现一致。
 * - 预测段用半透明带填充 + 中位路径实线，多路径用细线弱化。
 */
@Composable
fun CandlestickChart(
    data: ChartData,
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 300.dp,
    showBand: Boolean = true,
    showPaths: Boolean = false,
    darkTheme: Boolean = true,
) {
    val totalSlots = data.history.size + data.forecast.size
    if (totalSlots == 0) {
        Box(modifier.fillMaxWidth().height(height)) {
            Text(
                "暂无数据",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(16.dp),
            )
        }
        return
    }

    var selected by remember { mutableStateOf<Int?>(null) }

    val priceRange = remember(data) { computePriceRange(data) }
    val overlayRange = remember(data) { computeOverlayRange(data.overlays) }
    val lo = min(priceRange.first, overlayRange.first)
    val hi = max(priceRange.second, overlayRange.second)
    val span = if (hi - lo < 1e-9) 1f else hi - lo
    val loPad = lo - span * 0.06f
    val hiPad = hi + span * 0.06f

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(height)
            .pointerInput(totalSlots) {
                detectTapGestures(
                    onTap = { offset ->
                        val slotWidth = size.width / totalSlots
                        val idx = (offset.x / slotWidth).toInt()
                        selected = if (idx in 0 until totalSlots) idx else null
                    },
                )
            },
    ) {
        val slotWidth = size.width / totalSlots
        val bodyWidth = (slotWidth * 0.62f).coerceAtLeast(1f)
        fun yOf(price: Float): Float =
            size.height - (price - loPad) / (hiPad - loPad) * size.height

        drawGrid(loPad, hiPad, darkTheme)
        drawAxisLabels(loPad, hiPad, darkTheme)

        // ---- 置信带 ----
        if (showBand && data.bandLow != null && data.bandHigh != null && data.forecast.isNotEmpty()) {
            val path = Path()
            val startX = data.history.size * slotWidth
            data.bandHigh.forEachIndexed { i, v ->
                val x = startX + (i + 0.5f) * slotWidth
                val y = yOf(v)
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            for (i in data.bandLow.indices.reversed()) {
                path.lineTo(startX + (i + 0.5f) * slotWidth, yOf(data.bandLow!![i]))
            }
            path.close()
            drawPath(path, ChartColors.Band)
        }

        // ---- 历史 K 线 ----
        data.history.forEachIndexed { i, c ->
            drawCandle(
                x = (i + 0.5f) * slotWidth,
                bodyWidth = bodyWidth,
                open = c.open, high = c.high, low = c.low, close = c.close,
                yOf = ::yOf,
            )
        }

        // ---- 预测段: 单路径细线 ----
        if (showPaths && data.paths.isNotEmpty()) {
            val startX = data.history.size * slotWidth
            val color = ChartColors.ForecastSoft.copy(alpha = 0.5f)
            data.paths.forEach { path ->
                val p = Path()
                path.forEachIndexed { i, v ->
                    val x = startX + (i + 0.5f) * slotWidth
                    if (i == 0) p.moveTo(x, yOf(v)) else p.lineTo(x, yOf(v))
                }
                drawPath(p, color, style = Stroke(width = 1f))
            }
        }

        // ---- 预测段: 中位路径 ----
        if (data.forecast.isNotEmpty()) {
            val startX = data.history.size * slotWidth
            val lastClose = data.history.lastOrNull()?.close
            val p = Path()
            if (lastClose != null) p.moveTo(data.history.size * slotWidth, yOf(lastClose))
            data.forecast.forEachIndexed { i, f ->
                val x = startX + (i + 0.5f) * slotWidth
                p.lineTo(x, yOf(f.close))
            }
            drawPath(p, ChartColors.Forecast, style = Stroke(width = 2.5f))

            // 预测 K 线（更细、半透明）
            data.forecast.forEachIndexed { i, f ->
                drawCandle(
                    x = startX + (i + 0.5f) * slotWidth,
                    bodyWidth = bodyWidth * 0.7f,
                    open = f.open, high = f.high, low = f.low, close = f.close,
                    yOf = ::yOf,
                    alpha = 0.75f,
                )
            }

            // 分隔线
            drawLine(
                color = ChartColors.Crosshair,
                start = Offset(data.history.size * slotWidth, 0f),
                end = Offset(data.history.size * slotWidth, size.height),
                strokeWidth = 1f,
            )
        }

        // ---- 指标叠加线 ----
        data.overlays.forEach { series -> drawSeries(series, slotWidth, ::yOf, data.history.size) }

        // ---- 选中十字线 ----
        selected?.let { idx ->
            if (idx in 0 until totalSlots) {
                val x = (idx + 0.5f) * slotWidth
                drawLine(ChartColors.Crosshair, Offset(x, 0f), Offset(x, size.height), strokeWidth = 1f)
            }
        }
    }
}

// --------------------------------------------------------------------------------------
// 绘制辅助
// --------------------------------------------------------------------------------------

private fun DrawScope.drawCandle(
    x: Float,
    bodyWidth: Float,
    open: Float,
    high: Float,
    low: Float,
    close: Float,
    yOf: (Float) -> Float,
    alpha: Float = 1f,
) {
    val bullish = close >= open
    val color = if (bullish) ChartColors.Bull else ChartColors.Bear
    val c = color.copy(alpha = alpha)
    drawLine(c, Offset(x, yOf(high)), Offset(x, yOf(low)), strokeWidth = 1f)
    val top = yOf(max(open, close))
    val bottom = yOf(min(open, close))
    val h = (bottom - top).coerceAtLeast(1f)
    drawRect(color = c, topLeft = Offset(x - bodyWidth / 2, top), size = Size(bodyWidth, h))
}

private fun DrawScope.drawSeries(
    series: IndicatorSeries,
    slotWidth: Float,
    yOf: (Float) -> Float,
    offset: Int,
) {
    val path = Path()
    var started = false
    series.values.forEachIndexed { i, v ->
        if (v == null) {
            started = false
            return@forEachIndexed
        }
        val x = (i + 0.5f) * slotWidth
        val y = yOf(v)
        if (!started) {
            path.moveTo(x, y)
            started = true
        } else {
            path.lineTo(x, y)
        }
    }
    drawPath(
        path,
        Color(series.colorArgb),
        style = Stroke(width = if (series.dashed) 1f else 1.6f),
    )
}

private fun DrawScope.drawGrid(lo: Float, hi: Float, dark: Boolean) {
    val gridColor = ChartColors.Grid
    val rows = 4
    for (i in 0..rows) {
        val y = size.height * i / rows
        drawLine(gridColor, Offset(0f, y), Offset(size.width, y), strokeWidth = 1f)
    }
}

private fun DrawScope.drawAxisLabels(lo: Float, hi: Float, dark: Boolean) {
    // 轴标签用 Compose Text 绘制在 Canvas 之外（见 PriceAxis 组件），这里只画刻度线
    val rows = 4
    for (i in 0..rows) {
        val y = size.height * i / rows
        drawLine(ChartColors.Axis, Offset(size.width - 6f, y), Offset(size.width, y), strokeWidth = 1.5f)
    }
}

private fun computePriceRange(data: ChartData): Pair<Float, Float> {
    var lo = Float.POSITIVE_INFINITY
    var hi = Float.NEGATIVE_INFINITY
    data.history.forEach {
        if (it.low < lo) lo = it.low
        if (it.high > hi) hi = it.high
    }
    data.forecast.forEach {
        if (it.low < lo) lo = it.low
        if (it.high > hi) hi = it.high
        if (it.closeP10 < lo) lo = it.closeP10
        if (it.closeP90 > hi) hi = it.closeP90
    }
    if (lo.isInfinite() || hi.isInfinite()) return 0f to 1f
    if (abs(hi - lo) < 1e-9f) return lo to (lo + 1f)
    return lo to hi
}

private fun computeOverlayRange(overlays: List<IndicatorSeries>): Pair<Float, Float> {
    var lo = Float.POSITIVE_INFINITY
    var hi = Float.NEGATIVE_INFINITY
    overlays.forEach { s ->
        s.values.forEach { v ->
            if (v != null) {
                if (v < lo) lo = v
                if (v > hi) hi = v
            }
        }
    }
    if (lo.isInfinite() || hi.isInfinite()) return Float.POSITIVE_INFINITY to Float.NEGATIVE_INFINITY
    return lo to hi
}

/** 副图：任意指标面板（MACD / RSI / KDJ）。 */
@Composable
fun IndicatorChart(
    panel: IndicatorPanel,
    slotCount: Int,
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 90.dp,
) {
    if (slotCount == 0) return
    val range = remember(panel, slotCount) {
        val fixed = panel.fixedRange
        if (fixed != null) {
            fixed.start to fixed.endInclusive
        } else {
            var lo = Float.POSITIVE_INFINITY
            var hi = Float.NEGATIVE_INFINITY
            fun scan(values: List<Float?>) {
                values.forEach { v ->
                    if (v != null) {
                        lo = min(lo, v)
                        hi = max(hi, v)
                    }
                }
            }
            panel.series.forEach { scan(it.values) }
            panel.histogram?.let { scan(it) }
            if (lo.isInfinite()) 0f to 1f else lo to hi
        }
    }
    val lo = range.first
    val hi = range.second
    val span = if (hi - lo < 1e-9f) 1f else hi - lo
    val loPad = lo - span * 0.08f
    val hiPad = hi + span * 0.08f

    Canvas(modifier.fillMaxWidth().height(height)) {
        val slotWidth = size.width / slotCount
        fun yOf(v: Float) = size.height - (v - loPad) / (hiPad - loPad) * size.height

        if (panel.zeroLine) {
            val y = yOf(0f)
            if (y in 0f..size.height) {
                drawLine(ChartColors.Grid, Offset(0f, y), Offset(size.width, y), strokeWidth = 1f)
            }
        }

        panel.histogram?.let { hist ->
            hist.forEachIndexed { i, v ->
                if (v == null) return@forEachIndexed
                val zeroY = yOf(0f)
                val y = yOf(v)
                val color = if (v >= 0) ChartColors.Bull else ChartColors.Bear
                val top = min(zeroY, y)
                val h = max(1f, abs(y - zeroY))
                drawRect(
                    color = color.copy(alpha = 0.8f),
                    topLeft = Offset(i * slotWidth + slotWidth * 0.2f, top),
                    size = Size(slotWidth * 0.6f, h),
                )
            }
        }

        panel.series.forEach { series ->
            val path = Path()
            var started = false
            series.values.forEachIndexed { i, v ->
                if (v == null) {
                    started = false
                    return@forEachIndexed
                }
                val x = (i + 0.5f) * slotWidth
                val y = yOf(v)
                if (!started) {
                    path.moveTo(x, y)
                    started = true
                } else {
                    path.lineTo(x, y)
                }
            }
            drawPath(path, Color(series.colorArgb), style = Stroke(width = 1.4f))
        }
    }
}

/** 成交量副图。与主图共用槽位宽度，保证对齐。 */
@Composable
fun VolumeChart(
    history: List<Candle>,
    forecastSize: Int,
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 60.dp,
) {
    val total = history.size + forecastSize
    if (total == 0) return
    val maxVolume = remember(history) { history.maxOfOrNull { it.volume } ?: 1f }
    Canvas(modifier.fillMaxWidth().height(height)) {
        val slotWidth = size.width / total
        val scale = if (maxVolume <= 0f) 1f else size.height / maxVolume
        history.forEachIndexed { i, c ->
            val h = c.volume * scale
            drawRect(
                color = if (c.isBullish) ChartColors.Bull.copy(alpha = 0.6f) else ChartColors.Bear.copy(alpha = 0.6f),
                topLeft = Offset(i * slotWidth + slotWidth * 0.18f, size.height - h),
                size = Size(slotWidth * 0.64f, h),
            )
        }
    }
}

/** 通用折线/多条曲线面板，用于回测净值曲线。 */
@Composable
fun MultiLineChart(
    series: List<Pair<String, Pair<List<Float>, Color>>>,
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 180.dp,
    baseline: Float? = 1f,
) {
    val allValues = series.flatMap { it.second.first }
    if (allValues.isEmpty()) return
    var lo = allValues.min()
    var hi = allValues.max()
    baseline?.let { lo = min(lo, it); hi = max(hi, it) }
    val span = if (hi - lo < 1e-9f) 1f else hi - lo
    val loPad = lo - span * 0.08f
    val hiPad = hi + span * 0.08f

    Canvas(modifier.fillMaxWidth().height(height)) {
        for (i in 0..3) {
            val y = size.height * i / 3
            drawLine(ChartColors.Grid, Offset(0f, y), Offset(size.width, y), strokeWidth = 1f)
        }
        baseline?.let {
            val y = size.height - (it - loPad) / (hiPad - loPad) * size.height
            if (y in 0f..size.height) {
                drawLine(
                    ChartColors.Crosshair,
                    Offset(0f, y),
                    Offset(size.width, y),
                    strokeWidth = 1f,
                )
            }
        }
        val count = series.maxOf { it.second.first.size }
        if (count < 2) return@Canvas
        val step = size.width / (count - 1)
        series.forEach { (_, value) ->
            val (values, color) = value
            val path = Path()
            values.forEachIndexed { i, v ->
                val x = i * step
                val y = size.height - (v - loPad) / (hiPad - loPad) * size.height
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(width = 2f))
        }
    }
}
