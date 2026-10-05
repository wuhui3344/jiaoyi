package com.kronos.assistant.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.kronos.assistant.data.DefaultSymbols
import com.kronos.assistant.data.formatCompact
import com.kronos.assistant.domain.BacktestParams
import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.Direction
import com.kronos.assistant.domain.Market
import com.kronos.assistant.domain.SymbolInfo
import com.kronos.assistant.domain.Timeframe
import com.kronos.assistant.ui.AppUiState
import com.kronos.assistant.ui.AppViewModel
import com.kronos.assistant.ui.components.CandlestickChart
import com.kronos.assistant.ui.components.ChartData
import com.kronos.assistant.ui.components.IndicatorChart
import com.kronos.assistant.ui.components.MultiLineChart
import com.kronos.assistant.ui.components.VolumeChart
import com.kronos.assistant.ui.theme.ChartColors
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs

// ======================================================================================
// 分析页
// ======================================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalysisScreen(state: AppUiState, vm: AppViewModel, modifier: Modifier = Modifier) {
    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 12.dp),
    ) {
        SymbolHeader(state, vm)
        Spacer(Modifier.height(8.dp))

        TimeframeSelector(state, vm)
        Spacer(Modifier.height(8.dp))

        ChartCard(state)
        Spacer(Modifier.height(8.dp))

        ParameterCard(state, vm)
        Spacer(Modifier.height(8.dp))

        RunButton(state, vm)
        Spacer(Modifier.height(8.dp))

        state.error?.let { ErrorCard(it) { vm.clearError() } }
        state.modelError?.let { ErrorCard(it) {} }

        state.report?.let { report ->
            InsightCard(
                direction = report.insight.direction,
                confidence = report.insight.confidence,
                expectedReturn = report.insight.expectedReturn,
                volatility = report.insight.volatility,
                summary = report.insight.summary,
                bullets = report.insight.bullets,
                risks = report.insight.risks,
            )
            Spacer(Modifier.height(8.dp))

            ForecastTable(report.points)
            Spacer(Modifier.height(8.dp))

            ForecastDetail(report.forecast.params, report.forecast.elapsedMillis)
        }

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun SymbolHeader(state: AppUiState, vm: AppViewModel) {
    val report = state.report
    val last: Candle? = report?.history?.lastOrNull()
    val prevClose = report?.history?.getOrNull(report.history.size - 2)?.close
    val change = if (last != null && prevClose != null && abs(prevClose) > 1e-9f) {
        (last.close - prevClose) / prevClose
    } else {
        0f
    }
    val isWatched = state.watchlist.any { it.id == state.symbol.id }

    Row(
        Modifier.fillMaxWidth().padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(state.symbol.name, style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.width(8.dp))
                Text(
                    state.symbol.market.label,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(
                state.symbol.code,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (last != null) {
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    formatPrice(last.close),
                    style = MaterialTheme.typography.titleLarge,
                    fontFamily = FontFamily.Monospace,
                    color = colorOf(change),
                )
                Text(
                    "${if (change >= 0) "+" else ""}${"%.2f".format(change * 100)}%",
                    style = MaterialTheme.typography.labelSmall,
                    color = colorOf(change),
                )
            }
        }
        IconButton(onClick = { vm.toggleWatch() }) {
            Icon(
                if (isWatched) Icons.Filled.Star else Icons.Outlined.StarOutline,
                contentDescription = "自选",
                tint = if (isWatched) Color(0xFFFFB300) else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun TimeframeSelector(state: AppUiState, vm: AppViewModel) {
    Row(
        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Timeframe.recommended.forEach { tf ->
            FilterChip(
                selected = state.timeframe == tf,
                onClick = { vm.setTimeframe(tf) },
                label = { Text(tf.label) },
            )
        }
    }
}

@Composable
private fun ChartCard(state: AppUiState) {
    val report = state.report
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(vertical = 8.dp)) {
            val history = report?.history.orEmpty()
            val forecast = report?.points.orEmpty()
            val visible = history.takeLast(state.visibleBars.coerceAtLeast(20))

            CandlestickChart(
                data = ChartData(
                    history = visible,
                    forecast = forecast,
                    bandLow = if (state.showBand) forecast.map { it.closeP10 } else null,
                    bandHigh = if (state.showBand) forecast.map { it.closeP90 } else null,
                    overlays = report?.indicators?.overlays.orEmpty(),
                    paths = if (state.showPaths) report?.forecast?.paths?.map { p -> p.map { it.close } }.orEmpty() else emptyList(),
                ),
                height = 280.dp,
                showBand = state.showBand,
                showPaths = state.showPaths,
            )
            if (visible.isNotEmpty()) {
                VolumeChart(
                    history = visible,
                    forecastSize = forecast.size,
                    height = 56.dp,
                    modifier = Modifier.padding(horizontal = 2.dp),
                )
            }
            report?.indicators?.panels?.forEach { panel ->
                HorizontalDivider(Modifier.padding(vertical = 4.dp))
                Row(Modifier.padding(horizontal = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(panel.name, style = MaterialTheme.typography.labelSmall)
                    Spacer(Modifier.width(10.dp))
                    panel.series.forEach { s ->
                        Text(
                            s.name,
                            style = MaterialTheme.typography.labelSmall,
                            color = Color(s.colorArgb),
                            modifier = Modifier.padding(end = 8.dp),
                        )
                    }
                }
                IndicatorChart(panel = panel, slotCount = visible.size + forecast.size, height = 84.dp)
            }
            if (report == null) {
                Box(Modifier.fillMaxWidth().height(120.dp), contentAlignment = Alignment.Center) {
                    Text(
                        if (state.modelLoading) state.modelStatus ?: "模型准备中…" else "点击下方「开始预测」生成分析",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun ParameterCard(state: AppUiState, vm: AppViewModel) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Text("推理参数", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(6.dp))

            ParamSlider(
                label = "历史长度 (lookback)",
                value = state.lookback.toFloat(),
                range = 64f..(AppUiState.MAX_CONTEXT - state.predLen).toFloat(),
                step = 32f,
                display = "${state.effectiveLookback} 根" +
                    if (state.effectiveLookback < state.lookback) "（受 512 上下文限制已截断）" else "",
            ) { vm.setLookback(it.toInt()) }

            ParamSlider(
                label = "预测步数 (pred_len)",
                value = state.predLen.toFloat(),
                range = 4f..120f,
                step = 4f,
                display = "${state.predLen} 根",
            ) { vm.setPredLen(it.toInt()) }

            ParamSlider(
                label = "采样路径数 (sample_count)",
                value = state.sampleCount.toFloat(),
                range = 1f..10f,
                step = 1f,
                display = "${state.sampleCount} 条",
            ) { vm.setSampleCount(it.toInt()) }

            ParamSlider(
                label = "温度 (temperature)",
                value = state.temperature,
                range = 0.1f..2f,
                step = 0.05f,
                display = "%.2f".format(state.temperature),
            ) { vm.setTemperature(it) }

            ParamSlider(
                label = "核采样 (top_p)",
                value = state.topP,
                range = 0.1f..1f,
                step = 0.05f,
                display = "%.2f".format(state.topP),
            ) { vm.setTopP(it) }

            ToggleRow("显示置信区间", state.showBand) { vm.setShowBand(it) }
            ToggleRow("叠加全部采样路径", state.showPaths) { vm.setShowPaths(it) }

            ParamSlider(
                label = "图表显示 K 线数量",
                value = state.visibleBars.toFloat(),
                range = 40f..400f,
                step = 20f,
                display = "${state.visibleBars} 根",
            ) { vm.setVisibleBars(it.toInt()) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ParamSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    step: Float,
    display: String,
    onChange: (Float) -> Unit,
) {
    Column(Modifier.padding(vertical = 2.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            Text(
                display,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
                fontFamily = FontFamily.Monospace,
            )
        }
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = onChange,
            valueRange = range,
            steps = if (step <= 0f) 0 else (((range.endInclusive - range.start) / step).toInt() - 1).coerceAtLeast(0),
        )
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun RunButton(state: AppUiState, vm: AppViewModel) {
    Column(Modifier.fillMaxWidth()) {
        Button(
            onClick = { vm.analyze() },
            enabled = !state.running && !state.modelLoading,
            modifier = Modifier.fillMaxWidth().height(48.dp),
        ) {
            Icon(Icons.Filled.PlayArrow, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text(if (state.running) "推理中…" else "开始预测")
        }
        if (state.running) {
            Spacer(Modifier.height(8.dp))
            Text(
                buildString {
                    append(state.stage ?: "推理中…")
                    if (state.totalSteps > 0 && state.step > 0) {
                        append("  (${state.step}/${state.totalSteps} 步)")
                    }
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (state.totalSteps > 0) {
                LinearProgressIndicator(
                    progress = { state.step.toFloat() / state.totalSteps },
                    modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                )
            }
        }
    }
}

@Composable
private fun ErrorCard(message: String, onDismiss: () -> Unit) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text("出错了", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            Text(message, style = MaterialTheme.typography.bodyMedium)
            TextButton(onClick = onDismiss) { Text("知道了") }
        }
    }
}

@Composable
private fun InsightCard(
    direction: Direction,
    confidence: Float,
    expectedReturn: Float,
    volatility: Float,
    summary: String,
    bullets: List<String>,
    risks: List<String>,
) {
    val accent = when (direction) {
        Direction.UP -> ChartColors.Bull
        Direction.DOWN -> ChartColors.Bear
        Direction.NEUTRAL -> Color(0xFF90A4AE)
    }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier
                        .background(accent.copy(alpha = 0.18f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                ) {
                    Text(direction.label, color = accent, fontWeight = FontWeight.SemiBold)
                }
                Spacer(Modifier.width(10.dp))
                Text(
                    "置信度 ${(confidence * 100).toInt()}%",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    "预期 ${if (expectedReturn >= 0) "+" else ""}${"%.2f".format(expectedReturn * 100)}%",
                    style = MaterialTheme.typography.labelSmall,
                    color = colorOf(expectedReturn),
                )
            }
            Spacer(Modifier.height(8.dp))
            Text(summary, style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(8.dp))
            bullets.forEach { Text("• $it", style = MaterialTheme.typography.bodyMedium) }
            if (risks.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "风险提示",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.error,
                )
                risks.forEach {
                    Text("• $it", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (volatility > 0f) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "预测区间年化波动率约 ${(volatility * 100).toInt()}%",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ForecastTable(points: List<com.kronos.assistant.domain.ForecastPoint>) {
    if (points.isEmpty()) return
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Text("预测明细", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(6.dp))
            val fmt = remember { SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()) }
            Row(Modifier.fillMaxWidth()) {
                HeaderCell("时间", 1.2f)
                HeaderCell("开", 1f)
                HeaderCell("高", 1f)
                HeaderCell("低", 1f)
                HeaderCell("收", 1f)
                HeaderCell("10%", 1f)
                HeaderCell("90%", 1f)
            }
            HorizontalDivider(Modifier.padding(vertical = 4.dp))
            points.take(24).forEach { p ->
                Row(Modifier.fillMaxWidth().padding(vertical = 1.dp)) {
                    BodyCell(fmt.format(Date(p.timestamp)), 1.2f)
                    BodyCell(formatPrice(p.open), 1f)
                    BodyCell(formatPrice(p.high), 1f)
                    BodyCell(formatPrice(p.low), 1f)
                    BodyCell(formatPrice(p.close), 1f, color = colorOf(p.close - p.open))
                    BodyCell(formatPrice(p.closeP10), 1f)
                    BodyCell(formatPrice(p.closeP90), 1f)
                }
            }
            if (points.size > 24) {
                Text(
                    "… 共 ${points.size} 个预测点",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}

@Composable
private fun RowScope.HeaderCell(text: String, weight: Float) {
    Text(
        text,
        modifier = Modifier.weight(weight),
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun RowScope.BodyCell(text: String, weight: Float, color: Color = Color.Unspecified) {
    Text(
        text,
        modifier = Modifier.weight(weight),
        style = MaterialTheme.typography.labelSmall,
        fontFamily = FontFamily.Monospace,
        color = color,
    )
}

@Composable
private fun ForecastDetail(params: com.kronos.assistant.domain.ForecastParams, elapsed: Long) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Text("本次推理", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            Text(
                "lookback=${params.lookback}  pred_len=${params.predLen}  " +
                    "sample_count=${params.sampleCount}  T=%.2f  top_p=%.2f".format(params.temperature, params.topP),
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                "全部计算在设备端完成，行情数据不会离开手机。",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

// ======================================================================================
// 自选页
// ======================================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WatchlistScreen(state: AppUiState, vm: AppViewModel, modifier: Modifier = Modifier) {
    var showAddDialog by remember { mutableStateOf(false) }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("自选标的", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            IconButton(onClick = { showAddDialog = true }) {
                Icon(Icons.Filled.Add, contentDescription = "添加")
            }
        }
        Spacer(Modifier.height(8.dp))

        if (state.watchlist.isEmpty()) {
            Text(
                "还没有自选标的，点击右上角 + 添加",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        state.watchlist.forEach { symbol ->
            Card(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(symbol.name, style = MaterialTheme.typography.titleMedium)
                        Text(
                            "${symbol.code} · ${symbol.market.label}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    TextButton(onClick = { vm.selectSymbol(symbol) }) { Text("分析") }
                    IconButton(onClick = { vm.removeFromWatch(symbol) }) {
                        Icon(Icons.Filled.Delete, contentDescription = "移除")
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))
        Text("常用标的", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(6.dp))
        DefaultSymbols.items.filter { candidate -> state.watchlist.none { it.id == candidate.id } }
            .forEach { symbol ->
                Card(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                    Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(symbol.name, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                symbol.code,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        TextButton(onClick = { vm.addToWatch(symbol) }) { Text("加入自选") }
                    }
                }
            }
    }

    if (showAddDialog) {
        AddSymbolDialog(
            onDismiss = { showAddDialog = false },
            onConfirm = { symbol ->
                vm.addToWatch(symbol)
                showAddDialog = false
            },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AddSymbolDialog(onDismiss: () -> Unit, onConfirm: (SymbolInfo) -> Unit) {
    var code by remember { mutableStateOf("") }
    var market by remember { mutableStateOf(Market.CRYPTO) }
    var futuresExchange by remember { mutableStateOf("SHFE") }

    androidx.compose.material3.AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("添加标的") },
        text = {
            Column {
                Row(Modifier.horizontalScroll(rememberScrollState())) {
                    Market.entries.forEach { m ->
                        FilterChip(
                            selected = market == m,
                            onClick = { market = m },
                            label = { Text(m.label) },
                            modifier = Modifier.padding(end = 6.dp),
                        )
                    }
                }
                Spacer(Modifier.height(8.dp))
                if (market == Market.CN_FUTURES) {
                    Row(Modifier.horizontalScroll(rememberScrollState())) {
                        listOf("SHFE", "DCE", "CZCE", "INE", "CFFEX").forEach { ex ->
                            FilterChip(
                                selected = futuresExchange == ex,
                                onClick = { futuresExchange = ex },
                                label = { Text(ex) },
                                modifier = Modifier.padding(end = 6.dp),
                            )
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }
                OutlinedTextField(
                    value = code,
                    onValueChange = { code = it.trim() },
                    label = {
                        Text(
                            when (market) {
                                Market.CRYPTO -> "交易对，如 BTCUSDT"
                                Market.CN_STOCK -> "6 位代码，如 600519"
                                Market.CN_FUTURES -> "合约，如 rb2501"
                                Market.US_STOCK -> "代码，如 AAPL"
                                Market.HK_STOCK -> "5 位代码，如 00700"
                            },
                        )
                    },
                    singleLine = true,
                )
                Text(
                    "提示：国内期货请先选交易所再填合约（例如 SHFE + rb2501）。",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
        },
        confirmButton = {
            TextButton(
                enabled = code.isNotBlank(),
                onClick = {
                    val symbol = when (market) {
                        Market.CRYPTO -> SymbolInfo.crypto(code)
                        Market.CN_STOCK -> SymbolInfo.cn(code)
                        Market.CN_FUTURES -> SymbolInfo.futures("$futuresExchange.$code", code.uppercase())
                        Market.US_STOCK -> SymbolInfo.us(code)
                        Market.HK_STOCK -> SymbolInfo.hk(code)
                    }
                    onConfirm(symbol)
                },
            ) { Text("添加") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

// ======================================================================================
// 回测页
// ======================================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BacktestScreen(state: AppUiState, vm: AppViewModel, modifier: Modifier = Modifier) {
    var predLen by remember { mutableStateOf(state.predLen.coerceAtMost(24)) }
    var step by remember { mutableStateOf(8) }
    var feeBp by remember { mutableStateOf(5f) }
    var thresholdBp by remember { mutableStateOf(10f) }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp)) {
        Text("滚动回测", style = MaterialTheme.typography.titleLarge)
        Text(
            "对该标的做多次滚动预测，按预测收益方向决定持有/空仓，扣除双边手续费后与买入持有对比。",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(10.dp))

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                ParamSlider(
                    "每次预测步数", predLen.toFloat(), 4f..24f, 4f, "$predLen 根",
                ) { predLen = it.toInt() }
                ParamSlider(
                    "滚动步长", step.toFloat(), 4f..24f, 4f, "$step 根",
                ) { step = it.toInt() }
                ParamSlider(
                    "手续费 (单边)", feeBp, 0f..30f, 1f, "%.0f bp".format(feeBp),
                ) { feeBp = it }
                ParamSlider(
                    "建仓阈值", thresholdBp, 0f..100f, 5f, "%.0f bp".format(thresholdBp),
                ) { thresholdBp = it }
            }
        }
        Spacer(Modifier.height(10.dp))

        Button(
            onClick = {
                vm.runBacktest(
                    BacktestParams(
                        lookback = state.effectiveLookback,
                        predLen = predLen,
                        step = step,
                        feeRate = feeBp / 10000f,
                        sampleCount = 1,
                        temperature = 1f,
                        topK = 1,
                        topP = 1f,
                    ),
                )
            },
            enabled = !state.backtestRunning && !state.running && state.modelReady,
            modifier = Modifier.fillMaxWidth().height(48.dp),
        ) {
            Icon(Icons.Filled.Refresh, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text(if (state.backtestRunning) "回测中…" else "开始回测")
        }

        if (state.backtestRunning) {
            Spacer(Modifier.height(8.dp))
            Text(
                "${state.backtestStage ?: ""}  (${state.backtestProgress}/${state.backtestTotal})",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (state.backtestTotal > 0) {
                LinearProgressIndicator(
                    progress = { state.backtestProgress.toFloat() / state.backtestTotal },
                    modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                )
            }
            Text(
                "设备端回测需要为每个窗口跑一次完整推理，请耐心等待。",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
            )
        }

        state.backtestError?.let {
            Spacer(Modifier.height(8.dp))
            ErrorCard(it) { vm.clearBacktest() }
        }

        state.backtest?.let { result ->
            Spacer(Modifier.height(12.dp))
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("净值曲线", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    MultiLineChart(
                        series = listOf(
                            "策略" to (result.points.map { it.strategy } to Color(0xFF64B5F6)),
                            "基准(买入持有)" to (result.points.map { it.benchmark } to ChartColors.Benchmark),
                        ),
                        baseline = 1f,
                        height = 180.dp,
                    )
                }
            }
            Spacer(Modifier.height(8.dp))

            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("绩效指标", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(6.dp))
                    MetricRow("策略总收益", pct(result.metrics.totalReturn), colorOf(result.metrics.totalReturn))
                    MetricRow("基准收益", pct(result.metrics.benchmarkReturn), colorOf(result.metrics.benchmarkReturn))
                    MetricRow("超额收益", pct(result.metrics.totalReturn - result.metrics.benchmarkReturn),
                        colorOf(result.metrics.totalReturn - result.metrics.benchmarkReturn))
                    MetricRow("年化收益", pct(result.metrics.annualizedReturn), colorOf(result.metrics.annualizedReturn))
                    MetricRow("最大回撤", pct(result.metrics.maxDrawdown, negative = true), ChartColors.Bear)
                    MetricRow("夏普比率", "%.2f".format(result.metrics.sharpe), Color.Unspecified)
                    MetricRow("胜率", "${(result.metrics.winRate * 100).toInt()}%", Color.Unspecified)
                    MetricRow("交易次数", "${result.metrics.tradeCount}", Color.Unspecified)
                    MetricRow("持仓占比", "${(result.metrics.exposure * 100).toInt()}%", Color.Unspecified)
                }
            }

            Spacer(Modifier.height(8.dp))
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("交易明细", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(6.dp))
                    if (result.trades.isEmpty()) {
                        Text("本次回测没有触发交易（阈值可能偏高）。", style = MaterialTheme.typography.bodyMedium)
                    } else {
                        val fmt = remember { SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()) }
                        result.trades.takeLast(30).reversed().forEach { t ->
                            Row(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                                Text(fmt.format(Date(t.entryTimestamp)), Modifier.weight(1.2f), style = MaterialTheme.typography.labelSmall)
                                Text(formatPrice(t.entryPrice), Modifier.weight(1f), style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
                                Text(formatPrice(t.exitPrice), Modifier.weight(1f), style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
                                Text(
                                    pct(t.returnPct),
                                    Modifier.weight(1f),
                                    style = MaterialTheme.typography.labelSmall,
                                    fontFamily = FontFamily.Monospace,
                                    color = colorOf(t.returnPct),
                                )
                            }
                        }
                    }
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "演示级回测：未建模滑点、冲击成本、涨跌停与停牌，指标仅供横向对比。",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun MetricRow(label: String, value: String, color: Color) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace,
            color = color,
        )
    }
}

// ======================================================================================
// 设置页
// ======================================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(state: AppUiState, vm: AppViewModel, modifier: Modifier = Modifier) {
    val config = state.indicatorConfig
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp)) {
        Text("显示设置", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                ToggleRow("均线 MA(5/10/20/60)", config.showMa) {
                    vm.setIndicatorConfig(config.copy(showMa = it))
                }
                ToggleRow("布林带 BOLL(20, 2)", config.showBoll) {
                    vm.setIndicatorConfig(config.copy(showBoll = it))
                }
                ToggleRow("MACD(12, 26, 9)", config.showMacd) {
                    vm.setIndicatorConfig(config.copy(showMacd = it))
                }
                ToggleRow("RSI(14)", config.showRsi) {
                    vm.setIndicatorConfig(config.copy(showRsi = it))
                }
            }
        }

        Spacer(Modifier.height(16.dp))
        Text("模型信息", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                InfoRow("状态", when {
                    state.modelError != null -> "加载失败"
                    state.modelReady -> "已就绪"
                    else -> state.modelStatus ?: "未加载"
                })
                InfoRow("模型", "NeoQuasar/Kronos-small (24.7M)")
                InfoRow("分词器", "NeoQuasar/Kronos-Tokenizer-base (4.0M)")
                InfoRow("运行方式", "ONNX Runtime · 设备端 CPU")
                InfoRow("上下文上限", "${AppUiState.MAX_CONTEXT} 根 K 线")
                InfoRow("精度", "FP32")
                if (state.modelError != null) {
                    Spacer(Modifier.height(6.dp))
                    Text(state.modelError, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
                }
            }
        }

        Spacer(Modifier.height(16.dp))
        Text("关于", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                Text(
                    "Kronos 是首个面向金融 K 线的开源基础模型（AAAI 2026），在 45 个交易所、" +
                        "超过 120 亿条 K 线记录上预训练。本应用把官方 PyTorch 推理改写为" +
                        "「预填充 + 增量解码」并导出 ONNX，全部计算在手机本地完成。",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    "免责声明",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.error,
                )
                Text(
                    "本工具的输出来自统计模型的抽样预测，不构成任何投资建议。" +
                        "金融市场价格具有高度随机性，请独立判断并自行承担风险。",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace)
    }
}

// ======================================================================================
// 小工具
// ======================================================================================

private fun colorOf(value: Float): Color =
    when {
        value > 0f -> ChartColors.Bull
        value < 0f -> ChartColors.Bear
        else -> Color.Unspecified
    }

private fun pct(value: Float, negative: Boolean = false): String {
    val v = if (negative) -abs(value) else value
    return "${if (v >= 0) "+" else ""}${"%.2f".format(v * 100)}%"
}

internal fun formatPrice(value: Float): String = when {
    abs(value) >= 1000 -> "%.2f".format(value)
    abs(value) >= 1 -> "%.4f".format(value)
    else -> "%.6f".format(value)
}

internal fun volumeText(value: Float): String = formatCompact(value)
