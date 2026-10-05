package com.kronos.assistant.domain

/**
 * 技术指标数据模型。
 *
 * 设计目标: 用同一套结构同时表达「叠加在主图上的指标」(MA / BOLL) 和
 * 「独立副图指标」(MACD / RSI / KDJ), 避免为每个指标都写一遍 UI。
 */

/** 一条曲线。`values` 与 K 线一一对应, 暖机期用 `null` 占位。 */
data class IndicatorSeries(
    val name: String,
    val colorArgb: Long,
    val values: List<Float?>,
    val dashed: Boolean = false,
)

/** 副图指标面板。 */
data class IndicatorPanel(
    val name: String,
    val series: List<IndicatorSeries>,
    /** 若不为空, 则按柱状图绘制 (如 MACD 柱)。 */
    val histogram: List<Float?>? = null,
    val histogramName: String = "hist",
    /** 是否绘制 0 轴参考线。 */
    val zeroLine: Boolean = false,
    /** 固定上下界 (如 RSI 的 0~100)。为 null 时自适应。 */
    val fixedRange: ClosedFloatingPointRange<Float>? = null,
)

/** 一次计算得到的全部指标。 */
data class IndicatorBundle(
    val overlays: List<IndicatorSeries>,
    val panels: List<IndicatorPanel>,
)

/** 指标参数。 */
data class IndicatorConfig(
    val maPeriods: List<Int> = listOf(5, 10, 20, 60),
    val bollPeriod: Int = 20,
    val bollK: Float = 2f,
    val macdFast: Int = 12,
    val macdSlow: Int = 26,
    val macdSignal: Int = 9,
    val rsiPeriod: Int = 14,
    val kdjPeriod: Int = 9,
    val showMa: Boolean = true,
    val showBoll: Boolean = false,
    val showMacd: Boolean = true,
    val showRsi: Boolean = true,
)

// --------------------------------------------------------------------------------------
// 回测
// --------------------------------------------------------------------------------------

/** 回测中的一笔交易。 */
data class BacktestTrade(
    val entryTimestamp: Long,
    val exitTimestamp: Long,
    val entryPrice: Float,
    val exitPrice: Float,
    /** 收益率 (已扣除手续费)。 */
    val returnPct: Float,
)

/** 回测净值曲线的采样点。 */
data class BacktestPoint(
    val timestamp: Long,
    val strategy: Float,
    val benchmark: Float,
)

/** 回测指标。 */
data class BacktestMetrics(
    val totalReturn: Float,
    val benchmarkReturn: Float,
    val annualizedReturn: Float,
    val maxDrawdown: Float,
    val sharpe: Float,
    val winRate: Float,
    val tradeCount: Int,
    val exposure: Float,
)

/** 一次回测的完整结果。 */
data class BacktestResult(
    val symbol: SymbolInfo,
    val timeframe: Timeframe,
    val points: List<BacktestPoint>,
    val trades: List<BacktestTrade>,
    val metrics: BacktestMetrics,
    val params: BacktestParams,
)

/** 回测配置。 */
data class BacktestParams(
    /** 每次预测使用的历史长度。 */
    val lookback: Int = 256,
    /** 预测步数, 也是持仓天数。 */
    val predLen: Int = 8,
    /** 选取得分最高的前 K 个标的分批建仓; 单标的回测时该值恒为 1。 */
    val topK: Int = 1,
    /** 单边手续费 (万分之几 -> 小数)。 */
    val feeRate: Float = 0.0005f,
    /** 每次滚动前进的步数。 */
    val step: Int = 8,
    val sampleCount: Int = 1,
    val temperature: Float = 1f,
    val topP: Float = 0.9f,
)

// --------------------------------------------------------------------------------------
// 文字解读
// --------------------------------------------------------------------------------------

enum class Direction(val label: String) {
    UP("偏多"),
    DOWN("偏空"),
    NEUTRAL("震荡"),
    ;

    companion object {
        fun of(expectedReturn: Float, threshold: Float = 0.002f): Direction = when {
            expectedReturn > threshold -> UP
            expectedReturn < -threshold -> DOWN
            else -> NEUTRAL
        }
    }
}

/** 模型给出的结构化研判。 */
data class Insight(
    val direction: Direction,
    /** 0~1, 综合路径一致性 / 波动 / 趋势给出。 */
    val confidence: Float,
    val expectedReturn: Float,
    /** 预测区间内的年化波动率估计。 */
    val volatility: Float,
    val summary: String,
    val bullets: List<String>,
    val risks: List<String>,
)
