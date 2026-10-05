package com.kronos.assistant.backtest

import com.kronos.assistant.data.nextTimestamps
import com.kronos.assistant.domain.BacktestMetrics
import com.kronos.assistant.domain.BacktestParams
import com.kronos.assistant.domain.BacktestPoint
import com.kronos.assistant.domain.BacktestResult
import com.kronos.assistant.domain.BacktestTrade
import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.SymbolInfo
import com.kronos.assistant.domain.Timeframe
import java.time.ZoneId
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 滚动回测。
 *
 * 策略逻辑对应官方 `finetune/qlib_test.py` 的思路，但针对**单标的、设备端可跑**做了简化：
 *
 * 1. 从 `lookback` 位置开始，每次向前滚动 `step` 根 K 线；
 * 2. 用截至该点的历史做一次预测，得到未来 `predLen` 根的期望收益；
 * 3. 期望收益 > `entryThreshold` 则持有，否则空仓（多标的场景下这一步等价于
 *    "按预测收益排序取 Top-K 持有"，单标的时 Top-K 退化为全仓/空仓）；
 * 4. 按周期结算，扣双边手续费。
 *
 * 为了让手机上能在可接受时间内跑完，窗口数量用 [maxWindows] 限制，并对齐到等间隔。
 * 这是个**演示级**回测：没有建模滑点、冲击成本、涨跌停与停牌。
 */
class Backtester(private val maxWindows: Int = 24) {

    /** 预测回调: 给定历史 K 线与未来时间戳，返回 `[predLen * 6]`（顺序为 OHLCVA）。 */
    fun interface Predictor {
        suspend fun predict(history: List<Candle>, futureTimestamps: LongArray): FloatArray
    }

    suspend fun run(
        candles: List<Candle>,
        symbol: SymbolInfo,
        timeframe: Timeframe,
        params: BacktestParams,
        zoneId: ZoneId,
        market: com.kronos.assistant.domain.Market,
        entryThreshold: Float = 0.001f,
        predictor: Predictor,
        onProgress: (Int, Int) -> Unit = { _, _ -> },
    ): BacktestResult {
        require(candles.size > params.lookback + params.predLen + params.step) {
            "历史数据不足，无法回测（至少需要 ${params.lookback + params.predLen + params.step + 1} 根 K 线）"
        }

        // 确定评估窗口位置
        val first = params.lookback
        val last = candles.size - params.predLen - 1
        val span = last - first
        if (span <= 0) {
            throw IllegalArgumentException("可用区间不足，请缩短 lookback 或 predLen")
        }
        val stride = maxOf(params.step, span / maxWindows)
        val anchors = ArrayList<Int>()
        var cursor = first
        while (cursor <= last && anchors.size < maxWindows + 1) {
            anchors += cursor
            cursor += stride
        }

        val fee = params.feeRate
        var equity = 1.0
        var benchmark = 1.0
        var peak = 1.0
        var maxDrawdown = 0.0
        var holding = false
        var entryPrice = 0f
        var entryTimestamp = 0L
        var entryEquity = 1.0
        var exposureBars = 0
        var totalBars = 0

        val points = ArrayList<BacktestPoint>(anchors.size + 1)
        val trades = ArrayList<BacktestTrade>()
        val startClose = candles[anchors.first()].close
        points += BacktestPoint(candles[anchors.first()].timestamp, 1f, 1f)

        anchors.forEachIndexed { index, anchor ->
            val history = candles.subList(0, anchor)
            val future = nextTimestamps(
                lastTimestamp = candles[anchor - 1].timestamp,
                timeframe = timeframe,
                count = params.predLen,
                market = market,
                zone = zoneId,
            )
            val forecast = predictor.predict(history, future)
            onProgress(index + 1, anchors.size)

            val predictedReturn = predictedReturnOf(forecast, params.predLen, history.last().close)
            val exitIndex = minOf(anchor + params.step, candles.size - 1)
            val exitClose = candles[exitIndex].close

            totalBars += exitIndex - anchor + 1
            if (holding) exposureBars += exitIndex - anchor + 1

            // ---- 决策: 结算上一段持仓, 再决定下一段 ----
            if (holding) {
                val gross = (exitClose - entryPrice) / entryPrice
                val net = gross - 2 * fee
                equity *= (1.0 + net)
                trades += BacktestTrade(
                    entryTimestamp = entryTimestamp,
                    exitTimestamp = candles[exitIndex].timestamp,
                    entryPrice = entryPrice,
                    exitPrice = exitClose,
                    returnPct = net,
                )
                // 持仓期间净值按资金曲线线性推进
                val markToMarket = equity * (1.0 + net * 0.0)
                pointUpdate(points, candles, anchor, exitIndex, markToMarket)
                holding = false
            }

            // 基准: 始终买入持有
            benchmark = (exitClose / startClose).toDouble()

            if (predictedReturn > entryThreshold) {
                holding = true
                entryPrice = candles[anchor].close
                entryTimestamp = candles[anchor].timestamp
                entryEquity = equity
                // 建仓成本
                equity *= (1.0 - fee)
            }

            val currentPeak = maxOf(peak, equity)
            peak = currentPeak
            val drawdown = if (peak > 0) (peak - equity) / peak else 0.0
            if (drawdown > maxDrawdown) maxDrawdown = drawdown
            points += BacktestPoint(candles[exitIndex].timestamp, equity.toFloat(), benchmark.toFloat())
        }

        // 收尾: 仍持仓则按最后价格平仓
        if (holding) {
            val exitClose = candles.last().close
            val net = (exitClose - entryPrice) / entryPrice - 2 * fee
            equity *= (1.0 + net)
            trades += BacktestTrade(
                entryTimestamp = entryTimestamp,
                exitTimestamp = candles.last().timestamp,
                entryPrice = entryPrice,
                exitPrice = exitClose,
                returnPct = net,
            )
        }

        val lastAnchor = anchors.last()
        val benchmarkReturn = (candles.last().close - candles[lastAnchor].close) / candles[lastAnchor].close
        val totalReturn = (equity - 1.0).toFloat()

        val periods = anchors.size
        val years = periods * stride * timeframe.minutes / (365.0 * 24 * 60)
        val annualized = if (years > 1e-6 && equity > 0) {
            (Math.pow(equity, 1.0 / years) - 1.0).toFloat()
        } else {
            totalReturn
        }

        val wins = trades.count { it.returnPct > 0 }
        val winRate = if (trades.isEmpty()) 0f else wins.toFloat() / trades.size

        // 用逐段收益率估算夏普（无风险利率按 0 处理）
        val returns = trades.map { it.returnPct.toDouble() }
        val sharpe = if (returns.size > 1) {
            val mean = returns.average()
            val variance = returns.sumOf { (it - mean) * (it - mean) } / (returns.size - 1)
            val sd = sqrt(variance)
            if (sd > 1e-9) (mean / sd * sqrt(252.0 / maxOf(1, params.predLen / maxOf(1, timeframe.minutes / 1440)))).toFloat() else 0f
        } else {
            0f
        }

        return BacktestResult(
            symbol = symbol,
            timeframe = timeframe,
            points = points,
            trades = trades,
            metrics = BacktestMetrics(
                totalReturn = totalReturn,
                benchmarkReturn = benchmarkReturn,
                annualizedReturn = annualized,
                maxDrawdown = maxDrawdown.toFloat(),
                sharpe = sharpe,
                winRate = winRate,
                tradeCount = trades.size,
                exposure = if (totalBars > 0) exposureBars.toFloat() / totalBars else 0f,
            ),
            params = params,
        )
    }

    private fun pointUpdate(
        points: MutableList<BacktestPoint>,
        candles: List<Candle>,
        from: Int,
        to: Int,
        equity: Double,
    ) {
        // 持仓段用等间隔采样展示净值变化，避免曲线出现断崖
        val steps = (to - from).coerceAtLeast(1)
        for (i in 1..steps) {
            val idx = (from + i).coerceAtMost(candles.size - 1)
            points += BacktestPoint(candles[idx].timestamp, equity.toFloat(), equity.toFloat())
        }
    }

    /** 从预测矩阵取"末点相对当前价的期望收益"。 */
    private fun predictedReturnOf(forecast: FloatArray, predLen: Int, lastClose: Float): Float {
        if (lastClose <= 0f || forecast.size < predLen * 6) return 0f
        val predictedClose = forecast[(predLen - 1) * 6 + 3]
        return (predictedClose - lastClose) / lastClose
    }
}
