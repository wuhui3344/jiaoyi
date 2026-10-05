package com.kronos.assistant.domain

import kotlinx.serialization.Serializable

/**
 * 市场类型。决定交易日历与所使用的行情数据源。
 */
enum class Market(val label: String) {
    CRYPTO("加密货币"),
    CN_STOCK("A 股"),
    CN_FUTURES("国内期货"),
    US_STOCK("美股"),
    HK_STOCK("港股"),
    ;

    /** 是否为 7x24 连续交易 (只有加密货币是)。 */
    val isContinuous: Boolean get() = this == CRYPTO
}

/**
 * K 线周期。
 *
 * [minutes] 用于生成未来时间戳 (预测区间), [millis] 为周期毫秒数。
 */
enum class Timeframe(
    val id: String,
    val label: String,
    val minutes: Int,
) {
    M1("1m", "1 分钟", 1),
    M5("5m", "5 分钟", 5),
    M15("15m", "15 分钟", 15),
    M30("30m", "30 分钟", 30),
    H1("1h", "1 小时", 60),
    H4("4h", "4 小时", 240),
    D1("1d", "日线", 1440),
    W1("1w", "周线", 10080),
    ;

    val millis: Long get() = minutes * 60_000L

    /** 是否为日内周期 (决定时间戳生成时是否需要跳过非交易时段)。 */
    val isIntraday: Boolean get() = minutes < 1440

    companion object {
        fun fromId(id: String): Timeframe = entries.firstOrNull { it.id == id } ?: M5

        /** 训练分布最接近的一批周期, 也是 App 默认展示的选项。 */
        val recommended: List<Timeframe> = listOf(M5, M15, M30, H1, H4, D1)
    }
}

/**
 * 单根 K 线。
 *
 * [amount] 为成交额; 若数据源未提供, 按 `volume * mean(open, high, low, close)` 估算,
 * 与官方 `KronosPredictor.predict` 的处理保持一致。
 */
@Serializable
data class Candle(
    val timestamp: Long,
    val open: Float,
    val high: Float,
    val low: Float,
    val close: Float,
    val volume: Float,
    val amount: Float,
) {
    val typicalPrice: Float get() = (high + low + close) / 3f
    val isBullish: Boolean get() = close >= open
}

/** 规范化后的标的标识。 */
@Serializable
data class SymbolInfo(
    val code: String,
    val name: String,
    val market: Market,
) {
    /** 在自选列表中使用的稳定 id。 */
    val id: String get() = "${market.name}:$code"

    companion object {
        fun crypto(pair: String, name: String = pair): SymbolInfo {
            val pretty = pair.replace("USDT", "/USDT").replace("USDC", "/USDC")
            return SymbolInfo(pair.uppercase(), name.ifBlank { pretty }, Market.CRYPTO)
        }

        fun cn(code: String, name: String = code) = SymbolInfo(code, name, Market.CN_STOCK)
        fun futures(code: String, name: String = code) = SymbolInfo(code, name, Market.CN_FUTURES)
        fun us(code: String, name: String = code) = SymbolInfo(code.uppercase(), name, Market.US_STOCK)
        fun hk(code: String, name: String = code) = SymbolInfo(code, name, Market.HK_STOCK)
    }
}

/** 完整的一段行情序列。 */
data class CandleSeries(
    val symbol: SymbolInfo,
    val timeframe: Timeframe,
    val candles: List<Candle>,
) {
    val isEmpty: Boolean get() = candles.isEmpty()
    val last: Candle? get() = candles.lastOrNull()
}

// --------------------------------------------------------------------------------------
// 预测
// --------------------------------------------------------------------------------------

/** 单条预测路径上的一个点 (反归一化后)。 */
data class ForecastPoint(
    val timestamp: Long,
    val open: Float,
    val high: Float,
    val low: Float,
    val close: Float,
    val volume: Float,
    val amount: Float,
    /** 多路径 close 的 10% / 90% 分位数, `sampleCount == 1` 时等于 close。 */
    val closeP10: Float,
    val closeP90: Float,
    /** 多路径 close 的标准差, 作为不确定度代理。 */
    val closeStd: Float,
) {
    fun toCandle() = Candle(timestamp, open, high, low, close, volume, amount)
}

/** 一次预测所使用的参数快照, 便于复现。 */
@Serializable
data class ForecastParams(
    val lookback: Int,
    val predLen: Int,
    val sampleCount: Int,
    val temperature: Float,
    val topK: Int,
    val topP: Float,
    val seed: Long,
)

/** 一次完整预测的结果。 */
data class ForecastResult(
    val symbol: SymbolInfo,
    val timeframe: Timeframe,
    /** 用作条件的历史 K 线 (已按 lookback 截断)。 */
    val history: List<Candle>,
    /** 预测点, 长度为 predLen。 */
    val points: List<ForecastPoint>,
    /** 全部采样路径, 形状 `[sampleCount][predLen]`。 */
    val paths: List<List<ForecastPoint>>,
    val params: ForecastParams,
    val elapsedMillis: Long,
) {
    /** 相对最后一根历史收盘价的预期收益率。 */
    val expectedReturn: Float
        get() {
            val last = history.lastOrNull()?.close ?: return 0f
            val target = points.lastOrNull()?.close ?: return 0f
            if (last <= 0f) return 0f
            return (target - last) / last
        }

    val lastClose: Float get() = history.lastOrNull()?.close ?: 0f
}

/** 推理过程中上报的进度。 */
sealed interface ForecastProgress {
    /** 已加载模型。 */
    data class ModelReady(val loadMillis: Long) : ForecastProgress

    /** 已取到行情。 */
    data class DataReady(val bars: Int) : ForecastProgress

    /** 正在做上下文编码 / 预填充。 */
    data object Prefill : ForecastProgress

    /** 自回归解码进度。 */
    data class Decoding(val step: Int, val total: Int) : ForecastProgress

    /** 全部完成。 */
    data class Done(val elapsedMillis: Long) : ForecastProgress
}
