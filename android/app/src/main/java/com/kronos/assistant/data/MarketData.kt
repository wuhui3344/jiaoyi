package com.kronos.assistant.data

import com.kronos.assistant.domain.Candle
import com.kronos.assistant.domain.Market
import com.kronos.assistant.domain.SymbolInfo
import com.kronos.assistant.domain.Timeframe
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit
import kotlin.math.roundToLong

/** 行情获取失败。 */
class MarketDataException(message: String, cause: Throwable? = null) : IOException(message, cause)

/** 统一的数据源接口。实现必须是线程安全的（会在 IO 线程池上被并发调用）。 */
interface MarketDataSource {
    /** 该数据源支持的标的列表，用于「发现」页面。 */
    fun supportedMarkets(): Set<Market>

    /** 拉取最近的 [limit] 根 K 线，按时间升序返回。 */
    fun fetchCandles(symbol: SymbolInfo, timeframe: Timeframe, limit: Int): List<Candle>
}

/** 共享的 HTTP 客户端。 */
internal object Http {
    val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .callTimeout(30, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    fun getJson(url: String): String {
        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "Mozilla/5.0 (Android) KronosAssistant/1.0")
            .header("Accept", "application/json, text/plain, */*")
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw MarketDataException("HTTP ${response.code} — ${response.request.url.encodedPath}")
            }
            return response.body?.string() ?: throw MarketDataException("响应为空")
        }
    }

    fun getText(url: String): String = getJson(url)
}

// --------------------------------------------------------------------------------------
// 加密货币: Binance
// --------------------------------------------------------------------------------------

/**
 * Binance 现货 K 线。
 *
 * `GET /api/v3/klines` 返回 `[开盘时间, 开, 高, 低, 收, 成交量, 收盘时间, 成交额, ...]`，
 * 其中成交额（quote asset volume）正好对应 Kronos 需要的 `amount` 字段。
 */
class BinanceDataSource : MarketDataSource {

    override fun supportedMarkets() = setOf(Market.CRYPTO)

    override fun fetchCandles(symbol: SymbolInfo, timeframe: Timeframe, limit: Int): List<Candle> {
        if (timeframe == Timeframe.H4) {
            // Binance 现货不提供 4h，最多 1h；用 4 根 1h 合成
            return aggregate(fetchCandles(symbol, Timeframe.H1, limit * 4), Timeframe.H4, limit)
        }
        val interval = binanceInterval(timeframe) ?: throw MarketDataException("Binance 不支持周期 ${timeframe.id}")
        val url = "https://api.binance.com/api/v3/klines" +
            "?symbol=${symbol.code}&interval=$interval&limit=${limit.coerceIn(1, 1000)}"
        val array = JSONArray(Http.getText(url))
        val out = ArrayList<Candle>(array.length())
        for (i in 0 until array.length()) {
            val row = array.getJSONArray(i)
            out += Candle(
                timestamp = row.getLong(0),
                open = row.getDouble(1).toFloat(),
                high = row.getDouble(2).toFloat(),
                low = row.getDouble(3).toFloat(),
                close = row.getDouble(4).toFloat(),
                volume = row.getDouble(5).toFloat(),
                amount = row.optDouble(7, Double.NaN).let { if (it.isNaN()) 0f else it.toFloat() },
            )
        }
        return out
    }

    private fun binanceInterval(tf: Timeframe) = when (tf) {
        Timeframe.M1 -> "1m"
        Timeframe.M5 -> "5m"
        Timeframe.M15 -> "15m"
        Timeframe.M30 -> "30m"
        Timeframe.H1 -> "1h"
        Timeframe.D1 -> "1d"
        Timeframe.W1 -> "1w"
        else -> null
    }

    /** 把细粒度 K 线合成为粗粒度（用于 4h 等 Binance 现货不提供的周期）。 */
    private fun aggregate(source: List<Candle>, target: Timeframe, limit: Int): List<Candle> {
        if (source.isEmpty()) return emptyList()
        val bucketMillis = target.millis
        val grouped = source.groupBy { it.timestamp / bucketMillis * bucketMillis }
        return grouped.entries.sortedBy { it.key }.takeLast(limit).map { (bucket, group) ->
            Candle(
                timestamp = bucket,
                open = group.first().open,
                high = group.maxOf { it.high },
                low = group.minOf { it.low },
                close = group.last().close,
                volume = group.sumOf { it.volume.toDouble() }.toFloat(),
                amount = group.sumOf { it.amount.toDouble() }.toFloat(),
            )
        }
    }
}

// --------------------------------------------------------------------------------------
// A 股 / 港股 / 国内期货: 东方财富
// --------------------------------------------------------------------------------------

/**
 * 东方财富 K 线接口，覆盖 A 股、港股与国内期货。
 *
 * 期货的 `secid` 前缀按交易所区分：上期所 113、大商所 114、郑商所 115、
 * 上海国际能源交易中心 142、中金所 8。
 * 返回的 `klines` 每行是 `时间,开,收,高,低,成交量,成交额,...`
 * —— 注意**第 2、3 位是「收」在前**，与 Binance 的 `开高收低` 不同。
 */
class EastMoneyDataSource : MarketDataSource {

    override fun supportedMarkets() = setOf(Market.CN_STOCK, Market.CN_FUTURES, Market.HK_STOCK)

    override fun fetchCandles(symbol: SymbolInfo, timeframe: Timeframe, limit: Int): List<Candle> {
        val secid = secId(symbol) ?: throw MarketDataException("无法识别标的代码: ${symbol.code}")
        val klt = kltCode(timeframe) ?: throw MarketDataException("东方财富不支持周期 ${timeframe.id}")
        val url = "https://push2his.eastmoney.com/api/qt/stock/kline/get" +
            "?secid=$secid&klt=$klt&fqt=1" +
            "&fields1=f1,f2,f3,f4,f5,f6" +
            "&fields2=f51,f52,f53,f54,f55,f56,f57" +
            "&beg=0&end=20500101&lmt=${limit.coerceIn(1, 2000)}"

        val root = JSONObject(Http.getText(url))
        val data = root.optJSONObject("data") ?: throw MarketDataException("无数据: ${symbol.code}")
        val klines = data.optJSONArray("klines") ?: throw MarketDataException("无 K 线数据: ${symbol.code}")
        if (klines.length() == 0) throw MarketDataException("K 线为空: ${symbol.code}")

        val zone = timezoneFor(symbol, data.optString("market"))
        val out = ArrayList<Candle>(klines.length())
        for (i in 0 until klines.length()) {
            val parts = klines.getString(i).split(',')
            if (parts.size < 7) continue
            out += Candle(
                timestamp = parseLocalTime(parts[0], zone, timeframe),
                open = parts[1].toFloat(),
                close = parts[2].toFloat(),
                high = parts[3].toFloat(),
                low = parts[4].toFloat(),
                volume = parts[5].toFloat(),
                amount = parts[6].toFloat(),
            )
        }
        return out
    }

    private fun secId(symbol: SymbolInfo): String? {
        val code = symbol.code.uppercase()
        return when (symbol.market) {
            Market.CN_STOCK -> {
                val digits = code.filter { it.isDigit() }
                if (digits.length != 6) return null
                // 6 开头为沪市，其余为深市（含创业板 3、中小板 0、北交所 8/4）
                val prefix = if (digits.startsWith("6") || digits.startsWith("5")) "1" else "0"
                "$prefix.$digits"
            }
            Market.HK_STOCK -> {
                val digits = code.filter { it.isDigit() }.padStart(5, '0')
                "116.$digits"
            }
            Market.CN_FUTURES -> {
                val m = FUTURES_PREFIX.matchEntire(code) ?: return null
                val exchange = when (m.groupValues[1]) {
                    "SHFE" -> "113"
                    "DCE" -> "114"
                    "CZCE" -> "115"
                    "INE" -> "142"
                    "CFFEX" -> "8"
                    else -> "113"
                }
                "$exchange.${m.groupValues[2].lowercase()}"
            }
            else -> null
        }
    }

    private fun kltCode(tf: Timeframe) = when (tf) {
        Timeframe.M1 -> "1"
        Timeframe.M5 -> "5"
        Timeframe.M15 -> "15"
        Timeframe.M30 -> "30"
        Timeframe.H1 -> "60"
        Timeframe.H4 -> "60" // 东方财富无 4h，用 1h 代替（调用方需注意）
        Timeframe.D1 -> "101"
        Timeframe.W1 -> "102"
    }

    private fun timezoneFor(symbol: SymbolInfo, marketCode: String): ZoneId = when {
        symbol.market == Market.HK_STOCK -> ZoneId.of("Asia/Hong_Kong")
        marketCode.toIntOrNull() == 116 -> ZoneId.of("Asia/Hong_Kong")
        else -> ZoneId.of("Asia/Shanghai")
    }

    private fun parseLocalTime(text: String, zone: ZoneId, tf: Timeframe): Long {
        val trimmed = text.trim()
        val formatter = if (tf.isIntraday) INTRADAY_FORMAT else DAY_FORMAT
        val parsed = LocalDateTime.parse(if (trimmed.length <= 10) "$trimmed 00:00" else trimmed, formatter)
        return parsed.atZone(zone).toInstant().toEpochMilli()
    }

    private companion object {
        val INTRADAY_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")
        val DAY_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")
        val FUTURES_PREFIX = Regex("(SHFE|DCE|CZCE|INE|CFFEX)[.:]?([A-Za-z]{1,2}\\d{3,4})")
    }
}

// --------------------------------------------------------------------------------------
// 美股: Yahoo Finance
// --------------------------------------------------------------------------------------

/**
 * Yahoo Finance chart 接口。
 *
 * 注意 Yahoo 不返回成交额，因此按 `volume × 典型价` 估算，这与官方
 * `KronosPredictor.predict` 在缺少 `amount` 列时的处理一致。
 */
class YahooDataSource : MarketDataSource {

    override fun supportedMarkets() = setOf(Market.US_STOCK)

    override fun fetchCandles(symbol: SymbolInfo, timeframe: Timeframe, limit: Int): List<Candle> {
        val interval = yahooInterval(timeframe) ?: throw MarketDataException("Yahoo 不支持周期 ${timeframe.id}")
        val range = yahooRange(timeframe, limit)
        val url = "https://query1.finance.yahoo.com/v8/finance/chart/" +
            "${symbol.code}?interval=$interval&range=$range&includePrePost=false"

        val result = JSONObject(Http.getText(url))
            .getJSONObject("chart")
            .optJSONArray("result")
            ?.optJSONObject(0)
            ?: throw MarketDataException("Yahoo 无数据: ${symbol.code}")

        val timestamps = result.getJSONArray("timestamp")
        val quote = result.getJSONObject("indicators").getJSONArray("quote").getJSONObject(0)
        val opens = quote.getJSONArray("open")
        val highs = quote.getJSONArray("high")
        val lows = quote.getJSONArray("low")
        val closes = quote.getJSONArray("close")
        val volumes = quote.getJSONArray("volume")

        val out = ArrayList<Candle>(timestamps.length())
        for (i in 0 until timestamps.length()) {
            if (closes.isNull(i) || opens.isNull(i)) continue
            val volume = if (volumes.isNull(i)) 0f else volumes.getDouble(i).toFloat()
            val close = closes.getDouble(i).toFloat()
            val high = if (highs.isNull(i)) close else highs.getDouble(i).toFloat()
            val low = if (lows.isNull(i)) close else lows.getDouble(i).toFloat()
            val open = opens.getDouble(i).toFloat()
            out += Candle(
                timestamp = timestamps.getLong(i) * 1000L,
                open = open,
                high = high,
                low = low,
                close = close,
                volume = volume,
                amount = volume * ((open + high + low + close) / 4f),
            )
        }
        return out.takeLast(limit)
    }

    private fun yahooInterval(tf: Timeframe) = when (tf) {
        Timeframe.M1 -> "1m"
        Timeframe.M5 -> "5m"
        Timeframe.M15 -> "15m"
        Timeframe.M30 -> "30m"
        Timeframe.H1 -> "1h"
        Timeframe.D1 -> "1d"
        Timeframe.W1 -> "1wk"
        else -> null
    }

    private fun yahooRange(tf: Timeframe, limit: Int): String = when (tf) {
        Timeframe.M1 -> "7d"
        Timeframe.M5, Timeframe.M15, Timeframe.M30 -> "1mo"
        Timeframe.H1 -> "3mo"
        Timeframe.D1 -> if (limit > 400) "5y" else "2y"
        else -> "5y"
    }
}

// --------------------------------------------------------------------------------------
// 仓库: 按 market 路由
// --------------------------------------------------------------------------------------

class MarketDataRepository(
    private val sources: List<MarketDataSource> = listOf(
        BinanceDataSource(),
        EastMoneyDataSource(),
        YahooDataSource(),
    ),
) {
    init {
        // 断言每个市场都有数据源覆盖，配置错误时尽早暴露
        val covered = sources.flatMap { it.supportedMarkets() }.toSet()
        Market.entries.forEach { market ->
            require(market in covered) { "市场 $market 没有对应的数据源" }
        }
    }

    private fun sourceFor(market: Market): MarketDataSource =
        sources.firstOrNull { market in it.supportedMarkets() }
            ?: throw MarketDataException("市场 $market 没有可用数据源")

    fun candles(symbol: SymbolInfo, timeframe: Timeframe, limit: Int): List<Candle> {
        val source = sourceFor(symbol.market)
        val candles = source.fetchCandles(symbol, timeframe, limit)
        if (candles.isEmpty()) throw MarketDataException("${symbol.code} 未返回任何 K 线")
        return candles
    }

    /** 交易所使用的时区，决定时间特征与未来时间戳的生成方式。 */
    fun timezoneFor(symbol: SymbolInfo): ZoneId = when (symbol.market) {
        Market.CRYPTO -> ZoneId.of("UTC")
        Market.CN_STOCK, Market.CN_FUTURES -> ZoneId.of("Asia/Shanghai")
        Market.HK_STOCK -> ZoneId.of("Asia/Hong_Kong")
        Market.US_STOCK -> ZoneId.of("America/New_York")
    }
}

/** 常用标的，用于首次启动时的「发现」列表。 */
object DefaultSymbols {
    val items: List<SymbolInfo> = listOf(
        SymbolInfo.crypto("BTCUSDT"),
        SymbolInfo.crypto("ETHUSDT"),
        SymbolInfo.crypto("SOLUSDT"),
        SymbolInfo.crypto("BNBUSDT"),
        SymbolInfo.cn("600519", "贵州茅台"),
        SymbolInfo.cn("000001", "平安银行"),
        SymbolInfo.cn("300750", "宁德时代"),
        SymbolInfo.cn("601318", "中国平安"),
        SymbolInfo.futures("SHFE.rb2501", "螺纹钢 2501"),
        SymbolInfo.futures("SHFE.cu2501", "沪铜 2501"),
        SymbolInfo.futures("DCE.i2501", "铁矿石 2501"),
        SymbolInfo.futures("CZCE.MA501", "甲醇 501"),
        SymbolInfo.us("AAPL", "苹果"),
        SymbolInfo.us("NVDA", "英伟达"),
        SymbolInfo.us("TSLA", "特斯拉"),
        SymbolInfo.hk("00700", "腾讯控股"),
        SymbolInfo.hk("09988", "阿里巴巴-W"),
    )
}

/** 把成交额缺失的 K 线补齐，规则与官方预测器一致。 */
fun List<Candle>.ensureAmount(): List<Candle> =
    map { if (it.amount > 0f) it else it.copy(amount = it.volume * (it.open + it.high + it.low + it.close) / 4f) }

/** 给定时区下的未来时间戳序列（用于预测区间）。 */
fun nextTimestamps(lastTimestamp: Long, timeframe: Timeframe, count: Int, market: Market, zone: ZoneId): LongArray {
    val out = LongArray(count)
    if (market.isContinuous) {
        for (i in 1..count) out[i - 1] = lastTimestamp + timeframe.millis * i
        return out
    }
    // 非 7x24 市场: 按交易日推进，跳过周末与收盘时段
    var cursor = LocalDateTime.ofInstant(java.time.Instant.ofEpochMilli(lastTimestamp), zone)
    var produced = 0
    var guard = 0
    while (produced < count && guard++ < count * 20) {
        cursor = cursor.plusMinutes(timeframe.minutes.toLong())
        if (cursor.dayOfWeek.value >= 6) {           // 周六/周日 -> 跳到周一开盘
            cursor = cursor.plusDays((8 - cursor.dayOfWeek.value).toLong())
            cursor = cursor.withHour(9).withMinute(30)
        }
        if (timeframe.isIntraday && !inTradingHours(cursor, market)) continue
        out[produced++] = cursor.atZone(zone).toInstant().toEpochMilli()
    }
    // 若交易日历推算不足，退化到固定间隔，保证长度正确
    while (produced < count) {
        out[produced] = (out[maxOf(0, produced - 1)] + timeframe.millis).let {
            if (produced == 0) lastTimestamp + timeframe.millis else it
        }
        produced++
    }
    return out
}

private fun inTradingHours(t: LocalDateTime, market: Market): Boolean {
    val minutes = t.hour * 60 + t.minute
    return when (market) {
        Market.CN_STOCK -> (570..690).contains(minutes) || (780..900).contains(minutes)
        Market.HK_STOCK -> (570..720).contains(minutes) || (780..960).contains(minutes)
        Market.US_STOCK -> (570..960).contains(minutes)
        Market.CN_FUTURES -> minutes >= 540 || minutes <= 150 || (630..690).contains(minutes) || (780..900).contains(minutes)
        Market.CRYPTO -> true
    }
}

/** 供 UI 展示的金额格式化。 */
fun formatCompact(value: Float): String {
    val abs = kotlin.math.abs(value)
    val suffix: String
    val scaled: Double
    when {
        abs >= 1e12 -> { scaled = value / 1e12; suffix = "万亿" }
        abs >= 1e8 -> { scaled = value / 1e8; suffix = "亿" }
        abs >= 1e4 -> { scaled = value / 1e4; suffix = "万" }
        else -> { scaled = value.toDouble(); suffix = "" }
    }
    val decimals = if (scaled.roundToLong().toDouble() == scaled) 0 else 2
    return String.format("%.${decimals}f%s", scaled, suffix)
}
