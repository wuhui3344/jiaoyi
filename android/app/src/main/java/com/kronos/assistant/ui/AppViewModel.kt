package com.kronos.assistant.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.kronos.assistant.KronosApplication
import com.kronos.assistant.analysis.AnalysisReport
import com.kronos.assistant.analysis.AnalysisService
import com.kronos.assistant.backtest.Backtester
import com.kronos.assistant.data.DefaultSymbols
import com.kronos.assistant.domain.BacktestParams
import com.kronos.assistant.domain.BacktestResult
import com.kronos.assistant.domain.IndicatorConfig
import com.kronos.assistant.domain.Market
import com.kronos.assistant.domain.SymbolInfo
import com.kronos.assistant.domain.Timeframe
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** 全应用共享的界面状态。 */
data class AppUiState(
    val modelReady: Boolean = false,
    val modelLoading: Boolean = false,
    val modelStatus: String? = null,
    val modelError: String? = null,

    val symbol: SymbolInfo = DefaultSymbols.items.first(),
    val timeframe: Timeframe = Timeframe.H1,
    val lookback: Int = 256,
    val predLen: Int = 48,
    val sampleCount: Int = 3,
    val temperature: Float = 1.0f,
    val topP: Float = 0.9f,

    val indicatorConfig: IndicatorConfig = IndicatorConfig(),
    val showBand: Boolean = true,
    val showPaths: Boolean = true,
    val visibleBars: Int = 120,

    val running: Boolean = false,
    val stage: String? = null,
    val step: Int = 0,
    val totalSteps: Int = 0,
    val error: String? = null,
    val report: AnalysisReport? = null,

    val watchlist: List<SymbolInfo> = emptyList(),

    val backtestRunning: Boolean = false,
    val backtestStage: String? = null,
    val backtestProgress: Int = 0,
    val backtestTotal: Int = 0,
    val backtest: BacktestResult? = null,
    val backtestError: String? = null,
) {
    /** 模型上下文约束下的实际历史长度。 */
    val effectiveLookback: Int get() = minOf(lookback, MAX_CONTEXT - predLen)

    companion object {
        /** Kronos-small 的 max_context。UI 侧用它做提示，真实约束以 manifest 为准。 */
        const val MAX_CONTEXT = 512
    }
}

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application as KronosApplication
    private val service = AnalysisService(app.marketData) { app.engine() }
    private val backtester = Backtester()
    private val runMutex = Mutex()

    private val _state = MutableStateFlow(AppUiState())
    val state: StateFlow<AppUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            app.watchlist.entries.collect { list -> _state.update { it.copy(watchlist = list) } }
        }
        viewModelScope.launch {
            _state.update { it.copy(modelLoading = true, modelStatus = "正在准备 Kronos 模型…") }
            runCatching { app.engine() }
                .onSuccess { _state.update { s -> s.copy(modelReady = true, modelLoading = false, modelStatus = null) } }
                .onFailure { e ->
                    _state.update {
                        it.copy(
                            modelLoading = false,
                            modelReady = false,
                            modelError = e.message ?: "模型加载失败",
                            modelStatus = null,
                        )
                    }
                }
        }
    }

    // ----------------------------------------------------------------------------------
    // 参数修改
    // ----------------------------------------------------------------------------------

    fun selectSymbol(symbol: SymbolInfo) = _state.update { it.copy(symbol = symbol, report = null, error = null) }

    fun setTimeframe(tf: Timeframe) = _state.update { it.copy(timeframe = tf, report = null) }

    fun setLookback(value: Int) = _state.update { it.copy(lookback = value.coerceIn(64, AppUiState.MAX_CONTEXT / 2)) }

    fun setPredLen(value: Int) = _state.update { it.copy(predLen = value.coerceIn(4, 120)) }

    fun setSampleCount(value: Int) = _state.update { it.copy(sampleCount = value.coerceIn(1, 10)) }

    fun setTemperature(value: Float) = _state.update { it.copy(temperature = value.coerceIn(0.1f, 2f)) }

    fun setTopP(value: Float) = _state.update { it.copy(topP = value.coerceIn(0.1f, 1f)) }

    fun setShowBand(value: Boolean) = _state.update { it.copy(showBand = value) }

    fun setShowPaths(value: Boolean) = _state.update { it.copy(showPaths = value) }

    fun setVisibleBars(value: Int) = _state.update { it.copy(visibleBars = value.coerceIn(40, 400)) }

    fun setIndicatorConfig(config: IndicatorConfig) = _state.update { it.copy(indicatorConfig = config) }

    fun toggleWatch(symbol: SymbolInfo = _state.value.symbol) = viewModelScope.launch {
        app.watchlist.toggle(symbol)
    }

    fun addToWatch(symbol: SymbolInfo) = viewModelScope.launch { app.watchlist.add(symbol) }

    fun removeFromWatch(symbol: SymbolInfo) = viewModelScope.launch { app.watchlist.remove(symbol) }

    // ----------------------------------------------------------------------------------
    // 预测
    // ----------------------------------------------------------------------------------

    fun analyze() {
        if (_state.value.running) return
        viewModelScope.launch {
            runMutex.withLock {
                val s = _state.value
                _state.update {
                    it.copy(running = true, error = null, stage = "准备中…", step = 0, totalSteps = s.predLen)
                }
                runCatching {
                    service.analyze(
                        symbol = s.symbol,
                        timeframe = s.timeframe,
                        requestedLookback = s.lookback,
                        predLen = s.predLen,
                        sampleCount = s.sampleCount,
                        temperature = s.temperature,
                        topP = s.topP,
                        indicatorConfig = s.indicatorConfig,
                        onStage = { text -> _state.update { it.copy(stage = text) } },
                        onStep = { done, total -> _state.update { it.copy(step = done, totalSteps = total) } },
                    )
                }.onSuccess { report ->
                    _state.update {
                        it.copy(running = false, stage = null, report = report, step = it.predLen, error = null)
                    }
                }.onFailure { e ->
                    _state.update {
                        it.copy(
                            running = false,
                            stage = null,
                            error = e.message ?: e::class.java.simpleName,
                        )
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------
    // 回测
    // ----------------------------------------------------------------------------------

    fun runBacktest(params: BacktestParams = BacktestParams(predLen = _state.value.predLen.coerceAtMost(24))) {
        if (_state.value.backtestRunning) return
        viewModelScope.launch {
            runMutex.withLock {
                val s = _state.value
                _state.update {
                    it.copy(backtestRunning = true, backtestError = null, backtest = null, backtestStage = "准备中…")
                }
                runCatching {
                    val engine = app.engine()
                    val zone = app.marketData.timezoneFor(s.symbol)
                    val lookback = engine.manifest.effectiveLookback(params.lookback, params.predLen)
                    val candles = app.marketData.candles(s.symbol, s.timeframe, lookback + 400)
                    backtester.run(
                        candles = candles,
                        symbol = s.symbol,
                        timeframe = s.timeframe,
                        params = params.copy(lookback = lookback),
                        zoneId = zone,
                        market = s.symbol.market,
                        predictor = { history, future ->
                            engine.predict(
                                candles = history,
                                futureTimestamps = future,
                                params = com.kronos.assistant.domain.ForecastParams(
                                    lookback = lookback,
                                    predLen = params.predLen,
                                    sampleCount = 1,
                                    temperature = 1f,
                                    topK = 1,
                                    topP = 1f,
                                    seed = 42L,
                                ),
                                zoneId = zone,
                            )[0]
                        },
                        onProgress = { done, total ->
                            _state.update { it.copy(backtestProgress = done, backtestTotal = total, backtestStage = "回测中…") }
                        },
                    )
                }.onSuccess { result ->
                    _state.update {
                        it.copy(backtestRunning = false, backtest = result, backtestStage = null, backtestError = null)
                    }
                }.onFailure { e ->
                    _state.update {
                        it.copy(
                            backtestRunning = false,
                            backtestStage = null,
                            backtestError = e.message ?: e::class.java.simpleName,
                        )
                    }
                }
            }
        }
    }

    fun clearError() = _state.update { it.copy(error = null) }

    fun clearBacktest() = _state.update { it.copy(backtest = null, backtestError = null) }
}

/** 供 UI 判断某个市场是否走 7x24 交易日历。 */
fun Market.isTwentyFourSeven(): Boolean = this == Market.CRYPTO
