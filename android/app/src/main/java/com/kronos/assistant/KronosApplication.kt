package com.kronos.assistant

import android.app.Application
import android.util.Log
import com.kronos.assistant.data.MarketDataRepository
import com.kronos.assistant.data.local.WatchlistStore
import com.kronos.assistant.inference.KronosEngine
import com.kronos.assistant.inference.KronosModelStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * 进程级单例容器。
 *
 * 这里**不用 DI 框架**：依赖关系只有 4 个，手写容器更直观，也避免额外的构建复杂度
 * 与运行时开销。
 *
 * [KronosEngine] 的加载代价较高（读取约 120MB 的 ONNX 图），因此：
 * - 首次访问时惰性加载；
 * - 用 [Mutex] 保证并发调用只加载一次；
 * - 加载完成后常驻，避免每次预测都重新解析模型。
 */
class KronosApplication : Application() {

    val marketData: MarketDataRepository by lazy { MarketDataRepository() }
    val watchlist: WatchlistStore by lazy { WatchlistStore(this) }

    /** 应用级协程作用域，用于生命周期与进程一致的初始化任务。 */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val engineMutex = Mutex()

    @Volatile
    private var engineCache: KronosEngine? = null

    @Volatile
    private var loadFailure: Throwable? = null

    /** 模型加载进度，UI 可订阅。 */
    @Volatile
    var loadProgress: String? = null
        private set

    suspend fun engine(): KronosEngine {
        engineCache?.let { return it }
        return engineMutex.withLock {
            engineCache?.let { return@withLock it }
            loadFailure?.let { throw it }
            try {
                loadProgress = "正在释放模型文件…"
                val dir = KronosModelStore.ensureInstalled(this) { name, index ->
                    loadProgress = "正在释放模型文件 ($index/$name)…"
                }
                loadProgress = "正在加载 ONNX Runtime 会话…"
                val t0 = System.currentTimeMillis()
                val engine = KronosEngine.create(dir)
                Log.i(TAG, "Kronos 引擎加载完成，耗时 ${System.currentTimeMillis() - t0}ms")
                engineCache = engine
                loadProgress = null
                engine
            } catch (t: Throwable) {
                loadFailure = t
                loadProgress = null
                throw t
            }
        }
    }

    /** 预热：在首页出现时后台开始加载，避免用户点「开始预测」才等待。 */
    fun warmUp() {
        appScope.launch { runCatching { engine() }.onFailure { Log.w(TAG, "模型预热失败", it) } }
    }

    override fun onTerminate() {
        engineCache?.close()
        engineCache = null
        super.onTerminate()
    }

    private companion object {
        const val TAG = "KronosApplication"
    }
}
