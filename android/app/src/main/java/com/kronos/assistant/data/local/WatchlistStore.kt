package com.kronos.assistant.data.local

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.kronos.assistant.domain.Market
import com.kronos.assistant.domain.SymbolInfo
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

private val Context.dataStore by preferencesDataStore(name = "kronos_prefs")

/** 自选列表的持久化（DataStore + JSON，条目数量很小，无需数据库）。 */
class WatchlistStore(private val context: Context) {

    private val json = Json { ignoreUnknownKeys = true }

    @Serializable
    private data class Entry(val code: String, val name: String, val market: String)

    val entries: Flow<List<SymbolInfo>> = context.dataStore.data.map { prefs ->
        val raw = prefs[WATCHLIST] ?: return@map defaultWatchlist()
        runCatching {
            json.decodeFromString<List<Entry>>(raw).mapNotNull { it.toSymbol() }
        }.getOrElse { defaultWatchlist() }
    }

    suspend fun update(list: List<SymbolInfo>) {
        val encoded = json.encodeToString(
            list.map { Entry(it.code, it.name, it.market.name) },
        )
        context.dataStore.edit { it[WATCHLIST] = encoded }
    }

    suspend fun toggle(symbol: SymbolInfo) {
        val current = currentOnce().toMutableList()
        val existing = current.indexOfFirst { it.id == symbol.id }
        if (existing >= 0) current.removeAt(existing) else current.add(0, symbol)
        update(current)
    }

    suspend fun add(symbol: SymbolInfo) {
        val latest = currentOnce()
        if (latest.none { it.id == symbol.id }) update(listOf(symbol) + latest)
    }

    suspend fun remove(symbol: SymbolInfo) {
        update(currentOnce().filterNot { it.id == symbol.id })
    }

    /** 取当前值的快照。DataStore 的 `data` 是冷流，用 `first()` 保证只读一次并立即返回。 */
    private suspend fun currentOnce(): List<SymbolInfo> = entries.first()

    private fun Entry.toSymbol(): SymbolInfo? {
        val m = Market.entries.firstOrNull { it.name == market } ?: return null
        return SymbolInfo(code, name.ifBlank { code }, m)
    }

    private fun defaultWatchlist(): List<SymbolInfo> = listOf(
        SymbolInfo.crypto("BTCUSDT"),
        SymbolInfo.crypto("ETHUSDT"),
        SymbolInfo.cn("600519", "贵州茅台"),
        SymbolInfo.us("NVDA", "英伟达"),
    )

    private companion object {
        val WATCHLIST = stringPreferencesKey("watchlist")
    }
}
