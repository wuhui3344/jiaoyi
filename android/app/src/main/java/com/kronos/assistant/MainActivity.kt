package com.kronos.assistant

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.ShowChart
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.TrendingUp
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.kronos.assistant.ui.AppViewModel
import com.kronos.assistant.ui.screens.AnalysisScreen
import com.kronos.assistant.ui.screens.BacktestScreen
import com.kronos.assistant.ui.screens.SettingsScreen
import com.kronos.assistant.ui.screens.WatchlistScreen
import com.kronos.assistant.ui.theme.KronosTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            KronosTheme {
                KronosApp()
            }
        }
    }
}

private enum class Destination(val route: String, val label: String) {
    Analysis("analysis", "分析"),
    Watchlist("watchlist", "自选"),
    Backtest("backtest", "回测"),
    Settings("settings", "设置"),
}

@Composable
private fun KronosApp(viewModel: AppViewModel = viewModel()) {
    val navController = rememberNavController()
    val state by viewModel.state.collectAsState()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = backStackEntry?.destination

    Scaffold(
        bottomBar = {
            NavigationBar {
                Destination.entries.forEach { destination ->
                    val selected = currentDestination?.hierarchy?.any { it.route == destination.route } == true
                    NavigationBarItem(
                        selected = selected,
                        onClick = {
                            navController.navigate(destination.route) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {
                            Icon(
                                imageVector = when (destination) {
                                    Destination.Analysis -> Icons.Filled.ShowChart
                                    Destination.Watchlist -> Icons.Filled.Star
                                    Destination.Backtest -> Icons.Filled.TrendingUp
                                    Destination.Settings -> Icons.Filled.Settings
                                },
                                contentDescription = destination.label,
                            )
                        },
                        label = { Text(destination.label) },
                    )
                }
            }
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            NavHost(
                navController = navController,
                startDestination = Destination.Analysis.route,
                modifier = Modifier.fillMaxSize(),
            ) {
                composable(Destination.Analysis.route) {
                    AnalysisScreen(state, viewModel)
                }
                composable(Destination.Watchlist.route) {
                    WatchlistScreen(state, viewModel)
                }
                composable(Destination.Backtest.route) {
                    BacktestScreen(state, viewModel)
                }
                composable(Destination.Settings.route) {
                    SettingsScreen(state, viewModel)
                }
            }
        }
    }

    // 首次进入时后台预热模型，减少用户等待
    androidx.compose.runtime.LaunchedEffect(Unit) {
        (viewModel.getApplication<android.app.Application>() as? KronosApplication)?.warmUp()
    }
}

/** 预览用：确保主题在 IDE 预览里也能渲染。 */
@Composable
internal fun PreviewBackground(content: @Composable () -> Unit) {
    Column(Modifier.fillMaxSize().padding(0.dp)) {
        Text("", style = MaterialTheme.typography.bodyMedium)
        content()
    }
}
