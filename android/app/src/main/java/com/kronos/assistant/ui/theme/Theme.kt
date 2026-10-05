package com.kronos.assistant.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/** 涨跌色（A 股习惯：红涨绿跌，与国际惯例相反，这里作为可配置项暴露）。 */
object ChartColors {
    val Bull = Color(0xFFEF5350)
    val Bear = Color(0xFF26A69A)
    val BullSoft = Color(0x33EF5350)
    val BearSoft = Color(0x3326A69A)

    val Forecast = Color(0xFF42A5F5)
    val ForecastSoft = Color(0x3342A5F5)
    val Band = Color(0x2442A5F5)
    val Grid = Color(0x1AFFFFFF)
    val Axis = Color(0x99FFFFFF)
    val Crosshair = Color(0x66FFFFFF)
    val Benchmark = Color(0xFF9E9E9E)
}

private val DarkScheme = darkColorScheme(
    primary = Color(0xFF64B5F6),
    onPrimary = Color(0xFF06202E),
    primaryContainer = Color(0xFF123A52),
    onPrimaryContainer = Color(0xFFCDE7FF),
    secondary = Color(0xFF80CBC4),
    background = Color(0xFF0B1220),
    onBackground = Color(0xFFE4EAF2),
    surface = Color(0xFF121B2B),
    onSurface = Color(0xFFE4EAF2),
    surfaceVariant = Color(0xFF1B2637),
    onSurfaceVariant = Color(0xFFB4C0D0),
    error = Color(0xFFEF5350),
    outline = Color(0xFF2A3648),
)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF1565C0),
    secondary = Color(0xFF00796B),
    background = Color(0xFFF7F9FC),
    surface = Color(0xFFFFFFFF),
    surfaceVariant = Color(0xFFE8EDF4),
)

private val AppTypography = Typography(
    titleLarge = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 20.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelSmall = TextStyle(fontSize = 11.sp),
    // 数字用等宽字体，避免跳动
    bodyLarge = TextStyle(fontSize = 16.sp, fontFamily = FontFamily.Monospace),
)

@Composable
fun KronosTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkScheme else LightScheme,
        typography = AppTypography,
        content = content,
    )
}
