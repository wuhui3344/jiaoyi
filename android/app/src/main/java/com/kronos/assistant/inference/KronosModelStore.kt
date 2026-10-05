package com.kronos.assistant.inference

import android.content.Context
import android.util.Log
import java.io.File

/**
 * 负责把打包在 `assets/kronos/` 下的 ONNX 模型释放到内部存储。
 *
 * 为什么不直接读 assets？
 * ONNX Runtime 的 `createSession` 需要一个真实文件路径（要 mmap 权重），
 * 而 assets 在 APK 里是压缩包内的条目，无法直接给出路径。
 *
 * 释放策略：
 * - 用**模型文件名 + 大小**生成指纹写入 `model.version`，指纹不变则跳过拷贝；
 * - 拷贝先写 `*.tmp` 再 rename，避免中途被杀进程留下半个文件。
 */
object KronosModelStore {

    private const val TAG = "KronosModelStore"
    const val ASSET_DIR = "kronos"
    const val MANIFEST_NAME = "kronos_manifest.json"
    private const val VERSION_MARKER = "model.version"

    /** 模型文件较大（约 120MB），释放时上报进度，便于 UI 显示。 */
    fun ensureInstalled(context: Context, onProgress: ((String, Int) -> Unit)? = null): File {
        val target = File(context.filesDir, ASSET_DIR)
        if (!target.exists() && !target.mkdirs()) {
            error("无法创建模型目录: ${target.absolutePath}")
        }

        val assetFiles = context.assets.list(ASSET_DIR)?.toList().orEmpty()
        if (assetFiles.isEmpty()) {
            error("APK 中未找到 $ASSET_DIR/ 资源，请先执行 tools/onnx_export/export_onnx.py")
        }

        val fingerprint = assetFiles.sorted().joinToString("|") { name ->
            "$name:${context.assets.open("$ASSET_DIR/$name").use { it.available() }}"
        }
        val marker = File(target, VERSION_MARKER)

        if (marker.takeIf { it.isFile }?.readText() == fingerprint &&
            assetFiles.all { File(target, it).isFile }
        ) {
            Log.i(TAG, "模型已就绪，跳过释放")
            return target
        }

        Log.i(TAG, "开始释放模型 (${assetFiles.size} 个文件) 到 ${target.absolutePath}")
        assetFiles.forEachIndexed { index, name ->
            onProgress?.invoke(name, index)
            val out = File(target, name)
            val tmp = File(target, "$name.tmp")
            context.assets.open("$ASSET_DIR/$name").use { input ->
                tmp.outputStream().use { output -> input.copyTo(output, DEFAULT_BUFFER_SIZE * 8) }
            }
            if (out.exists()) out.delete()
            if (!tmp.renameTo(out)) {
                tmp.copyTo(out, overwrite = true)
                tmp.delete()
            }
        }
        marker.writeText(fingerprint)
        Log.i(TAG, "模型释放完成")
        return target
    }

    fun loadManifest(modelDir: File): KronosManifest {
        val file = File(modelDir, MANIFEST_NAME)
        if (!file.isFile) {
            error("缺少 $MANIFEST_NAME，模型未正确导出")
        }
        return KronosManifest.parse(file.readText())
    }
}
