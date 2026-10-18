package com.kurodaki.phonewhisperplus

import android.content.Context
import android.os.StatFs
import okhttp3.OkHttpClient
import okhttp3.Request
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.*
import java.util.concurrent.TimeUnit

data class Model(
    val name: String,
    val archive: String,
    val sizeMb: Int,
    val quality: String,
    val language: String = "English",
    val recommended: Boolean = false,
)

/**
 * Full catalog of installable sherpa-onnx offline models.
 * All archive names are verified against the live
 * k2-fsa/sherpa-onnx "asr-models" GitHub release.
 */
val MODEL_CATALOG = listOf(
    // --- English, recommended picks ---
    Model("Parakeet 110M", "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8",
        104, "★★★ Best value", "English", recommended = true),
    Model("Moonshine Tiny", "sherpa-onnx-moonshine-tiny-en-int8",
        108, "★★☆ Fastest", "English"),
    Model("Whisper Base (en)", "sherpa-onnx-whisper-base.en",
        209, "★★★ Solid baseline", "English"),
    Model("Moonshine Base", "sherpa-onnx-moonshine-base-en-int8",
        251, "★★★ Fast + accurate", "English"),
    Model("Parakeet 0.6B v2", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        482, "★★★★ Best quality", "English"),
    Model("Parakeet 0.6B v3", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        487, "★★★★ Best quality", "English"),

    // --- Whisper, more English sizes ---
    Model("Whisper Tiny (en)", "sherpa-onnx-whisper-tiny.en",
        118, "★★☆ Smallest Whisper", "English"),
    Model("Whisper Small (en)", "sherpa-onnx-whisper-small.en",
        636, "★★★★ High accuracy", "English"),
    Model("Whisper Distil Small (en)", "sherpa-onnx-whisper-distil-small.en",
        454, "★★★ Distilled, fast", "English"),
    Model("Whisper Distil Medium (en)", "sherpa-onnx-whisper-distil-medium.en",
        1007, "★★★★ Distilled, accurate", "English"),
    Model("Whisper Medium (en)", "sherpa-onnx-whisper-medium.en",
        1906, "★★★★★ Max accuracy", "English"),

    // --- Multilingual ---
    Model("Whisper Base (multi)", "sherpa-onnx-whisper-base",
        208, "★★★ 90+ languages", "Multilingual"),
    Model("Whisper Small (multi)", "sherpa-onnx-whisper-small",
        639, "★★★★ 90+ languages", "Multilingual"),
    Model("Whisper Turbo (multi)", "sherpa-onnx-whisper-turbo",
        564, "★★★★ Fast, 90+ languages", "Multilingual"),
    Model("SenseVoice (zh/en/ja/ko/yue)", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        1048, "★★★★ 5 Asian languages + English", "Multilingual"),
    Model("Canary 180M (en/es/de/fr)", "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
        154, "★★★★ Fast, 4 languages", "Multilingual"),

    // --- Chinese / Cantonese ---
    Model("Paraformer (zh)", "sherpa-onnx-paraformer-zh-2023-09-14",
        234, "★★★★ Mandarin Chinese", "Chinese"),
    Model("Paraformer (zh/yue/en)", "sherpa-onnx-paraformer-trilingual-zh-cantonese-en",
        1059, "★★★★ Mandarin, Cantonese, English", "Chinese"),

    // --- Other ---
    Model("Paraformer (en)", "sherpa-onnx-paraformer-en-2024-03-09",
        1022, "★★★ English, Alibaba FunASR", "English"),
)

/** Distinct languages in catalog order, for grouping the model list in the UI. */
fun modelCatalogLanguages(): List<String> =
    MODEL_CATALOG.map { it.language }.distinct()

sealed class DownloadState {
    data class Downloading(val progress: Float) : DownloadState()
    object Extracting : DownloadState()
    object Done : DownloadState()
    data class Error(val message: String) : DownloadState()
}

object ModelDownloader {
    private const val BASE_URL =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

    // Generous timeouts: some models are 1-2 GB and mobile networks are slow/unstable.
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    fun modelDir(ctx: Context, model: Model) =
        File(ctx.filesDir, "models/${model.archive}")

    fun isInstalled(ctx: Context, model: Model) =
        modelDir(ctx, model).exists()

    /** Free space (in MB) available on the filesystem backing the app's internal storage. */
    fun freeSpaceMb(ctx: Context): Long {
        val stat = StatFs(ctx.filesDir.path)
        return (stat.availableBytes / (1024 * 1024))
    }

    /** Download and extract model. Callbacks fire on background thread. */
    fun download(ctx: Context, model: Model, onState: (DownloadState) -> Unit) {
        // A downloaded archive is roughly the same size unpacked as a rough safety margin;
        // require enough headroom for both the temp archive and its extracted contents.
        val requiredMb = (model.sizeMb * 2.2).toLong()
        if (freeSpaceMb(ctx) < requiredMb) {
            onState(DownloadState.Error("Not enough storage (~${requiredMb}MB needed)"))
            return
        }

        val url = "$BASE_URL/${model.archive}.tar.bz2"
        // Unique temp filename per model avoids collisions if two downloads are kicked off
        // close together, and keeps a failed download from corrupting a retry.
        val tmpFile = File(ctx.cacheDir, "${model.archive}.${System.currentTimeMillis()}.tmp")
        val outDir = File(ctx.filesDir, "models")
        val finalDir = modelDir(ctx, model)

        Thread {
            try {
                downloadFile(url, tmpFile, onState)
                onState(DownloadState.Extracting)
                // Extract to a staging dir first, then atomically rename into place, so a
                // crash or cancel mid-extraction never leaves a half-installed model that
                // isInstalled() would wrongly report as present.
                val stagingDir = File(outDir, "${model.archive}.partial")
                stagingDir.deleteRecursively()
                extractTarBz2(tmpFile, stagingDir)

                val extractedRoot = stagingDir.listFiles()?.singleOrNull { it.isDirectory }
                    ?: stagingDir
                finalDir.deleteRecursively()
                if (!extractedRoot.renameTo(finalDir)) {
                    extractedRoot.copyRecursively(finalDir, overwrite = true)
                }
                stagingDir.deleteRecursively()

                onState(DownloadState.Done)
            } catch (e: Exception) {
                finalDir.deleteRecursively()
                onState(DownloadState.Error(e.message ?: "Unknown error"))
            } finally {
                tmpFile.delete()
            }
        }.start()
    }

    fun delete(ctx: Context, model: Model) =
        modelDir(ctx, model).deleteRecursively()

    private fun downloadFile(
        url: String, dest: File, onState: (DownloadState) -> Unit
    ) {
        val response = client.newCall(Request.Builder().url(url).build()).execute()
        if (!response.isSuccessful) throw IOException("HTTP ${response.code}")
        val body = response.body ?: throw IOException("Empty response")
        val total = body.contentLength()
        var downloaded = 0L
        var lastReported = 0f

        body.byteStream().use { src ->
            FileOutputStream(dest).use { dst ->
                val buf = ByteArray(65536)
                var n: Int
                while (src.read(buf).also { n = it } != -1) {
                    dst.write(buf, 0, n)
                    downloaded += n
                    if (total > 0) {
                        val progress = downloaded.toFloat() / total
                        // Throttle UI updates to ~1% steps instead of every 64KB chunk.
                        if (progress - lastReported >= 0.01f || progress >= 1f) {
                            lastReported = progress
                            onState(DownloadState.Downloading(progress))
                        }
                    }
                }
            }
        }
    }

    /** Extract tar.bz2 to outDir. Validates paths to prevent traversal. */
    fun extractTarBz2(archive: File, outDir: File) {
        outDir.mkdirs()
        val bzIn = BZip2CompressorInputStream(BufferedInputStream(FileInputStream(archive)))
        TarArchiveInputStream(bzIn).use { tar ->
            generateSequence { tar.nextEntry }.forEach { entry ->
                val dest = File(outDir, entry.name)
                require(dest.canonicalPath.startsWith(outDir.canonicalPath)) {
                    "Path traversal: ${entry.name}"
                }
                if (entry.isDirectory) dest.mkdirs()
                else {
                    dest.parentFile?.mkdirs()
                    FileOutputStream(dest).use { tar.copyTo(it) }
                }
            }
        }
    }
}
