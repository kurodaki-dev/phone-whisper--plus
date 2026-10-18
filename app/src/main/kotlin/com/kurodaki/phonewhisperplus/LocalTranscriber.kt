package com.kurodaki.phonewhisperplus

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.*
import java.io.File

/**
 * Local on-device transcription via sherpa-onnx.
 * Models are loaded from the app's external files dir.
 */
class LocalTranscriber private constructor(private val recognizer: OfflineRecognizer) {

    /** Transcribe raw PCM float samples. Blocking — call from background thread. */
    fun transcribe(samples: FloatArray, sampleRate: Int = 16000): String {
        val stream = recognizer.createStream()
        stream.acceptWaveform(samples, sampleRate)
        recognizer.decode(stream)
        val result = recognizer.getResult(stream)
        stream.release()
        return result.text.trim()
    }

    fun release() = recognizer.release()

    companion object {
        private const val TAG = "LocalTranscriber"

        /** Find available model dirs under the app's files/models/ dir */
        fun availableModels(ctx: Context): List<String> {
            val modelsDir = File(ctx.filesDir, "models")
            if (!modelsDir.exists()) return emptyList()
            return modelsDir.listFiles()?.filter { it.isDirectory }?.map { it.name } ?: emptyList()
        }

        /** Create a LocalTranscriber for the given model directory name. Returns null on failure. */
        fun create(ctx: Context, modelName: String): LocalTranscriber? {
            val modelDir = File(ctx.filesDir, "models/$modelName")
            if (!modelDir.exists()) {
                Log.e(TAG, "Model dir not found: $modelDir")
                return null
            }

            val config = detectModelConfig(modelDir) ?: run {
                Log.e(TAG, "Could not detect model type in $modelDir")
                return null
            }

            return try {
                val threads = optimalThreadCount()
                config.modelConfig.numThreads = threads
                val recognizer = OfflineRecognizer(assetManager = null, config = config)
                Log.i(TAG, "Loaded model: $modelName (threads=$threads)")
                LocalTranscriber(recognizer)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: ${e.message}", e)
                null
            }
        }

        /** Use up to 4 threads, scaled to the device's core count, for faster decoding. */
        private fun optimalThreadCount(): Int {
            val cores = Runtime.getRuntime().availableProcessors()
            return cores.coerceIn(1, 4)
        }

        /** Auto-detect model type from files present in the directory. */
        private fun detectModelConfig(dir: File): OfflineRecognizerConfig? {
            val p = dir.absolutePath
            val tokens = findFile(p, "tokens", suffix = ".txt") ?: return null

            // Moonshine v1 (preprocessor + uncached/cached decoder) or v2 (encoder + merged decoder)
            val moonshinePreprocessor = findFile(p, "preprocess")
            if (moonshinePreprocessor != null) {
                val uncached = findFile(p, "uncached_decode")
                val cached = findFile(p, "cached_decode")
                if (uncached != null && cached != null) {
                    return buildConfig(tokens) {
                        moonshine = OfflineMoonshineModelConfig(
                            preprocessor = moonshinePreprocessor,
                            encoder = findFile(p, "encode") ?: return@buildConfig null,
                            uncachedDecoder = uncached,
                            cachedDecoder = cached,
                        )
                    }
                }
            }
            val moonshineMerged = findFile(p, "merged_decode")
            if (moonshineMerged != null) {
                return buildConfig(tokens) {
                    moonshine = OfflineMoonshineModelConfig(
                        encoder = findFile(p, "encode") ?: return@buildConfig null,
                        mergedDecoder = moonshineMerged,
                    )
                }
            }

            // NeMo Canary (encoder + decoder, directory name mentions "canary")
            if (dir.name.contains("canary", ignoreCase = true)) {
                val encoder = findFile(p, "encoder")
                val decoder = findFile(p, "decoder")
                if (encoder != null && decoder != null) {
                    return buildConfig(tokens) {
                        canary = OfflineCanaryModelConfig(encoder = encoder, decoder = decoder)
                    }
                }
            }

            // Whisper (encoder + decoder, no joiner)
            val whisperEncoder = findFile(p, "encoder")
            val whisperDecoder = findFile(p, "decoder")
            if (whisperEncoder != null && whisperDecoder != null && findFile(p, "joiner") == null) {
                return buildConfig(tokens) {
                    whisper = OfflineWhisperModelConfig(encoder = whisperEncoder, decoder = whisperDecoder)
                    modelType = "whisper"
                }
            }

            // NeMo transducer / Parakeet TDT (encoder + decoder + joiner)
            val encoder = findFile(p, "encoder")
            val decoder = findFile(p, "decoder")
            val joiner = findFile(p, "joiner")
            if (encoder != null && decoder != null && joiner != null) {
                return buildConfig(tokens) {
                    transducer = OfflineTransducerModelConfig(encoder = encoder, decoder = decoder, joiner = joiner)
                    modelType = "nemo_transducer"
                }
            }

            // SenseVoice (single model.onnx, directory name mentions "sense-voice"/"sensevoice")
            val singleModel = findFile(p, "model")
            if (singleModel != null && (dir.name.contains("sense-voice", true) || dir.name.contains("sensevoice", true))) {
                return buildConfig(tokens) {
                    senseVoice = OfflineSenseVoiceModelConfig(model = singleModel, useInverseTextNormalization = true)
                }
            }

            // Paraformer (single model.onnx, directory name mentions "paraformer")
            if (singleModel != null && dir.name.contains("paraformer", true)) {
                return buildConfig(tokens) {
                    paraformer = OfflineParaformerModelConfig(model = singleModel)
                }
            }

            // NeMo CTC / Parakeet CTC (single model.onnx, fallback)
            if (singleModel != null) {
                return buildConfig(tokens) {
                    nemo = OfflineNemoEncDecCtcModelConfig(model = singleModel)
                }
            }

            return null
        }

        private inline fun buildConfig(
            tokens: String,
            block: OfflineModelConfig.() -> OfflineModelConfig?
        ): OfflineRecognizerConfig? {
            val modelConfig = OfflineModelConfig(tokens = tokens).block() ?: return null
            return OfflineRecognizerConfig(modelConfig = modelConfig)
        }

        /**
         * Find a file whose name contains [token] (case-sensitive, matching sherpa-onnx's own
         * naming conventions, e.g. "base.en-encoder.int8.onnx" or "encoder.int8.onnx").
         * Prefers int8-quantized variants for smaller size and faster inference.
         */
        private fun findFile(dir: String, token: String, suffix: String? = null): String? {
            val files = File(dir).listFiles() ?: return null
            val matches = files.filter {
                it.name.contains(token) &&
                    (suffix == null || it.name.endsWith(suffix)) &&
                    (suffix != null || it.name.endsWith(".onnx") || it.name.endsWith(".ort"))
            }
            // Prefer int8 quantized for speed/size, but avoid matching "encoder" when we want "decoder" etc.
            return matches.firstOrNull { it.name.contains("int8") }?.absolutePath
                ?: matches.firstOrNull()?.absolutePath
        }
    }
}
