package com.kurodaki.phonewhisperplus

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.animation.LinearInterpolator
import android.widget.FrameLayout
import kotlin.math.max
import kotlin.math.min

/**
 * Live waveform: a row of rounded bars whose heights react to the current
 * microphone input level in real time, like a voice-message recording UI.
 */
class WaveformView(context: Context) : FrameLayout(context) {

    companion object {
        private const val BAR_COUNT = 5
        private const val MIN_LEVEL = 0.08f
    }

    private val dp = resources.displayMetrics.density
    private val barWidth = 3.5f * dp
    private val barGap = 4f * dp
    private val barMinHeight = 6f * dp
    private val barMaxHeight = 26f * dp

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    // Target levels (0f..1f) set from the audio thread; current levels animate toward them.
    private val targetLevels = FloatArray(BAR_COUNT) { MIN_LEVEL }
    private val currentLevels = FloatArray(BAR_COUNT) { MIN_LEVEL }
    private var animator: ValueAnimator? = null

    init {
        setWillNotDraw(false)
    }

    /** Feed a new normalized input level (0f..1f). Safe to call from any thread. */
    fun pushLevel(level: Float) {
        val clamped = level.coerceIn(0f, 1f)
        post {
            // Shift levels left, add the new one at the end — classic scrolling waveform.
            for (i in 0 until BAR_COUNT - 1) targetLevels[i] = targetLevels[i + 1]
            targetLevels[BAR_COUNT - 1] = max(MIN_LEVEL, clamped)
            ensureAnimating()
        }
    }

    fun reset() {
        for (i in targetLevels.indices) { targetLevels[i] = MIN_LEVEL; currentLevels[i] = MIN_LEVEL }
        animator?.cancel()
        animator = null
        invalidate()
    }

    private fun ensureAnimating() {
        if (animator?.isRunning == true) return
        animator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 90
            interpolator = LinearInterpolator()
            addUpdateListener {
                var changed = false
                for (i in currentLevels.indices) {
                    val diff = targetLevels[i] - currentLevels[i]
                    if (kotlin.math.abs(diff) > 0.01f) {
                        currentLevels[i] += diff * 0.5f
                        changed = true
                    }
                }
                if (changed) invalidate()
            }
            start()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val totalWidth = BAR_COUNT * barWidth + (BAR_COUNT - 1) * barGap
        var x = (width - totalWidth) / 2f
        val centerY = height / 2f

        for (i in 0 until BAR_COUNT) {
            val h = barMinHeight + (barMaxHeight - barMinHeight) * min(1f, currentLevels[i])
            val top = centerY - h / 2f
            val bottom = centerY + h / 2f
            canvas.drawRoundRect(x, top, x + barWidth, bottom, barWidth / 2f, barWidth / 2f, paint)
            x += barWidth + barGap
        }
    }
}
