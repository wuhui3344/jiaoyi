package com.kronos.assistant.inference

import java.util.Arrays
import java.util.Random

/**
 * 采样器，复刻官方 `kronos.py` 的 `top_k_top_p_filtering` + `sample_from_logits`。
 *
 * 与官方保持一致的几个细节：
 * 1. 先除以温度，再做过滤。
 * 2. **`top_k > 0` 时官方会提前返回，不再应用 `top_p`**；因此这两个参数是
 *    "二选一"而不是叠加。
 * 3. `top_k` 的过滤条件是 `logits < 第 k 大的值`，也就是说出现并列时会保留**多于 k 个**
 *    候选，而不是严格截断到 k。
 * 4. `top_p` 保留的是累积概率刚好超过阈值的那个最小集合（外加第一个 token）。
 *
 * 贪心（`topK = 1`）路径与官方逐位一致；开启随机采样时使用本类的 `Random`，
 * 因此**不保证**与 PyTorch 的 `multinomial` 产生相同序列（统计等价即可）。
 */
class Sampler(seed: Long) {

    private val random = Random(seed)

    /** 复用的临时缓冲区，避免每个 token 都分配。 */
    private var sortedProbs = FloatArray(0)
    private var probs = FloatArray(0)

    /**
     * 从 logits 中采样一个 token。
     *
     * @param logits 长度 = 词表大小 (1024)
     * @return 采样出的 token id
     */
    fun sample(logits: FloatArray, temperature: Float, topK: Int, topP: Float): Int {
        val n = logits.size
        if (probs.size != n) {
            probs = FloatArray(n)
            sortedProbs = FloatArray(n)
        }

        val temp = if (temperature <= 0f) 1f else temperature
        var maxLogit = Float.NEGATIVE_INFINITY
        for (i in 0 until n) {
            val v = logits[i] / temp
            probs[i] = v
            if (v > maxLogit) maxLogit = v
        }

        var threshold = Float.NEGATIVE_INFINITY
        if (topK > 0) {
            // 第 k 大的值作为阈值
            System.arraycopy(probs, 0, sortedProbs, 0, n)
            Arrays.sort(sortedProbs)
            val k = topK.coerceAtMost(n)
            threshold = sortedProbs[n - k]
        } else if (topP < 1f) {
            // 先把概率算出来，排序后找累积概率刚超过 topP 的那个概率值
            var sum = 0.0
            for (i in 0 until n) {
                probs[i] = kotlin.math.exp((probs[i] - maxLogit).toDouble()).toFloat()
                sum += probs[i]
            }
            val inv = (1.0 / sum).toFloat()
            System.arraycopy(probs, 0, sortedProbs, 0, n)
            for (i in 0 until n) sortedProbs[i] *= inv
            Arrays.sort(sortedProbs)
            var cumulative = 0f
            threshold = sortedProbs[n - 1]
            for (i in n - 1 downTo 0) {
                cumulative += sortedProbs[i]
                threshold = sortedProbs[i]
                if (cumulative > topP) break
            }
            // probs 里现在存的是归一化后的概率，下面直接用
            return sampleFromProbs(probs, threshold)
        }

        // 用阈值过滤后重新做 softmax
        var sum = 0.0
        for (i in 0 until n) {
            val v = probs[i]
            if (v < threshold) {
                probs[i] = 0f
            } else {
                val e = kotlin.math.exp((v - maxLogit).toDouble()).toFloat()
                probs[i] = e
                sum += e
            }
        }
        val inv = (1.0 / sum).toFloat()
        for (i in 0 until n) probs[i] *= inv
        return sampleFromProbs(probs, 0f)
    }

    /** 按概率做轮盘赌；`threshold` 用于剔除低于该概率的候选（保留并列项）。 */
    private fun sampleFromProbs(probs: FloatArray, threshold: Float): Int {
        var sum = 0.0
        for (p in probs) if (p >= threshold) sum += p
        if (sum <= 0.0) {
            // 数值退化时回退到 argmax，保证总能给出结果
            var best = 0
            var bestProb = Float.NEGATIVE_INFINITY
            for (i in probs.indices) if (probs[i] > bestProb) { bestProb = probs[i]; best = i }
            return best
        }
        val target = random.nextDouble() * sum
        var acc = 0.0
        for (i in probs.indices) {
            if (probs[i] < threshold) continue
            acc += probs[i]
            if (acc >= target) return i
        }
        for (i in probs.indices.reversed()) if (probs[i] >= threshold) return i
        return probs.indices.maxByOrNull { probs[it] } ?: 0
    }

    fun reseed(seed: Long) = random.setSeed(seed)
}
