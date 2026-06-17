package com.irislenssdk.tracking

import android.os.Build

/**
 * 에뮬레이터 감지 — MediaPipe GPU delegate가 에뮬레이터에서 EGL 초기화 실패(0x300c 등)를
 * 일으키는 사례가 많아 CPU 강제 폴백에 사용한다 (docs/research/mediapipe-android.md §3).
 */
object EmulatorDetector {
    val isEmulator: Boolean by lazy {
        val fingerprint = Build.FINGERPRINT ?: ""
        fingerprint.startsWith("generic") ||
            fingerprint.startsWith("unknown") ||
            fingerprint.contains("emulator", ignoreCase = true) ||
            fingerprint.contains("sdk_gphone", ignoreCase = true) ||
            fingerprint.contains("generic_x86", ignoreCase = true) ||
            (Build.HARDWARE ?: "").let { it.contains("goldfish") || it.contains("ranchu") } ||
            (Build.PRODUCT ?: "").contains("sdk_gphone", ignoreCase = true)
    }
}
