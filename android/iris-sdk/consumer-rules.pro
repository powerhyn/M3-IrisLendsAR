# IrisLensSDK Android - Consumer ProGuard Rules
#
# SDK를 사용하는 앱에 자동 적용되는 ProGuard 규칙
# - SDK 공개 API 보존
# - JNI 메서드 보존
#
# @version 1.0.0

# =============================================================================
# JNI 네이티브 메서드 (필수)
# =============================================================================

# IrisLensSDK JNI 메서드 보존
-keep class com.irislenssdk.IrisLensSDK {
    private static native *;
}

# =============================================================================
# 공개 API 클래스 (필수)
# =============================================================================

# 메인 SDK 클래스
-keep public class com.irislenssdk.IrisLensSDK {
    public *;
}

# 결과 클래스 (JNI에서 생성)
-keep public class com.irislenssdk.IrisResult {
    public *;
}
-keep public class com.irislenssdk.IrisResult$Iris {
    public *;
}

# 설정 클래스
-keep public class com.irislenssdk.LensConfig {
    public *;
}
-keep public class com.irislenssdk.LensConfig$Builder {
    public *;
}

# =============================================================================
# Kotlin 확장 (선택적 - Kotlin 사용자용)
# =============================================================================

-keep public class com.irislenssdk.kotlin.** {
    public *;
}

# =============================================================================
# 추적 글루 (W4-C: tracking/ → SDK AAR 승격)
# =============================================================================

# 공개 추적 API (FaceTracker / TasksToIrisResult / TrackingSnapshot / LandmarkIndices /
# EmulatorDetector / math). 콜백 람다·생성자가 리플렉션/난독화에 깨지지 않게 보존.
-keep public class com.irislenssdk.tracking.** {
    public *;
}

# MediaPipe Tasks (tasks-vision) — iris-sdk api() 전이 의존. 네이티브 JNI·리플렉션
# 로딩 클래스가 소비자 R8에서 제거되지 않게 보존(공식 권장 keep).
-keep class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# 열거형 보존
-keepclassmembers enum com.irislenssdk.kotlin.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# =============================================================================
# 속성 보존
# =============================================================================

# 애노테이션 보존 (리플렉션용)
-keepattributes *Annotation*

# 예외 정보 보존 (디버깅용)
-keepattributes Exceptions
