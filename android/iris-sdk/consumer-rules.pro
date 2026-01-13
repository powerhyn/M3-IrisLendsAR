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
