# IrisLensSDK Android - ProGuard Rules
#
# SDK 라이브러리 ProGuard/R8 난독화 규칙
# - JNI 메서드 보존
# - 공개 API 보존
#
# @version 1.0.0

# =============================================================================
# 일반 Android 설정
# =============================================================================

# 소스 파일 및 라인 번호 유지 (스택 트레이스용)
-keepattributes SourceFile,LineNumberTable

# 제네릭 타입 정보 유지 (리플렉션용)
-keepattributes Signature

# 애노테이션 유지
-keepattributes *Annotation*

# 예외 정보 유지
-keepattributes Exceptions

# =============================================================================
# JNI 네이티브 메서드 보존
# =============================================================================

# JNI에서 호출하는 모든 네이티브 메서드 보존
-keepclasseswithmembernames class * {
    native <methods>;
}

# =============================================================================
# IrisLensSDK 공개 API 보존
# =============================================================================

# 메인 SDK 클래스 및 공개 메서드 보존
-keep public class com.irislenssdk.IrisLensSDK {
    public *;
    private static native *;
}

# 결과 클래스 보존 (JNI에서 생성)
-keep public class com.irislenssdk.IrisResult {
    public *;
}
-keep public class com.irislenssdk.IrisResult$Iris {
    public *;
}

# 설정 클래스 보존
-keep public class com.irislenssdk.LensConfig {
    public *;
}
-keep public class com.irislenssdk.LensConfig$Builder {
    public *;
}

# =============================================================================
# Kotlin 확장 보존
# =============================================================================

# Kotlin 확장 클래스 보존
-keep public class com.irislenssdk.kotlin.** {
    public *;
}

# Kotlin 인라인 클래스 보존
-keepclassmembers class com.irislenssdk.kotlin.** {
    public synthetic *** box-impl(...);
    public static *** unbox-impl(***);
}

# Kotlin 코루틴 지원
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# =============================================================================
# 열거형(Enum) 보존
# =============================================================================

# 모든 열거형 값 보존 (JNI에서 사용 가능)
-keepclassmembers enum com.irislenssdk.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

-keepclassmembers enum com.irislenssdk.kotlin.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# =============================================================================
# 콜백 인터페이스 보존
# =============================================================================

# 리스너 및 콜백 인터페이스 보존
-keep public interface com.irislenssdk.IrisLensSDK$OnDetectionListener {
    public *;
}

# =============================================================================
# 직렬화 지원
# =============================================================================

# Serializable 클래스 필드 보존
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# Parcelable 클래스 보존
-keep class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator *;
}

# =============================================================================
# 성능 최적화
# =============================================================================

# 인라인 최적화 허용
-allowaccessmodification

# 클래스 최적화
-optimizationpasses 5

# =============================================================================
# 경고 억제
# =============================================================================

# 알려진 안전한 경고 억제
-dontwarn javax.annotation.**
-dontwarn org.jetbrains.annotations.**
-dontwarn kotlin.reflect.jvm.internal.**
-dontwarn java.lang.invoke.StringConcatFactory
