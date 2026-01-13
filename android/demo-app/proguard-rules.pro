# IrisLensSDK Android - Demo App ProGuard Rules
#
# 데모 앱 ProGuard 규칙
#
# @version 1.0.0

# =============================================================================
# Demo App 규칙
# =============================================================================

# 액티비티 보존
-keep public class com.irislenssdk.demo.MainActivity {
    public *;
}

# =============================================================================
# CameraX
# =============================================================================

-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# =============================================================================
# Material Components
# =============================================================================

-keep class com.google.android.material.** { *; }
-dontwarn com.google.android.material.**
