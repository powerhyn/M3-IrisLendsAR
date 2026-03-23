# Review Scope

## Target

Vivid Post-Processing 필터 구현 — develop 브랜치 대비 현재 브랜치의 변경 사항 (PR 스타일 리뷰)

화면 전체에 화사한 느낌을 주는 GPU 전용 포스트프로세싱 필터. Vibrance + 밝기 리프트 + 웜톤 시프트를 단일 패스 셰이더로 구현. 기존 beauty 필터 파이프라인과 독립적으로 동작하며, enabled=false 상태에서도 vivid만 활성화 가능.

## Files (10 files, +307/-12 lines)

### C++ Core
- `cpp/include/iris_sdk/beauty_filter.h` — BeautyFilterConfigV2 구조체 + Helper 확장
- `cpp/include/iris_sdk/sdk_api.h` — IrisBeautyConfigV2 C API 구조체 확장
- `cpp/src/sdk_api_v2.cpp` — C API 변환 함수 + enabled 가드 + 기본값
- `cpp/src/gpu/shader_sources.cpp` — VIVID_POSTPROCESS_FRAGMENT 셰이더
- `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` — vivid 멤버/메서드 선언
- `cpp/src/gpu/gpu_beauty_backend.cpp` — buildEffectiveConfig + vivid 패스 + applyTextureId 수정

### Android JNI Bindings
- `android/iris-sdk/src/main/java/com/irislenssdk/BeautyFilterConfigV2.java` — Java 필드 + Builder
- `android/iris-sdk/src/main/cpp/jni_utils.h` — JniCache vivid field ID 선언
- `android/iris-sdk/src/main/cpp/iris_jni.cpp` — JNI field ID 초기화 + 매핑 함수

### Config
- `.claude/settings.local.json` — (무관)

## Flags

- Security Focus: no
- Performance Critical: no
- Strict Mode: no
- Framework: C++17 / OpenGL ES 3.1 / Android JNI

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
