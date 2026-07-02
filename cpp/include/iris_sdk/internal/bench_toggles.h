/**
 * @file bench_toggles.h
 * @brief internal 벤치 토글 — 공개 sdk_api.h 미노출. JNI 전용 경계. (W4-A §6.4 — 수동 extern 정식화)
 *
 * 헤더 없이 JNI가 수동 extern으로 재선언해 호출하던 internal 벤치 토글 9종을
 * 단일 헤더로 정식화한다. 정의는 sdk_api_v2.cpp의 extern "C" 블록에 있으며,
 * 이 헤더를 정의 측(sdk_api_v2.cpp)과 소비 측(JNI) 양쪽에서 include 하면
 * 컴파일러가 선언↔정의 시그니처 일치를 강제한다(시그니처 드리프트 무검출 구조 해소).
 *
 * 외부 소비자 0 (JNI만 호출). 공개 SDK 패키징 표면이 아니므로 install에서 제외.
 *
 * IRIS_SDK_EXPORT 부착 여부는 sdk_api_v2.cpp 정의부 현 상태를 그대로 미러한다
 * (env·reflection 4종은 부착, 나머지 5종은 미부착). export.h의 빈 매크로지만
 * 정의-선언 일관성을 위해 일치시킨다.
 */

#ifndef IRIS_SDK_INTERNAL_BENCH_TOGGLES_H
#define IRIS_SDK_INTERNAL_BENCH_TOGGLES_H

#include "iris_sdk/sdk_api.h"  // IrisSdkError 가시화

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// [sclera] P6-W5 §5.9: B1/B8 4조합 벤치용 sclera veto 수식 토글.
// mode 0=legacy, 1=color-veto(Codex), 2=luma-only(Gemini).
// ----------------------------------------------------------------------------
void iris_sdk_set_lens_sclera_veto_mode(int mode);

// ----------------------------------------------------------------------------
// [eyelid-mask] EYECLIP A-2: 눈꺼풀 마스크 모드 토글. 0=Y-slab, 1=ellipse, 2=contour.
// 벤치 단계 internal — 정식 활성 시 sdk_api.h 승격.
// ----------------------------------------------------------------------------
void iris_sdk_set_lens_eyelid_mask_mode(int mode);

// ----------------------------------------------------------------------------
// [env·reflection] P6-W4 §5.7/§5.11: 환경 반사 internal C API.
// W5~W6 정식 활성 시점에 sdk_api.h로 승격 검토.
// ----------------------------------------------------------------------------
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_env_map(const uint8_t* data, int width, int height);
IRIS_SDK_EXPORT void iris_sdk_unload_env_map(void);
IRIS_SDK_EXPORT void iris_sdk_set_reflection_mode(int mode);
IRIS_SDK_EXPORT void iris_sdk_set_reflection_intensity(float intensity);

// ----------------------------------------------------------------------------
// [blink·gate·detail·luma] P6-W6 / P7-W2: 블링크 ramp(B5) / 저조도 gate(B9) /
// 디테일 재주입(C10) / avg_iris_luma 실측↔fallback A/B 벤치 토글.
// ----------------------------------------------------------------------------
void iris_sdk_set_lens_blink_up_ms(float ms);
void iris_sdk_set_lens_gate_threshold(float threshold);
void iris_sdk_set_lens_detail_reinject(int enabled);
void iris_sdk_set_use_measured_luma(int enabled);
// P7-W4 §5.8: TintLinearV2 흰자 빛남 cap 벤치 토글 (유효 틴트 배율 상한).
void iris_sdk_set_lens_sclera_tint_max(float cap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IRIS_SDK_INTERNAL_BENCH_TOGGLES_H
