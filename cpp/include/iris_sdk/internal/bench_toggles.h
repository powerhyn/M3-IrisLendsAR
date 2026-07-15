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
// NLR 클리핑: tuck 리매핑 강도 [0,1] 벤치 토글 (LensSim §3 이식, 0=항등).
void iris_sdk_set_lens_clip_tuck(float t);
// NLR-W2 벤치: 고정 K↔적응 증폭 A/B [0,1] 토글 (0=고정 기본, 1=구 적응식).
void iris_sdk_set_lens_adapt_k(float v);
// NLR-W2 R5: 흰자 페이드 시작점(홍채 반경 단위) 라이브 튜닝 토글 — 검출 반경 오차 보정용. 창 폭 +0.20 고정.
void iris_sdk_set_lens_fade_start(float v);

// ----------------------------------------------------------------------------
// [interior-taper] P8-W4B 벤치: 얼굴 내부 축소(thinChin) taper 비율 프리셋 선택 토글.
// preset 0=기본, 1=볼강조, 2=입코강조, 3=약하게 (jaw_warp_geometry.h kInteriorTaperPresets).
// 범위 밖 인덱스는 소비 시점(computeJawWarp)에서 프리셋 0으로 폴백.
// 상태는 backend 멤버가 아니라 sdk 레벨 std::atomic<int>(sdk_api_v2.cpp)에 보관 →
// GL 컨텍스트 재생성에도 유지. gpu_beauty_backend 가 매 프레임 getter 로 읽어 전달.
// UI 스레드 store ↔ GL 스레드 load 무락 교환(atomic). 다수 의견 수집 후 승자 고정·제거 예정.
// ----------------------------------------------------------------------------
void iris_sdk_set_interior_taper_preset(int preset);
int  iris_sdk_get_interior_taper_preset(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IRIS_SDK_INTERNAL_BENCH_TOGGLES_H
