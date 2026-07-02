/**
 * @file gpu_lens_renderer.h
 * @brief GPU 기반 렌즈 렌더러 (OpenGL ES 3.1)
 *
 * Android 데모 CameraGLRenderer.kt의 검증된 GPU 렌즈 파이프라인을
 * SDK C++ 코어로 포팅. 전 플랫폼에서 동일한 렌즈 품질 보장.
 *
 * 기능:
 * - 비대칭 타원 마스크 (fitEyeEllipse)
 * - 눈꺼풀 클리핑 (Y-slab + 타원 듀얼 모드)
 * - 8종 블렌드 모드 (GPU 셰이더)
 * - Sclera Protection (기하학적 + 색상 기반)
 * - Contact Shadow (눈꺼풀 아래 그림자)
 * - OneEuroFilter 스무딩 (타원 + 눈꺼풀 파라미터)
 * - 눈꺼풀 홀드 프레임 (5프레임 캐시)
 */

#ifndef IRIS_SDK_GPU_LENS_RENDERER_H
#define IRIS_SDK_GPU_LENS_RENDERER_H

#include "iris_sdk/gpu/eye_render_packet.h"
#include "iris_sdk/gpu/shader_manager.h"
#include "iris_sdk/gpu/texture_pool.h"
#include "iris_sdk/lens_sku_metadata.h"
#include "iris_sdk/one_euro_filter.h"
#include "iris_sdk/types.h"

#include <memory>
#include <mutex>
#include <array>
#include <chrono>
#include <string>

namespace iris_sdk {

class IRenderContext;

#if IRIS_SDK_GPU_AVAILABLE
class GLESRenderContext;
#endif

/**
 * @brief GPU 기반 렌즈 렌더러 (OpenGL ES 3.1)
 *
 * Android 데모 CameraGLRenderer.kt의 검증된 GPU 렌즈 파이프라인을
 * SDK C++ 코어로 포팅. 전 플랫폼에서 동일한 렌즈 품질 보장.
 *
 * 기능:
 * - 비대칭 타원 마스크 (fitEyeEllipse)
 * - 눈꺼풀 클리핑 (Y-slab + 타원 듀얼 모드)
 * - 8종 블렌드 모드 (GPU 셰이더)
 * - Sclera Protection (기하학적 + 색상 기반)
 * - Contact Shadow (눈꺼풀 아래 그림자)
 * - OneEuroFilter 스무딩 (타원 + 눈꺼풀 파라미터)
 * - 눈꺼풀 홀드 프레임 (5프레임 캐시)
 */
class GPULensRenderer {
public:
    GPULensRenderer();
    ~GPULensRenderer();

    GPULensRenderer(const GPULensRenderer&) = delete;
    GPULensRenderer& operator=(const GPULensRenderer&) = delete;
    GPULensRenderer(GPULensRenderer&&) = delete;
    GPULensRenderer& operator=(GPULensRenderer&&) = delete;

    // ========================================
    // 라이프사이클
    // ========================================

    bool initialize(IRenderContext* render_context);
    void release();
    bool isInitialized() const;

    // ========================================
    // 텍스처 관리
    // ========================================

    /// 렌즈 텍스처 업로드 (RGBA 데이터)
    /// P6-W7: sku_id는 SKU 레지스트리 조회용으로 보존된 인프라(현재 렌더링 분기는 없음 —
    /// 림발은 에셋이 책임). 기존 3-인자 호출부 호환 위해 기본값.
    bool loadLensTexture(const uint8_t* data, int width, int height,
                         const std::string& sku_id = "");

    /// 렌즈 텍스처 해제
    void unloadLensTexture();

    /// 텍스처 로드 여부
    bool hasLensTexture() const;

    // ========================================
    // 환경 반사 (P6-W4 §5.7/§5.11)
    // ========================================

    /// P6-W4 §5.7: env_map 텍스처 로드 (RGB 8bit, demo assets `env/` 경로).
    /// W3 §5.13 합의 — SDK 내장 보류 (W8~W9). mipmap 자동 생성.
    bool loadEnvMap(const uint8_t* data, int width, int height);

    /// P6-W4: env_map 텍스처 해제.
    void unloadEnvMap();

    /// P6-W4 §5.11: 반사 소스 런타임 토글 (방식 A uniform 스위치).
    /// mode: 0=OFF, 1=EnvMap, 2=Periphery. W4 B2 벤치 24클립에서 토글.
    void setReflectionMode(int mode);

    /// P6-W4 §5.7 hand-off: 강도 튜닝 (W3 §5.7 기본 0.3 — R1 미토론 항목).
    /// intensity는 [0.0, 1.0]로 clamp.
    void setReflectionIntensity(float intensity);

    // ========================================
    // 렌더링
    // ========================================

    /**
     * @brief 텍스처 ID 기반 렌즈 렌더링 (Zero-Copy)
     *
     * @param input_texture 입력 카메라 프레임 텍스처 ID
     * @param output_texture 출력 텍스처 ID (생성됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param iris_result 홍채 검출 결과 (face_mesh 포함)
     * @param config 렌즈 설정
     * @return 에러 코드
     */
    ErrorCode renderToTexture(
        uint32_t input_texture,
        uint32_t* output_texture,
        int width, int height,
        const IrisResult& iris_result,
        const LensConfig& config);

    // ========================================
    // 설정
    // ========================================

    void setScleraProtectEnabled(bool enabled);
    /// P6-W5 §5.9: B1/B8 4조합 벤치용 sclera veto 수식 토글.
    /// mode: 0=legacy, 1=color-veto(Codex), 2=luma-only(Gemini). 범위 외 값은 0으로 clamp.
    void setScleraVetoMode(int mode);
    void setContactShadowEnabled(bool enabled);
    void setContactShadowIntensity(float intensity);
    void setEllipseMaskEnabled(bool enabled);
    /// EYECLIP A-2: 눈꺼풀 마스크 모드 (0=Y-slab, 1=ellipse, 2=contour). 범위 외 clamp[0,2].
    void setEyelidMaskMode(int mode);

    // ========================================
    // P6-W6: 블링크 ramp(B5) / 저조도 디테일 gate(B9) / C10 디테일 재주입 토글
    // ========================================
    /// P6-W6 §1.3 B5: 블링크 up ramp 95% 도달 시간(ms). 토글 60/80/120, 기본 80. clamp[30,200].
    void setBlinkUpMs(float ms);
    /// P6-W6 §5.7 B9: 저조도 디테일 gate 임계값(linear avg luma). 토글 0.10/0.15/0.25, 기본 0.15. clamp[0,1].
    void setGateThreshold(float t);
    /// P6-W6 §5.2 C10: 홍채 inner 디테일 재주입 on/off (기본 on).
    void setDetailReinject(bool enabled);
    /// P7-W2 §5.6: avg_iris_luma 실측↔fallback A/B 토글 (기본 false=fallback, 안전 롤백).
    ///   false면 updateAvgIrisLuma가 항상 fallback chain → 0.1225. true면 packet 실측 사용.
    void setUseMeasuredLuma(bool enabled);

    // ========================================
    // P6-W7: 림발 자동감지 fallback + SKU 메타데이터
    // ========================================
    /// 외부(바인딩/데모)가 파싱된 SKU 레지스트리를 주입. 외부 소유, null 허용.
    /// 림발은 에셋이 책임지므로 현재 SKU별 동작은 없음(인프라 보존, 향후 W5 prefers_crl 등 활용).
    void setSkuRegistry(const LensSkuRegistry* registry);

    /**
     * @deprecated P5-W3-05 S1에서 고정 조명 하이라이트 폐기. C5 환경 반사 계층이 대체.
     * 호환성 위해 선언은 유지하지만 no-op. 호출부는 제거 권장.
     */
    [[deprecated("P5-W3-05 S1: 고정 조명 하이라이트 폐기. C5 환경 반사 계층이 대체.")]]
    void setHighlightEnabled(bool enabled);

    // ========================================
    // 타원 피팅 (공개 유틸리티)
    // ========================================

    /**
     * @brief Face Mesh 랜드마크에서 비대칭 타원 피팅
     *
     * @param face_mesh 478점 Face Mesh (IrisLandmark[478])
     * @param is_left_eye true=왼쪽 눈
     * @param out_cx, out_cy 타원 중심 (정규화 좌표)
     * @param out_rx_inner 내안각 반경
     * @param out_rx_outer 외안각 반경
     * @param out_ry Y 반경
     * @param out_rotation 회전 각도 (라디안)
     */
    static void fitEyeEllipse(const IrisLandmark* face_mesh, bool is_left_eye,
                               float& out_cx, float& out_cy,
                               float& out_rx_inner, float& out_rx_outer,
                               float& out_ry, float& out_rotation);

    /**
     * @brief 눈꺼풀 Y 좌표 추출 (중앙값)
     */
    static float medianLandmarkY(const IrisLandmark* face_mesh, const int* indices, int count);

private:
    bool initializeShaders();
    void setupFullscreenQuad();
    void renderFullscreenQuad();
    void updateEyelidCache(const IrisResult& iris_result);
    void updateEllipseCache(const IrisResult& iris_result);
    void updateContourCache(const IrisResult& iris_result);  // EYECLIP A-2

    /// W1 §5.3 fallback chain (실측 source 미연결 버전).
    /// 1) packet.avg_iris_luma → 2) hold(직전 유효값) < kAvgLumaMaxHoldFrames → 3) fallback 상수.
    /// 실측은 W6 (비동기 readback + 노출 정규화)에서 packet에 채울 예정.
    /// 반환은 clamp [kAvgLumaClampMin, kAvgLumaClampMax].
    float updateAvgIrisLuma(const gpu::EyeRenderPacket& left,
                            const gpu::EyeRenderPacket& right);

    // 렌더 컨텍스트 (외부 소유)
#if IRIS_SDK_GPU_AVAILABLE
    GLESRenderContext* render_context_ = nullptr;
#else
    void* render_context_ = nullptr;
#endif

    std::unique_ptr<ShaderManager> shader_manager_;
    std::unique_ptr<TexturePool> texture_pool_;

    // 풀스크린 쿼드
    GLuint quad_vao_ = 0;
    GLuint quad_vbo_ = 0;

    // 셰이더 프로그램
    GLuint lens_program_ = 0;

    // 렌즈 텍스처
    GLuint lens_texture_ = 0;
    int lens_texture_width_ = 0;
    int lens_texture_height_ = 0;

    // P6-W4 §5.7/§5.11: 환경 반사 상태 (lens_texture_ 미러).
    GLuint env_map_texture_ = 0;
    int env_map_width_ = 0;
    int env_map_height_ = 0;
    int reflection_mode_ = 0;             // 0=OFF (W3 scaffold no-op 기본 유지)
    float reflection_intensity_ = 0.3f;   // W3 §5.7 기본

    // 설정
    // 기본값은 Kotlin 데모 참조 구현(CameraGLRenderer.kt 558~562)과 일치
    bool sclera_protect_ = true;
    int sclera_veto_mode_ = 0;  // P6-W5: 0=legacy, 1=color-veto(Codex), 2=luma-only(Gemini)
    bool contact_shadow_ = false;
    float shadow_intensity_ = 0.15f;
    int eyelid_mask_mode_ = 0;  // EYECLIP A-2: 0=Y-slab, 1=ellipse, 2=contour (기존 use_ellipse_mask_ 대체)
    // P5-W3-05 S1 D5: highlight_enabled_ 멤버 제거 (uniform/기능 모두 폐기)

    // P6-W7: SKU 레지스트리 (외부 소유, null 허용). 림발 셰이더 기능은 제거됐고
    // 현재 SKU 정보에 따른 렌더링 분기는 없음 — 인프라만 보존(향후 W5 prefers_crl 등 활용).
    const LensSkuRegistry* sku_registry_ = nullptr;

    // ========================================
    // Uniform Location 캐시
    // ========================================
    struct LensUniforms {
        GLint uCameraTexture = -1;
        GLint uLensTexture = -1;

        // 홍채 파라미터
        GLint uLeftIrisCenter = -1;
        GLint uLeftIrisRadius = -1;
        GLint uRightIrisCenter = -1;
        GLint uRightIrisRadius = -1;

        // 렌즈 설정
        GLint uOpacity = -1;
        GLint uLensScale = -1;
        GLint uEdgeFeather = -1;
        GLint uBlendMode = -1;
        GLint uApplyLeft = -1;
        GLint uApplyRight = -1;
        GLint uFrameAspect = -1;

        // 눈꺼풀 클리핑
        GLint uLeftEyeTop = -1;
        GLint uLeftEyeBottom = -1;
        GLint uRightEyeTop = -1;
        GLint uRightEyeBottom = -1;
        GLint uEyelidFeather = -1;

        // 기능 플래그
        GLint uScleraProtect = -1;
        GLint uScleraVetoMode = -1;  // P6-W5 §5.9: B1/B8 4조합 벤치용 veto 수식 토글
        GLint uContactShadow = -1;
        GLint uShadowIntensity = -1;
        GLint uMaxDetail = -1;

        // 비대칭 타원
        GLint uUseEllipseMask = -1;
        GLint uLeftEyeEllipseCenter = -1;
        GLint uLeftEyeEllipseRadii = -1;
        GLint uLeftEyeEllipseRot = -1;
        GLint uRightEyeEllipseCenter = -1;
        GLint uRightEyeEllipseRadii = -1;
        GLint uRightEyeEllipseRot = -1;
        // EYECLIP A-2: contour 마스크 (모두 GLint — valid-count reinterpret_cast 루프 :519 보호)
        GLint uLeftEyeContour = -1;
        GLint uRightEyeContour = -1;
        GLint uLeftContourAABB = -1;
        GLint uRightContourAABB = -1;

        // 기타
        GLint uAvgIrisLum = -1;
        GLint uDetH = -1;
        // P5-W3-05 S1 D5: uHighlightEnabled 멤버 제거

        // P6-W3 §5.6: C5 환경 반사 가산 계층 uniform location (W4 B2 벤치에서 활용).
        GLint uSourceType = -1;
        GLint uReflectionIntensity = -1;
        GLint uEnvMap = -1;

        // P6-W6 §5.2/§5.7: C10 디테일 재주입 + B9 gate + C7 블링크 ramp.
        GLint uTexelSize = -1;
        GLint uGateThreshold = -1;
        GLint uDetailReinject = -1;
        GLint uLowLightActive = -1;  // P7-W2 §5.4: gate 전용 저조도 래치 상태 (0..1)
        GLint uLeftRenderAlpha = -1;
        GLint uRightRenderAlpha = -1;
    } lens_uniforms_;

    void cacheLensUniforms();

    // ========================================
    // OneEuroFilter 스무딩
    // ========================================
    static constexpr float FILTER_MIN_CUTOFF = 4.0f;
    static constexpr float FILTER_BETA = 15.0f;
    static constexpr float FILTER_BETA_EYELID = 12.0f;
    static constexpr float FILTER_D_CUTOFF = 1.0f;

    // 눈꺼풀 필터 (좌/우 x 상/하 = 4개)
    OneEuroFilter eyelid_top_filters_[2]{
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF},
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF}
    };
    OneEuroFilter eyelid_bottom_filters_[2]{
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF},
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF}
    };

    // 타원 필터 (좌/우 x 6파라미터 = 12개)
    struct EllipseFilters {
        OneEuroFilter cx{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter cy{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter rx_inner{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter rx_outer{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter ry{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter rotation{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
    };
    EllipseFilters ellipse_filters_[2];  // [0]=left, [1]=right

    // EYECLIP A-2: contour 16점 per-point 필터 (ellipse와 동일 소스 랜드마크 → 동일 계수)
    struct ContourFilters {
        OneEuroFilter x[16];
        OneEuroFilter y[16];
        ContourFilters() {
            for (int i = 0; i < 16; ++i) {
                x[i] = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF);
                y[i] = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF);
            }
        }
    };
    ContourFilters contour_filters_[2];  // [0]=left, [1]=right

    // ========================================
    // 눈꺼풀 캐시 (홀드 프레임)
    // ========================================
    static constexpr int EYELID_HOLD_FRAMES = 5;

    struct EyelidCache {
        float top = 0.0f;
        float bottom = 1.0f;
        int valid_frames = 0;
    };
    EyelidCache eyelid_cache_[2];  // [0]=left, [1]=right

    struct EllipseCache {
        float cx = 0.0f, cy = 0.0f;
        float rx_inner = 0.0f, rx_outer = 0.0f;
        float ry = 0.0f, rotation = 0.0f;
        int valid_frames = 0;
    };
    EllipseCache ellipse_cache_[2];

    // EYECLIP A-2: contour 16점 캐시 (raw 이미지 공간 저장 — 변환은 업로드 시점만)
    struct ContourCache {
        float x[16];
        float y[16];
        int valid_frames = 0;
    };
    ContourCache contour_cache_[2];  // [0]=left, [1]=right

    // ========================================
    // per-eye 마지막 정상 홍채 pose hold (BUGFIX: 눈 일부 감김 시 렌즈 드롭 방지)
    // ========================================
    // 반쯤 감은 눈은 MediaPipe 홍채 5점이 [0,1] 밖으로 튀어 left/right_detected=false가
    // 되지만, 눈꺼풀(face mesh)은 계속 추적된다. 드롭 프레임의 현재 홍채 중심은 garbage라
    // 못 쓰므로 마지막 정상 pose를 hold해 렌즈를 제자리에 유지하고, 셰이더 Y-slab이
    // 보이는 영역으로 클리핑하게 한다. (public detected 의미는 불변 — render-only)
    // 좌표 규약은 iris_result.left_iris[0]와 동일(정규화, Y-flip 前 raw). 반경은 정규화값.
    // hold 수명은 eyelid_cache_.valid_frames에 연동(얼굴이 EYELID_HOLD_FRAMES 동안
    // 사라지면 함께 정리) — 별도 카운터 없이 캐시 수명 재사용(과설계 회피).
    float held_iris_cx_[2] = {0.0f, 0.0f};
    float held_iris_cy_[2] = {0.0f, 0.0f};
    float held_iris_r_[2]  = {0.0f, 0.0f};
    bool  has_held_pose_[2] = {false, false};

    // [BUGFIX] blink-ramp(render_alpha fade) 토글 — 기본 OFF(실기기 결정).
    //   가시성은 셰이더 eyelidMask(눈꺼풀 클립)만으로 제어한다. 실기기 실측 결과:
    //   (1) 클립이 반쯤 감음을 깔끔히 처리하고, (2) 완전 감음 시 MediaPipe 눈꺼풀 잔여 gap으로
    //   얇은 띠만 남는 경미한 한계가 있으나, (3) eye_opening 신호가 0.005~0.012로 작고 프레임
    //   노이즈(±0.002)가 커 fade 임계값을 안정적으로 잡을 수 없다(뜬 눈도 투명해지는 부작용).
    //   → fade를 끄고 클립에 일임하는 게 단순·견고. (원래 "반쯤 감으면 렌즈 사라짐" 버그의
    //   근본 트리거가 이 fade였다 — held/반쯤 감음에서 과발동.)
    bool blink_ramp_enabled_ = false;

    // 이전 출력 텍스처 추적
    GLuint previous_output_texture_ = 0;

    // ========================================
    // P6-W1: avg_iris_luma fallback chain (실측 source는 W6 이관)
    // ========================================
    // 실측(self-measure)은 Android 카메라 텍스처(EXTERNAL_OES) + 임시 FBO attach +
    // glReadPixels 조합이 GL state 오염을 일으켜 검은 화면 회귀 발생 → revert.
    // 정식 측정 경로는 W6에서 비동기 PBO readback or detector CPU 버퍼 활용.
    // 현 단계는 packet.avg_iris_luma(미연결) → hold → fallback 상수 3단만 동작.
    // W1 §5.2.1: uAvgIrisLum은 linear 공간 값 (Rec.709 linear 평균 luma).
    // 0.1225 = 0.35² — Codex R2의 sRGB 0.35 추정치를 감마 2.0 근사로 linear 변환한 등가값.
    // (W2 squaring 제거 후 시각 회귀 fix — 이전 squaring 코드의 결과 0.35*0.35와 동등.)
    static constexpr float kAvgLumaFallback      = 0.1225f;
    static constexpr int   kAvgLumaMaxHoldFrames = 3;
    static constexpr float kAvgLumaClampMin      = 0.01f;  // linear 하한 (sRGB 0.1 등가)
    static constexpr float kAvgLumaClampMax      = 0.81f;  // linear 상한 (sRGB 0.9 등가)

    float current_avg_luma_    = kAvgLumaFallback;
    int   avg_luma_hold_count_ = 0;
    bool  avg_luma_has_valid_  = false;

    // P6-W2 §5.9: invalid blend ID(3/4/6/etc.) 1회 경고 (debug 빌드 한정).
    //   유효 ID = {0, 1, 2, 5, 7}. 그 외는 셰이더에서 TintLinearV2 fallback.
    bool invalid_blend_warned_ = false;

    // ========================================
    // P6-W6: 블링크 ramp(C7) + 저조도 gate(B9) + 디테일 재주입(C10) 토글 상태
    // ========================================
    // C7 블링크 시간적 envelope (실측 dt 기반 EMA). [0]=left, [1]=right.
    float render_alpha_[2] = {1.0f, 1.0f};
    std::chrono::steady_clock::time_point last_render_ts_;
    bool has_last_ts_ = false;

    // B5/B9/C10 런타임 토글. 기본은 W6 §5.5/§5.7 중간값 (실기기 벤치로 확정).
    float blink_up_ms_   = 80.0f;   // B5 up ramp 95% 도달 시간 (60/80/120)
    // B9 저조도 gate 임계값. 기본 0.10 — 저조도 사용 시나리오가 드문 뷰티 시뮬레이션
    // 특성상 C10 디테일을 일반 환경에서 항상 살리는 쪽 채택(도메인 판단). gate 로직은
    // 보존되어 실측 연결 시 극단 저조도(luma<0.07)만 자동 감쇄.
    float gate_threshold_ = 0.10f;  // B9 토글 후보 0.10/0.15/0.25
    bool  detail_reinject_ = true;  // C10 on/off

    // P7-W2 §5.6: 실측 luma A/B 토글. false면 packet 실측을 무시하고 fallback 0.1225만.
    // 기본 true(정식 ON) — S23+ 실기기 검증 완료(2026-06-10): 무회귀 개선
    //   (어두움=fallback과 clamp 7.0 동일, 밝음=over-tint 교정 scale 7.0→1.27, 육안 자연 확인).
    //   토글로 fallback 롤백 가능. cross-tier(MID/LOW)는 P7-W5에서 재확인.
    bool  use_measured_luma_ = true;

    // P7-W2 §5.4: 저조도 gate 전용 dual-threshold 래치(hysteresis). enter<0.08→true,
    // exit>0.12→false. ⚠️ gate(uLowLightActive)에만 영향. uAvgIrisLum(블렌드 정규화
    // 분모)은 raw EMA값 그대로 — 히스테리시스로 변조 금지(블렌드 깨짐).
    bool  is_low_light_ = false;
    static constexpr float kLowLightEnter = 0.08f;  // 이 미만이면 저조도 진입
    static constexpr float kLowLightExit  = 0.12f;  // 이 초과면 저조도 해제

    // W6 §1.3: down ramp는 고정(생리적 눈 감김이 뜸보다 빠름).
    static constexpr float kBlinkDownMs        = 60.0f;
    // eyeOpening 이하면 눈 감김 판정 (blink_ramp_enabled_ ON일 때만 사용 — 기본 OFF).
    static constexpr float kBlinkCloseThreshold = 0.02f;
    static constexpr float kDtClampMinMs       = 1.0f;
    static constexpr float kDtClampMaxMs       = 100.0f;  // 앱 복귀 등 큰 dt 튐 방지
    static constexpr float kDefaultDtMs        = 16.67f;  // 첫 프레임 dt

    bool initialized_ = false;
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_GPU_LENS_RENDERER_H
