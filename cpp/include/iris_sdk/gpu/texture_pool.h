/**
 * @file texture_pool.h
 * @brief GPU 텍스처 및 프레임버퍼 풀링 관리
 *
 * 렌더 타겟용 텍스처와 FBO를 풀링하여 할당/해제 오버헤드를 줄이고
 * 메모리 사용을 최적화합니다.
 */

#ifndef IRIS_SDK_TEXTURE_POOL_H
#define IRIS_SDK_TEXTURE_POOL_H

#include "shader_manager.h"  // GPU_AVAILABLE 매크로 및 GL 타입 정의

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if IRIS_SDK_GPU_AVAILABLE
// GL 상수 정의는 shader_manager.h에서 include됨
#else
// Desktop 스텁용 GL 상수
#ifndef GL_RGBA
#define GL_RGBA 0x1908
#endif
#ifndef GL_FRAMEBUFFER_COMPLETE
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif
#endif

namespace iris_sdk {

/**
 * @brief GPU 텍스처 풀
 *
 * 렌더 타겟용 텍스처와 FBO를 사전 할당하고 재사용하여
 * 런타임 메모리 할당 오버헤드를 최소화합니다.
 *
 * Thread-safe: 모든 public 메서드는 mutex로 보호됩니다.
 */
class TexturePool {
public:
    /**
     * @brief 텍스처 정보 구조체
     */
    struct TextureInfo {
        GLuint texture_id = 0;  ///< GL 텍스처 ID
        GLuint fbo_id = 0;      ///< 연결된 FBO ID (렌더 타겟용)
        int width = 0;          ///< 텍스처 너비
        int height = 0;         ///< 텍스처 높이
        GLenum format = GL_RGBA; ///< 픽셀 포맷
        bool in_use = false;    ///< 사용 중 여부

        /// 메모리 크기 계산 (바이트)
        size_t memorySize() const {
            int bpp = (format == GL_RGBA) ? 4 : 3;
            return static_cast<size_t>(width) * height * bpp;
        }

        /// 유효성 검사
        bool isValid() const { return texture_id != 0; }
    };

    /**
     * @brief 풀 통계 정보
     */
    struct PoolStats {
        int total_textures = 0;      ///< 총 텍스처 수
        int in_use = 0;              ///< 사용 중인 텍스처 수
        int available = 0;           ///< 사용 가능한 텍스처 수
        size_t total_memory_bytes = 0; ///< 총 메모리 사용량
    };

    TexturePool();
    ~TexturePool();

    // 복사/이동 금지
    TexturePool(const TexturePool&) = delete;
    TexturePool& operator=(const TexturePool&) = delete;
    TexturePool(TexturePool&&) = delete;
    TexturePool& operator=(TexturePool&&) = delete;

    //=========================================================================
    // 초기화/해제
    //=========================================================================

    /**
     * @brief 풀 초기화
     *
     * @param max_textures 최대 텍스처 수 (1-16)
     * @param max_width 최대 텍스처 너비
     * @param max_height 최대 텍스처 높이
     * @return 성공 시 true
     */
    bool initialize(int max_textures, int max_width, int max_height);

    /**
     * @brief 모든 리소스 해제
     */
    void release();

    /**
     * @brief 초기화 여부 확인
     */
    bool isInitialized() const { return initialized_; }

    //=========================================================================
    // 텍스처 획득/반환
    //=========================================================================

    /**
     * @brief 렌더 타겟용 텍스처 획득
     *
     * 요청 크기에 맞는 텍스처를 풀에서 찾거나 새로 생성합니다.
     *
     * @param width 요청 너비
     * @param height 요청 높이
     * @return 텍스처 정보 포인터 (실패 시 nullptr)
     *
     * @note 반환된 포인터는 releaseTexture()로 반환해야 합니다.
     */
    TextureInfo* acquireRenderTarget(int width, int height);

    /**
     * @brief 텍스처 반환
     *
     * @param info 반환할 텍스처 정보
     */
    void releaseTexture(TextureInfo* info);

    /**
     * @brief 텍스처 ID로 풀 관리 텍스처 반환 (조회+반환 원자화)
     *
     * texture_id로 풀에서 관리하는 텍스처를 찾아 반환합니다.
     * 외부에 TextureInfo 포인터를 노출하지 않으므로 댕글링 리스크가 없습니다.
     *
     * @param texture_id GL 텍스처 ID
     * @return true if texture was found and released, false if not managed by pool
     */
    bool releaseTextureById(GLuint texture_id);

    /**
     * @brief Ping-Pong 버퍼 쌍 획득
     *
     * 필터 체이닝을 위한 두 개의 텍스처를 동시에 획득합니다.
     *
     * @param width 요청 너비
     * @param height 요청 높이
     * @param ping 첫 번째 텍스처 (출력)
     * @param pong 두 번째 텍스처 (출력)
     * @return 성공 시 true
     */
    bool acquirePingPongPair(int width, int height,
                             TextureInfo*& ping, TextureInfo*& pong);

    /**
     * @brief 모든 텍스처를 사용 가능 상태로 반환
     */
    void releaseAllTextures();

    //=========================================================================
    // 메모리 관리
    //=========================================================================

    /**
     * @brief 미사용 텍스처 정리
     *
     * 지정 시간 이상 사용되지 않은 텍스처를 해제하여 메모리를 절약합니다.
     *
     * @param max_idle_ms 최대 유휴 시간 (밀리초, 기본 5000ms)
     * @return 해제된 텍스처 수
     */
    int trim(int64_t max_idle_ms = 5000);

    /**
     * @brief 풀 크기 동적 조정
     *
     * 저사양 기기에서 OOM 방지를 위해 풀 크기를 줄일 수 있습니다.
     *
     * @param new_max_textures 새로운 최대 텍스처 수 (1-16)
     */
    void resizePool(int new_max_textures);

    /**
     * @brief 입력 크기 기반 텍스처 크기 상한 보장 (B2 idx5)
     *
     * 호스트가 portrait(예: 1080x1920)나 고해상도 텍스처를 넘기면 기존의
     * 하드코딩 상한(예: 1920x1080)을 초과하여 acquireRenderTarget이 매 프레임
     * silent 실패(nullptr 반환 → 파이프라인 전체 무력화)하던 결함을 막는다.
     *
     * 요청 크기가 현재 상한을 넘으면 상한을 요청 크기까지 확장한다(기존 텍스처는
     * lazy 생성이므로 즉시 재할당하지 않음). 상한을 줄이지는 않는다.
     *
     * @param width  요청 텍스처 너비
     * @param height 요청 텍스처 높이
     * @return 상한이 변경되었으면 true (신규/확장), 변경 없으면 false
     */
    bool ensureCapacity(int width, int height);

    /**
     * @brief 현재 메모리 사용량 조회
     *
     * @return 사용 중인 텍스처 메모리 (바이트)
     */
    size_t getUsedMemory() const;

    /**
     * @brief 풀 상태 조회
     */
    PoolStats getStats() const;

    //=========================================================================
    // 메모리 압력 콜백
    //=========================================================================

    /**
     * @brief 메모리 압력 콜백 타입
     *
     * Android onTrimMemory() 이벤트와 연동하여 자동으로 메모리를 정리합니다.
     * @param level Android ComponentCallbacks2 트림 레벨
     */
    using MemoryPressureCallback = std::function<void(int level)>;

    /**
     * @brief 메모리 압력 콜백 설정
     */
    void setMemoryPressureCallback(MemoryPressureCallback callback);

    /**
     * @brief 메모리 압력 알림 처리
     *
     * @param level Android trim level (10=RUNNING_LOW, 15=CRITICAL,
     *              40=BACKGROUND, 60=MODERATE, 80=COMPLETE)
     *
     * @warning [B2 idx22] 스레드 친화성 결함 — 현재 미연결(죽은 경로).
     *   이 메서드는 trim()/resizePool()을 통해 glDeleteFramebuffers/glDeleteTextures를
     *   직접 호출한다. 그러나 Android onTrimMemory()는 main 스레드 콜백이라 GL 컨텍스트가
     *   current가 아니다 — mutex_는 데이터 경합만 막을 뿐 GL 호출의 스레드 유효성을
     *   보장하지 못한다. **GL을 호출하는 모든 메서드는 GL 스레드에서만 호출해야 한다.**
     *
     *   따라서 main 스레드 onTrimMemory()에 직접 연결하면 안 된다(GL no-op로 리소스
     *   누수 또는 다른 컨텍스트 객체 오삭제). 올바른 연동은 '정리 요청 플래그만 세우고
     *   실제 glDelete는 다음 GL 스레드 진입(applyTextureId) 시 수행'하는 지연 정리
     *   패턴이며, 이는 ④ 단계에서 설계한다. 현재는 JNI/Java/sdk_api 어디에서도
     *   호출되지 않으므로 도달 불가하다.
     */
    void onMemoryPressure(int level);

private:
    /// 렌더 타겟 획득 (lock 미획득 전제, 내부 전용)
    TextureInfo* acquireRenderTargetLocked(int width, int height);

    /// 텍스처 반환 (lock 미획득 전제, 내부 전용)
    void releaseTextureLocked(TextureInfo* info);

    /// 새 텍스처 생성
    TextureInfo* createTexture(int width, int height);

    /// 사용 가능한 텍스처 검색
    TextureInfo* findAvailable(int width, int height);

    /// 풀 만석 시 유휴(불일치) 텍스처 1장을 즉시 회수 (GL 리소스 삭제 + 슬롯 제거)
    /// [B2 idx6 후속] 정확 크기 매칭 전환으로 불일치 유휴 텍스처가 풀에 영구
    /// 잔류할 수 있다. trim()의 유일 production 호출처 onMemoryPressure()가
    /// 현재 미연결(죽은 경로)이므로 회수 수단이 없어, 해상도 전환(전/후면 카메라,
    /// 프리뷰 크기 변경) 반복 시 만석으로 신해상도 acquire가 영구 실패할 수 있다.
    /// 이 함수는 GL 스레드에서만 호출되는 acquireRenderTargetLocked에서만 사용되며,
    /// 가장 오래 유휴인(LRU) 텍스처 1장을 evict하여 새 크기 생성 슬롯을 확보한다.
    /// @return 1장이라도 회수하면 true, 회수 가능한 유휴 텍스처가 없으면 false
    bool evictOneIdleLocked();

    /// 현재 시간 (밀리초)
    static int64_t currentTimeMs();

    std::vector<std::unique_ptr<TextureInfo>> textures_;
    std::unordered_map<GLuint, int64_t> last_used_time_;

    int max_textures_ = 8;
    int max_width_ = 1920;
    int max_height_ = 1080;

    // [B2 idx5] 상한 초과 거부 로그 스로틀 — 같은 크기가 매 프레임 거부될 때
    // 로그 스팸을 막되, 새 크기 거부는 1회 명시 로그를 남긴다.
    int last_rejected_w_ = 0;
    int last_rejected_h_ = 0;

    bool initialized_ = false;
    mutable std::mutex mutex_;

    MemoryPressureCallback memory_pressure_callback_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_TEXTURE_POOL_H
