# IrisLensSDK - Tests CMakeLists.txt
# TDD 기반 테스트 구성

# GoogleTest 설정
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# Windows에서 shared CRT 문제 방지
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# 공통 테스트 설정
set(TEST_COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# ============================================================
# P1-W3-02: types.h 테스트
# ============================================================
add_executable(test_types test_types.cpp)
target_include_directories(test_types PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_types PRIVATE
    GTest::gtest
    GTest::gtest_main
)
# add_test는 gtest_discover_tests가 대체함

# ============================================================
# P7-W2: iris ROI luma 측정 색공간(srgb²+Rec.709) 정합 테스트
# (self-contained reference, SDK/OpenCV 비의존)
# ============================================================
add_executable(test_iris_luma_measure test_iris_luma_measure.cpp)
target_include_directories(test_iris_luma_measure PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_iris_luma_measure PRIVATE
    GTest::gtest
    GTest::gtest_main
)

# ④ W4-D: test_iris_detector / test_mediapipe_detector(+_performance/_integration)
#   타깃 제거 — detector 인프라(iris_detector·mediapipe_detector·inference_thread)
#   물리 제거에 동반.

# ============================================================
# LensRenderer 테스트
# ============================================================
add_executable(test_lens_renderer test_lens_renderer.cpp)
target_include_directories(test_lens_renderer PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_lens_renderer PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (LensRenderer는 OpenCV 필수)
if(OpenCV_FOUND)
    target_link_libraries(test_lens_renderer PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_lens_renderer PRIVATE IRIS_SDK_HAS_OPENCV)
else()
    message(WARNING "LensRenderer tests require OpenCV. Tests will be disabled.")
endif()

# ④ W4-D: test_lens_renderer_integration 타깃 제거 — 이 통합 테스트는 SetUp부터
#   MediaPipeDetector(detector_->detect())로 검출 결과를 얻어 렌더링하는 구조라
#   detector 제거 시 본질적으로 성립 불가. 순수 LensRenderer 단위 테스트는
#   test_lens_renderer가 커버.

# ============================================================
# 테스트 디스커버리
# ============================================================
include(GoogleTest)
gtest_discover_tests(test_types DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_iris_luma_measure DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_lens_renderer DISCOVERY_TIMEOUT 60)

# ④ W4-D: test_frame_processor 타깃 제거 — detector/검출 경로 의존 테스트.
#   render-only FrameProcessor의 렌더 경로는 test_lens_renderer*가 커버.

# ============================================================
# P1-W4-04: SDKManager 테스트
# ============================================================
add_executable(test_sdk_manager test_sdk_manager.cpp)
target_include_directories(test_sdk_manager PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_sdk_manager PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (팩토리 메서드 테스트에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_sdk_manager PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_sdk_manager PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_sdk_manager DISCOVERY_TIMEOUT 60)

# ============================================================
# P1-W4-07: CameraDemo 기본 테스트
# ============================================================
add_executable(test_camera_demo test_camera_demo.cpp)
target_include_directories(test_camera_demo PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_camera_demo PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (필수)
if(OpenCV_FOUND)
    target_link_libraries(test_camera_demo PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_camera_demo PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_camera_demo DISCOVERY_TIMEOUT 60)

# ============================================================
# P1-W4-05: C API 래퍼 테스트
# ============================================================
add_executable(test_sdk_api test_sdk_api.cpp)
target_include_directories(test_sdk_api PRIVATE ${TEST_COMMON_INCLUDES})
target_compile_definitions(test_sdk_api PRIVATE IRIS_SDK_TEST_HOOKS)
target_link_libraries(test_sdk_api PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (팩토리 메서드 테스트에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_sdk_api PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_sdk_api PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_sdk_api DISCOVERY_TIMEOUT 60)

# ④ W4-D: test_integration 타깃 제거 — 전체 C API 검출 파이프라인(iris_sdk_detect/
#   process/submit) 의존 통합 테스트. 검출 인프라 제거에 동반.

# ============================================================
# P2-W1-01: RenderContext 테스트 (GPU 모듈)
# ============================================================
add_executable(test_render_context test_render_context.cpp)
target_include_directories(test_render_context PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_render_context PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (CPURenderContext에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_render_context PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_render_context PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_render_context DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W1-02: BeautyFilterConfigV2 테스트
# ============================================================
add_executable(test_beauty_config_v2 test_beauty_config_v2.cpp)
target_include_directories(test_beauty_config_v2 PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_beauty_config_v2 PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (BeautyFilter 구현에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_beauty_config_v2 PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_beauty_config_v2 PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_beauty_config_v2 DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W1-03: BeautyROIManager 테스트
# ============================================================
add_executable(test_beauty_roi_manager test_beauty_roi_manager.cpp)
target_include_directories(test_beauty_roi_manager PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_beauty_roi_manager PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (마스크 생성에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_beauty_roi_manager PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_beauty_roi_manager PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_beauty_roi_manager DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W1-04: BeautyProcessor 테스트
# ============================================================
add_executable(test_beauty_processor test_beauty_processor.cpp)
target_include_directories(test_beauty_processor PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_beauty_processor PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (CPUBeautyBackend에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_beauty_processor PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_beauty_processor PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_beauty_processor DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W4-01: GridMesh 테스트 (Face Warp)
# ============================================================
add_executable(test_grid_mesh test_grid_mesh.cpp)
target_include_directories(test_grid_mesh PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_grid_mesh PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_grid_mesh DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W3-01: GPUBeautyBackend 테스트
# ============================================================
add_executable(test_gpu_beauty_backend test_gpu_beauty_backend.cpp)
target_include_directories(test_gpu_beauty_backend PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_gpu_beauty_backend PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_gpu_beauty_backend DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W2-02: FastGuidedFilter 테스트
# ============================================================
add_executable(test_fast_guided_filter test_fast_guided_filter.cpp)
target_include_directories(test_fast_guided_filter PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_fast_guided_filter PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (FastGuidedFilter에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_fast_guided_filter PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_fast_guided_filter PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_fast_guided_filter DISCOVERY_TIMEOUT 120)

# ============================================================
# P2-W2-03: NewFilterEffects 테스트 (V2 필터)
# ============================================================
add_executable(test_new_filter_effects test_new_filter_effects.cpp)
target_include_directories(test_new_filter_effects PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_new_filter_effects PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (CPUBeautyBackend에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_new_filter_effects PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_new_filter_effects PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_new_filter_effects DISCOVERY_TIMEOUT 120)

# ============================================================
# P2-W4-02: FaceWarpController 테스트 (Slim Face / V-Line)
# ============================================================
add_executable(test_face_warp_controller test_face_warp_controller.cpp)
target_include_directories(test_face_warp_controller PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_face_warp_controller PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_face_warp_controller DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W4-03: Eye Enlargement 테스트
# ============================================================
add_executable(test_eye_enlargement test_eye_enlargement.cpp)
target_include_directories(test_eye_enlargement PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_eye_enlargement PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_eye_enlargement DISCOVERY_TIMEOUT 60)

# ============================================================
# P2-W5-02: Profiler 및 BufferPool 테스트
# ============================================================
add_executable(test_profiler test_profiler.cpp)
target_include_directories(test_profiler PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_profiler PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (BufferPool에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_profiler PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_profiler PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_profiler DISCOVERY_TIMEOUT 120)

# ============================================================
# P4-W3-04: OneEuroFilter 단위 테스트
# ============================================================
add_executable(test_one_euro_filter test_one_euro_filter.cpp)
target_include_directories(test_one_euro_filter PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_one_euro_filter PRIVATE
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(test_one_euro_filter DISCOVERY_TIMEOUT 60)

# ============================================================
# P8-W1: 피부 마스크 이마 확장 순수 함수 테스트
# ============================================================
add_executable(test_skin_mask_geometry test_skin_mask_geometry.cpp)
target_include_directories(test_skin_mask_geometry PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_skin_mask_geometry PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_skin_mask_geometry DISCOVERY_TIMEOUT 60)

# ============================================================
# ③-1: 랜드마크 주입 경계 — 파생 어댑터 + seqlock 더블버퍼 테스트 (ADR-0001 §6)
# ============================================================
add_executable(test_landmark_injection test_landmark_injection.cpp)
target_include_directories(test_landmark_injection PRIVATE ${TEST_COMMON_INCLUDES})
# 골든 baseline JSON 경로를 컴파일 타임 매크로로 주입(어댑터 동작 불변 대조용).
target_compile_definitions(test_landmark_injection PRIVATE
    GOLDEN_BASELINE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden/baseline")
target_link_libraries(test_landmark_injection PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# ③-3: C API 표면 경유 E2E(iris_set_landmarks/iris_get_injected_result). ④ W4-D에서
# detector 코어가 제거되어 TFLite 링크는 더 이상 필요 없다(코어 TFLite-free).
if(OpenCV_FOUND)
    target_link_libraries(test_landmark_injection PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_landmark_injection PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_landmark_injection DISCOVERY_TIMEOUT 60)

# ============================================================
# ④ W4-D 안전망: 18골든 전체 injection-derive 회전 정합 검증
# (detector 제거 後 — golden_capture --inject-from 경로가 detector 베이스라인과
#  ε 일치함을, 회전 케이스 포함 전체에서 증명. C API(iris_set_landmarks/
#  iris_get_injected_result) 경유. detector 제거로 TFLite 링크 불필요.)
# ============================================================
add_executable(test_golden_injection_derive test_golden_injection_derive.cpp)
target_include_directories(test_golden_injection_derive PRIVATE ${TEST_COMMON_INCLUDES})
target_compile_definitions(test_golden_injection_derive PRIVATE
    GOLDEN_BASELINE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden/baseline")
target_link_libraries(test_golden_injection_derive PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)
if(OpenCV_FOUND)
    target_link_libraries(test_golden_injection_derive PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_golden_injection_derive PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_golden_injection_derive DISCOVERY_TIMEOUT 60)

# ============================================================
# P4-W3-04: DeviceTier 분류 + FreqSep 매핑 테스트
# ============================================================
add_executable(test_device_tier test_device_tier.cpp)
target_include_directories(test_device_tier PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_device_tier PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_device_tier DISCOVERY_TIMEOUT 60)

# ============================================================
# P4-W3-05: 품질 측정/튜닝 모듈 테스트
# (QualityMetrics, TemporalAnalyzer, ABCompare, ReleaseGate, ParamTuner)
# ============================================================
add_executable(test_quality_tuning test_quality_tuning.cpp)
target_include_directories(test_quality_tuning PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_quality_tuning PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (QualityMetrics, ABCompare에 필수)
if(OpenCV_FOUND)
    target_link_libraries(test_quality_tuning PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_quality_tuning PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

gtest_discover_tests(test_quality_tuning DISCOVERY_TIMEOUT 120)

# ============================================================
# P5-W1: TemporalStabilizer 단위 테스트
# ============================================================
add_executable(test_temporal_stabilizer test_temporal_stabilizer.cpp)
target_include_directories(test_temporal_stabilizer PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_temporal_stabilizer PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_temporal_stabilizer DISCOVERY_TIMEOUT 60)

# ============================================================
# P5-W1: TemporalStabilizer 통합/안정성 메트릭 테스트
# ============================================================
add_executable(test_temporal_stability test_temporal_stability.cpp)
target_include_directories(test_temporal_stability PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_temporal_stability PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

gtest_discover_tests(test_temporal_stability DISCOVERY_TIMEOUT 120)

# ④ W4-D: test_inference_thread_lifecycle 타깃 제거 —
#   InferenceThread(검출 추론 스레드) 물리 제거에 동반.
