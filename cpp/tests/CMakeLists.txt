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
# P1-W3-01: IrisDetector 인터페이스 테스트
# ============================================================
add_executable(test_iris_detector test_iris_detector.cpp)
target_include_directories(test_iris_detector PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_iris_detector PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)
# add_test는 gtest_discover_tests가 대체함

# ============================================================
# P1-W3-03: MediaPipeDetector 테스트
# ============================================================
add_executable(test_mediapipe_detector test_mediapipe_detector.cpp)
target_include_directories(test_mediapipe_detector PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_mediapipe_detector PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# ============================================================
# MediaPipeDetector 성능 벤치마크 테스트
# ============================================================
add_executable(test_mediapipe_detector_performance test_mediapipe_detector_performance.cpp)
target_include_directories(test_mediapipe_detector_performance PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_mediapipe_detector_performance PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# ============================================================
# MediaPipeDetector 통합 테스트 (실제 TFLite 모델 + 실제 이미지)
# ============================================================
# 조건부 컴파일: TFLite와 OpenCV가 모두 있어야 실행됨
add_executable(test_mediapipe_detector_integration test_mediapipe_detector_integration.cpp)
target_include_directories(test_mediapipe_detector_integration PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_mediapipe_detector_integration PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

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

# OpenCV 연결 (통합 테스트에 필요)
if(OpenCV_FOUND)
    target_link_libraries(test_mediapipe_detector_integration PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_mediapipe_detector_integration PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

# TFLite 정의 전파 및 링크
# 1. TFLITE_FOUND 캐시 변수 확인 (상위 CMakeLists.txt에서 설정)
# 2. tensorflow-lite 타겟 존재 여부 확인
# 3. 빌드된 라이브러리 파일 직접 확인
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_mediapipe_detector_integration PRIVATE IRIS_SDK_HAS_TFLITE)
    message(STATUS "Integration test: TFLite enabled (TFLITE_FOUND=${TFLITE_FOUND})")

    # macOS: TFLite resource 모듈 심볼 미사용 허용
    # (Face Detection/Iris 모델에서 hashtable, variable ops 미사용)
    if(APPLE)
        target_link_options(test_mediapipe_detector_integration PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
        message(STATUS "  - Added macOS linker option for TFLite unused symbols")
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    # 라이브러리가 빌드되어 있지만 타겟이 없는 경우 직접 링크
    target_compile_definitions(test_mediapipe_detector_integration PRIVATE IRIS_SDK_HAS_TFLITE)

    # TensorFlow Lite 및 모든 의존성 라이브러리 수집
    # 주의: libfft2d_fft4f2d.a와 libfft2d_fftsg.a는 중복 심볼 충돌이 있음
    file(GLOB TFLITE_DEPS
        "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a"
        "${CMAKE_BINARY_DIR}/lib/libflatbuffers.a"
        "${CMAKE_BINARY_DIR}/lib/libfft2d_fftsg.a"
        "${CMAKE_BINARY_DIR}/lib/libfft2d_fftsg2d.a"
        "${CMAKE_BINARY_DIR}/lib/libfft2d_fftsg3d.a"
        "${CMAKE_BINARY_DIR}/lib/libfft2d_alloc.a"
        "${CMAKE_BINARY_DIR}/lib/libfft2d_shrtdct.a"
        "${CMAKE_BINARY_DIR}/lib/libfarmhash.a"
        "${CMAKE_BINARY_DIR}/lib/libruy*.a"
        "${CMAKE_BINARY_DIR}/lib/libcpuinfo*.a"
        "${CMAKE_BINARY_DIR}/lib/libXNNPACK.a"
        "${CMAKE_BINARY_DIR}/lib/libpthreadpool.a"
    )

    # Abseil 라이브러리 (TFLite 의존성)
    file(GLOB ABSEIL_LIBS "${CMAKE_BINARY_DIR}/lib/libabsl_*.a")

    target_link_libraries(test_mediapipe_detector_integration PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )

    # macOS: TFLite resource 모듈 심볼이 누락된 경우 (우리 모델에서 미사용)
    # 런타임에 해당 ops가 호출되지 않으면 문제 없음
    if(APPLE)
        target_link_options(test_mediapipe_detector_integration PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()

    list(LENGTH TFLITE_DEPS TFLITE_DEPS_COUNT)
    list(LENGTH ABSEIL_LIBS ABSEIL_LIBS_COUNT)

    message(STATUS "Integration test: TFLite enabled (pre-built library found)")
    message(STATUS "  - Linked libtensorflow-lite.a")
    message(STATUS "  - Linked ${TFLITE_DEPS_COUNT} TFLite dependency libraries")
    message(STATUS "  - Linked ${ABSEIL_LIBS_COUNT} Abseil libraries")
else()
    message(STATUS "Integration test: TFLite disabled (TFLITE_FOUND=${TFLITE_FOUND}, library not found)")
endif()

# ============================================================
# LensRenderer 통합 테스트 (MediaPipeDetector 연동)
# ============================================================
add_executable(test_lens_renderer_integration test_lens_renderer_integration.cpp)
target_include_directories(test_lens_renderer_integration PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_lens_renderer_integration PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (필수)
if(OpenCV_FOUND)
    target_link_libraries(test_lens_renderer_integration PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_lens_renderer_integration PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

# TFLite 연결 (MediaPipeDetector에 필요)
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_lens_renderer_integration PRIVATE IRIS_SDK_HAS_TFLITE)
    if(APPLE)
        target_link_options(test_lens_renderer_integration PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    target_compile_definitions(test_lens_renderer_integration PRIVATE IRIS_SDK_HAS_TFLITE)
    target_link_libraries(test_lens_renderer_integration PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )
    if(APPLE)
        target_link_options(test_lens_renderer_integration PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
endif()

# ============================================================
# 테스트 디스커버리
# ============================================================
include(GoogleTest)
gtest_discover_tests(test_types DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_iris_detector DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_mediapipe_detector DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_mediapipe_detector_performance DISCOVERY_TIMEOUT 120)
gtest_discover_tests(test_mediapipe_detector_integration DISCOVERY_TIMEOUT 180)
gtest_discover_tests(test_lens_renderer DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_lens_renderer_integration DISCOVERY_TIMEOUT 180)

# ============================================================
# FrameProcessor 테스트
# ============================================================
add_executable(test_frame_processor test_frame_processor.cpp)
target_include_directories(test_frame_processor PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_frame_processor PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (필수)
if(OpenCV_FOUND)
    target_link_libraries(test_frame_processor PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_frame_processor PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

# TFLite 연결 (MediaPipeDetector에 필요)
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_frame_processor PRIVATE IRIS_SDK_HAS_TFLITE)
    if(APPLE)
        target_link_options(test_frame_processor PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    target_compile_definitions(test_frame_processor PRIVATE IRIS_SDK_HAS_TFLITE)
    target_link_libraries(test_frame_processor PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )
    if(APPLE)
        target_link_options(test_frame_processor PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
endif()

gtest_discover_tests(test_frame_processor DISCOVERY_TIMEOUT 180)

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

# TFLite 연결 (팩토리 메서드 테스트에 필요)
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_sdk_manager PRIVATE IRIS_SDK_HAS_TFLITE)
    if(APPLE)
        target_link_options(test_sdk_manager PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    target_compile_definitions(test_sdk_manager PRIVATE IRIS_SDK_HAS_TFLITE)
    target_link_libraries(test_sdk_manager PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )
    if(APPLE)
        target_link_options(test_sdk_manager PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
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

# TFLite 연결
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_camera_demo PRIVATE IRIS_SDK_HAS_TFLITE)
    if(APPLE)
        target_link_options(test_camera_demo PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    target_compile_definitions(test_camera_demo PRIVATE IRIS_SDK_HAS_TFLITE)
    target_link_libraries(test_camera_demo PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )
    if(APPLE)
        target_link_options(test_camera_demo PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
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

# TFLite 연결 (통합 테스트에 필요)
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_sdk_api PRIVATE IRIS_SDK_HAS_TFLITE)
    if(APPLE)
        target_link_options(test_sdk_api PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    target_compile_definitions(test_sdk_api PRIVATE IRIS_SDK_HAS_TFLITE)
    target_link_libraries(test_sdk_api PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )
    if(APPLE)
        target_link_options(test_sdk_api PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
endif()

gtest_discover_tests(test_sdk_api DISCOVERY_TIMEOUT 60)

# ============================================================
# P1-W4-06: 통합 테스트 (전체 C API 파이프라인)
# ============================================================
add_executable(test_integration test_integration.cpp)
target_include_directories(test_integration PRIVATE ${TEST_COMMON_INCLUDES})
target_link_libraries(test_integration PRIVATE
    GTest::gtest
    GTest::gtest_main
    iris_sdk
)

# OpenCV 연결 (필수)
if(OpenCV_FOUND)
    target_link_libraries(test_integration PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(test_integration PRIVATE IRIS_SDK_HAS_OPENCV)
endif()

# TFLite 연결 (통합 테스트에 필요)
if(TFLITE_FOUND OR TARGET tensorflow-lite)
    target_compile_definitions(test_integration PRIVATE IRIS_SDK_HAS_TFLITE)
    if(APPLE)
        target_link_options(test_integration PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a")
    target_compile_definitions(test_integration PRIVATE IRIS_SDK_HAS_TFLITE)
    target_link_libraries(test_integration PRIVATE
        ${TFLITE_DEPS}
        ${ABSEIL_LIBS}
    )
    if(APPLE)
        target_link_options(test_integration PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()
endif()

gtest_discover_tests(test_integration DISCOVERY_TIMEOUT 300)

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
