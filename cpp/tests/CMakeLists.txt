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
# 테스트 디스커버리
# ============================================================
include(GoogleTest)
gtest_discover_tests(test_types DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_iris_detector DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_mediapipe_detector DISCOVERY_TIMEOUT 60)
gtest_discover_tests(test_mediapipe_detector_performance DISCOVERY_TIMEOUT 120)
gtest_discover_tests(test_mediapipe_detector_integration DISCOVERY_TIMEOUT 180)
gtest_discover_tests(test_lens_renderer DISCOVERY_TIMEOUT 60)
