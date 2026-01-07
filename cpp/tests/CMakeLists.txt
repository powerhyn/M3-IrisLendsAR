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
add_test(NAME test_types COMMAND test_types)

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
add_test(NAME test_iris_detector COMMAND test_iris_detector)

# ============================================================
# 테스트 디스커버리 (선택적)
# ============================================================
include(GoogleTest)
gtest_discover_tests(test_types)
gtest_discover_tests(test_iris_detector)
