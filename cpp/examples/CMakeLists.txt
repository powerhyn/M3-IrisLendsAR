# IrisLensSDK - Examples CMakeLists.txt
# 예제 애플리케이션 빌드 설정

# ============================================
# Hello Iris (기본 예제)
# ============================================

# Placeholder example for initial build verification
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/hello_iris.cpp
"// Placeholder example
#include <iostream>

// Forward declaration (will be in iris_sdk.h)
namespace iris_sdk {
    const char* get_version();
}

int main() {
    std::cout << \"IrisLensSDK v\" << iris_sdk::get_version() << std::endl;
    return 0;
}
")

add_executable(hello_iris hello_iris.cpp)
target_link_libraries(hello_iris PRIVATE iris_sdk)

# OpenCV가 필요함
find_package(OpenCV QUIET COMPONENTS core imgproc highgui videoio)

if(OpenCV_FOUND)
    # ④ W4-D: camera_demo / image_demo 제거 — 검출 경로(iris_sdk_detect/process)
    #   의존 인터랙티브 데모. 검출 인프라 제거에 동반. golden_capture는 주입 경로
    #   (iris_set_landmarks/iris_get_injected_result)로 전환되어 보존된다.
    set(MODEL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../shared/models")

    # ============================================
    # Golden Capture (골든 베이스라인 캡처 도구, 계획 2-2)
    # ============================================
    # 표준 입력 세트에 대해 검출 결과(JSON) + 렌즈 렌더/CPU beauty(PNG)를
    # 결정적으로 캡처한다. ④ W4-D 이후 주입 경로(--inject-from)만 사용한다:
    # 소스 baseline 478점을 iris_set_landmarks로 주입하고 iris_get_injected_result로 재파생.
    #
    # 사용법:
    #   golden_capture --input <png> --inject-from <baseline.json> --texture <png> --out <dir>
    #                  [--rotation {0|90|180|270}] [--mirror {0|1}] [--gamma <float>]
    #                  [--out-stem <name>] [--no-beauty]

    add_executable(golden_capture golden_capture.cpp)

    target_link_libraries(golden_capture
        PRIVATE
            iris_sdk
            ${OpenCV_LIBS}
    )

    target_include_directories(golden_capture
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
    )

    # C++17 필요 (std::filesystem)
    target_compile_features(golden_capture PRIVATE cxx_std_17)

    # macOS 특화 설정
    if(APPLE)
        target_link_options(golden_capture PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()

    # 모델 파일 복사 (빌드 후) — image_demo와 동일 패턴
    if(EXISTS "${MODEL_SOURCE_DIR}")
        add_custom_command(TARGET golden_capture POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:golden_capture>/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${MODEL_SOURCE_DIR}"
                $<TARGET_FILE_DIR:golden_capture>/models
            COMMENT "Copying model files for golden_capture..."
        )
    endif()

    message(STATUS "golden_capture: Enabled (OpenCV ${OpenCV_VERSION})")
else()
    message(STATUS "golden_capture: Disabled (OpenCV not found)")
endif()
