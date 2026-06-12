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

# ============================================
# Camera Demo (웹캠 실시간 데모)
# ============================================
# 웹캠을 사용하여 실시간 홍채 검출 및 렌즈 오버레이 데모
#
# 요구사항:
# - OpenCV (VideoIO, HighGUI, ImgProc)
# - 웹캠 (macOS: AVFoundation 지원)
#
# 키보드 조작:
#   ESC/Q  - 종료
#   SPACE  - 렌즈 텍스처 변경
#   L      - 랜드마크 표시 토글
#   F      - FPS 표시 토글
#   D      - 디버그 정보 토글
#   +/-    - 투명도 조절
#   [/]    - 크기 조절
#   S      - 스크린샷 저장
#   R      - 설정 초기화

# OpenCV가 필요함
find_package(OpenCV QUIET COMPONENTS core imgproc highgui videoio)

if(OpenCV_FOUND)
    add_executable(camera_demo camera_demo.cpp)

    target_link_libraries(camera_demo
        PRIVATE
            iris_sdk
            ${OpenCV_LIBS}
    )

    target_include_directories(camera_demo
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
    )

    # C++17 필요 (std::filesystem)
    target_compile_features(camera_demo PRIVATE cxx_std_17)

    # macOS 특화 설정
    if(APPLE)
        # AVFoundation 프레임워크 (카메라 액세스)
        target_link_libraries(camera_demo
            PRIVATE
                "-framework AVFoundation"
                "-framework CoreMedia"
                "-framework CoreVideo"
        )

        # TFLite 링크 옵션 (undefined 심볼 허용 - iris_sdk 내부 의존성)
        target_link_options(camera_demo PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )

        # 카메라 권한 Info.plist (옵션)
        set_target_properties(camera_demo PROPERTIES
            MACOSX_BUNDLE FALSE
        )
    endif()

    # 모델 파일 복사 (빌드 후)
    # 모델 경로: shared/models/ (프로젝트 루트 기준)
    set(MODEL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../shared/models")
    if(EXISTS "${MODEL_SOURCE_DIR}")
        add_custom_command(TARGET camera_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:camera_demo>/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${MODEL_SOURCE_DIR}"
                $<TARGET_FILE_DIR:camera_demo>/models
            COMMENT "Copying model files for camera_demo..."
        )
    else()
        message(WARNING "Model directory not found: ${MODEL_SOURCE_DIR}")
    endif()

    message(STATUS "camera_demo: Enabled (OpenCV ${OpenCV_VERSION})")

    # ============================================
    # Image Demo (단일 이미지 처리 데모)
    # ============================================
    # 커맨드라인에서 이미지 파일을 입력받아 홍채 검출 및 렌즈 오버레이
    #
    # 사용법:
    #   image_demo <input_image> [output_image]
    #
    # 예시:
    #   image_demo face.jpg                    # face_output.jpg로 저장
    #   image_demo face.jpg result.png         # result.png로 저장

    add_executable(image_demo image_demo.cpp)

    target_link_libraries(image_demo
        PRIVATE
            iris_sdk
            ${OpenCV_LIBS}
    )

    target_include_directories(image_demo
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
    )

    # C++17 필요 (std::filesystem)
    target_compile_features(image_demo PRIVATE cxx_std_17)

    # macOS 특화 설정
    if(APPLE)
        target_link_options(image_demo PRIVATE
            "-Wl,-undefined,dynamic_lookup"
        )
    endif()

    # 모델 파일 복사 (빌드 후)
    if(EXISTS "${MODEL_SOURCE_DIR}")
        add_custom_command(TARGET image_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:image_demo>/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${MODEL_SOURCE_DIR}"
                $<TARGET_FILE_DIR:image_demo>/models
            COMMENT "Copying model files for image_demo..."
        )
    endif()

    message(STATUS "image_demo: Enabled (OpenCV ${OpenCV_VERSION})")

    # ============================================
    # Golden Capture (골든 베이스라인 캡처 도구, 계획 2-2)
    # ============================================
    # 표준 입력 세트에 대해 검출 결과(JSON) + 렌즈 렌더/CPU beauty(PNG)를
    # 결정적으로 캡처한다. 회전 경로(blocker)를 골든이 커버하기 위해
    # 이미지를 실제 회전시킨 뒤 iris_sdk_detect_with_rotation에 전달한다.
    #
    # 사용법:
    #   golden_capture --input <png> --models <dir> --texture <png> --out <dir>
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
    message(STATUS "camera_demo: Disabled (OpenCV not found)")
    message(STATUS "image_demo: Disabled (OpenCV not found)")
    message(STATUS "golden_capture: Disabled (OpenCV not found)")
endif()
