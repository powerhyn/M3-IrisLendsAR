# IrisLensSDK - Examples CMakeLists.txt

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
