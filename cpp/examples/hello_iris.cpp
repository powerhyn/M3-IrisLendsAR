// Placeholder example
#include <iostream>

// Forward declaration (will be in iris_sdk.h)
namespace iris_sdk {
    const char* get_version();
}

int main() {
    std::cout << "IrisLensSDK v" << iris_sdk::get_version() << std::endl;
    return 0;
}
