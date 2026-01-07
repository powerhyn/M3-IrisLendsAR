/**
 * @file iris_sdk.h
 * @brief IrisLensSDK - Main header file
 *
 * Real-time iris tracking and virtual lens overlay AR SDK
 *
 * @version 0.1.0
 * @copyright 2026
 */

#ifndef IRIS_SDK_H
#define IRIS_SDK_H

// Version info
#define IRIS_SDK_VERSION_MAJOR 0
#define IRIS_SDK_VERSION_MINOR 1
#define IRIS_SDK_VERSION_PATCH 0
#define IRIS_SDK_VERSION_STRING "0.1.0"

// Core headers (to be added)
// #include "iris_sdk/types.h"
// #include "iris_sdk/iris_detector.h"
// #include "iris_sdk/lens_renderer.h"
// #include "iris_sdk/frame_processor.h"
// #include "iris_sdk/sdk_manager.h"
// #include "iris_sdk/sdk_api.h"

namespace iris_sdk {

/**
 * @brief Get SDK version string
 * @return Version string (e.g., "0.1.0")
 */
const char* get_version();

} // namespace iris_sdk

#endif // IRIS_SDK_H
