#!/bin/bash
# IrisLensSDK - Model Download Script
# Downloads MediaPipe TFLite models for iris tracking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/shared/models"

echo "======================================"
echo "IrisLensSDK Model Downloader"
echo "======================================"
echo ""
echo "Target directory: $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# MediaPipe model URLs (from official MediaPipe repository)
# These are the TFLite models used by MediaPipe for face mesh and iris tracking

# Face Detection (short range)
FACE_DETECTION_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
FACE_DETECTION_FILE="$MODELS_DIR/face_detection_short_range.tflite"

# Face Landmark (468 points)
FACE_LANDMARK_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_LANDMARK_FILE="$MODELS_DIR/face_landmark.task"

# Alternative: Classic MediaPipe models
# Note: Direct TFLite models may have different URLs
FACE_MESH_URL="https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite"
FACE_MESH_FILE="$MODELS_DIR/face_landmark.tflite"

IRIS_LANDMARK_URL="https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite"
IRIS_LANDMARK_FILE="$MODELS_DIR/iris_landmark.tflite"

download_model() {
    local url=$1
    local file=$2
    local name=$3

    if [[ -f "$file" ]]; then
        echo "✓ $name already exists, skipping..."
        return 0
    fi

    echo "Downloading $name..."
    if curl -L -o "$file" "$url" 2>/dev/null; then
        echo "✓ Downloaded $name"
    else
        echo "✗ Failed to download $name"
        echo "  URL: $url"
        return 1
    fi
}

# Download models
echo "Downloading MediaPipe models..."
echo ""

download_model "$FACE_DETECTION_URL" "$FACE_DETECTION_FILE" "Face Detection (short range)"
download_model "$FACE_MESH_URL" "$FACE_MESH_FILE" "Face Landmark (mesh)"
download_model "$IRIS_LANDMARK_URL" "$IRIS_LANDMARK_FILE" "Iris Landmark"

echo ""
echo "======================================"
echo "Download Complete!"
echo "======================================"
echo ""
echo "Models saved to: $MODELS_DIR"
echo ""
ls -la "$MODELS_DIR"/*.tflite 2>/dev/null || echo "No .tflite files found"
ls -la "$MODELS_DIR"/*.task 2>/dev/null || echo "No .task files found"
echo ""
