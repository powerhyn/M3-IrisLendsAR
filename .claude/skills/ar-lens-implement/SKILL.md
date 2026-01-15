# AR Lens Workflow Skill

## Trigger Patterns

This skill activates when the user says:
- "implement P1-W3-01"
- "execute P1-W3-01"
- "run P1-W3-01"
- "P1-W3-01을 실행해줘"
- "P1-W3-01 구현해줘"

Or with mode:
- "implement P1-W3-01 performance mode"
- "implement P1-W3-01 --mode=performance"
- "P1-W3-01을 성능 중심으로 구현해줘"

## Instructions

When triggered, execute this automated workflow:

### Step 1: Parse Task Information

Extract task ID from user input:
- Pattern: P1-Wx-xx (e.g., P1-W3-01, P1-W4-02)
- Extract mode: standard (default), performance, quick, or debug

### Step 2: Locate Task Document

Search for task document:
```
docs/workPaper/{TASK_ID}_*.md
```

Example:
- P1-W3-01 → P1-W3-01_iris_detector_interface.md
- Extract module name: iris_detector_interface

### Step 3: Read and Analyze Document

Parse sections:
- `### 목표` - Goals
- `### 산출물` - Deliverables  
- `### 검증 기준` - Validation criteria
- `### 선행 조건` - Prerequisites

### Step 4: Execute Workflow by Mode

#### Standard Mode (Default)

```
🚀 AR Lens Workflow: {TASK_ID} (standard mode)

[1/6] 📄 Document Analysis
- Read docs/workPaper/{TASK_ID}_*.md
- Extract requirements
- Identify deliverables

[2/6] 💻 Implementation
- Implement per specification
- Follow C++17 standards
- RAII, thread-safety, const-correctness
- Create header + implementation files
- With @agent-systems-programming:cpp-pro agent

[3/6] 🧪 Test Generation
Execute: /unit-testing:test-generate "{MODULE_NAME}"
- Unit tests
- Edge cases
- Performance tests
- Memory leak tests

[4/6] 👁️ Code Review
Execute: /code-review-ai:ai-review
- Security issues
- Performance concerns
- Best practices
- Memory safety

[5/6] ⚡ Optimization
- Apply review suggestions
- Refactor if needed
- Re-verify compilation

[6/6] ✅ Validation
- Check validation criteria
- Verify deliverables
- Suggest documentation updates
- If everything is perfect, implement the git commit
```

#### Performance Mode

Add these steps after standard workflow:
```
[4/7] 📊 Performance Profiling
Execute: /application-performance project

Target metrics:
- FPS: > 30
- Latency: < 33ms
- Memory: < 100MB

[6/7] ⚡ Performance Optimization
- Eliminate bottlenecks
- Optimize memory allocations
- Use SIMD where applicable
- Reduce copies
```

#### Quick Mode

Simplified workflow:
```
[1/3] 📄 Document Analysis (brief)
[2/3] 💻 Implementation
[3/3] ✅ Compilation Check
```

#### Debug Mode

```
[1/5] 🐛 Problem Analysis
Execute: /debugging-toolkit:smart-debug

[2/5] 🔍 Root Cause
- Stack trace analysis
- Memory leak detection (AddressSanitizer)
- Thread safety (ThreadSanitizer)

[3/5] 💻 Fix Implementation

[4/5] 🧪 Regression Test
Execute: /unit-testing:test-generate "{ISSUE_NAME} regression test"

[5/5] ✅ Verification
```

### Step 5: Provide Completion Summary

```
🎉 {TASK_ID} Complete!

📊 Summary:
- Files created: {LIST}
- Tests generated: {COUNT}
- Review issues fixed: {COUNT}
- Estimated time: {MINUTES} min

📝 Documentation Updates:

1. Update docs/workPaper/{TASK_ID}_*.md:
   - Status: ⏳ → ✅
   - Add execution details to "실행 내역"
   - Update "검증 결과" table

2. Update docs/workPaper/000_phase1_plan.md:
   - Progress: {OLD}% → {NEW}%

💡 Next Steps:
1. Review generated code
2. Update documentation  
3. Commit: git commit -m "feat: {TASK_ID} - {MODULE_NAME}"
4. Next task: {NEXT_TASK_ID}

🔧 Next Command:
implement {NEXT_TASK_ID}
```

## Implementation Examples

### Example 1: IrisDetector Interface (P1-W3-01)

When user says: "implement P1-W3-01"

1. Find: docs/workPaper/P1-W3-01_iris_detector_interface.md
2. Module: iris_detector_interface
3. Implement:

```cpp
// include/iris_sdk/iris_detector.h
#pragma once

#include <expected>
#include <opencv2/core.hpp>
#include "iris_result.h"

namespace iris_sdk {

struct DetectorConfig {
    std::string model_path;
    float confidence_threshold = 0.7f;
    bool enable_tracking = true;
};

enum class DetectorError {
    None,
    InvalidConfig,
    InitializationFailed,
    ModelLoadFailed,
    DetectionFailed
};

class IrisDetector {
public:
    virtual ~IrisDetector() = default;
    
    virtual std::expected<void, DetectorError> 
        Initialize(const DetectorConfig& config) = 0;
    
    virtual std::expected<IrisResult, DetectorError>
        Detect(const cv::Mat& frame) = 0;
    
    virtual void Release() = 0;
    
    IrisDetector(const IrisDetector&) = delete;
    IrisDetector& operator=(const IrisDetector&) = delete;
    IrisDetector(IrisDetector&&) = default;
    IrisDetector& operator=(IrisDetector&&) = default;
    
protected:
    IrisDetector() = default;
};

} // namespace iris_sdk
```

4. Execute: `/unit-testing:test-generate "iris_detector_interface"`
5. Execute: `/code-review-ai:ai-review`
6. Apply optimizations
7. Report completion

### Example 2: MediaPipeDetector (P1-W3-03) - Performance Mode

When user says: "implement P1-W3-03 performance mode"

1. Find: docs/workPaper/P1-W3-03_mediapipe_detector.md
2. Module: mediapipe_detector
3. Implement MediaPipeDetector class
4. Execute: `/unit-testing:test-generate "mediapipe_detector with performance benchmarks"`
5. Execute: `/application-performance project`
6. Profile and optimize to meet targets (30fps, 33ms)
7. Execute: `/code-review-ai:ai-review`
8. Verify performance metrics
9. Report completion

## Error Handling

### Document Not Found

```
❌ Error: Task document not found

Task ID: {TASK_ID}
Expected: docs/workPaper/{TASK_ID}_*.md

💡 Verify:
- Task ID format (P1-Wx-xx)
- Document exists
- Correct file naming
```

### Invalid Mode

```
❌ Error: Invalid mode

Specified: {MODE}
Valid: standard, performance, quick, debug

Example:
implement P1-W3-01 performance mode
```

### Dependency Not Met

```
⚠️ Warning: Dependencies not satisfied

Task: {TASK_ID}
Missing: {DEPENDENCIES}

💡 Complete dependencies first:
implement {DEPENDENCY_TASK_ID}
```

## Project Context

This skill is designed for the IrisLensSDK project:

**Project Structure:**
```
IrisLensSDK/
├── docs/workPaper/
│   ├── 000_phase1_plan.md
│   ├── 001_phase1_workflow.md
│   └── P1-Wx-xx_*.md
├── cpp/
│   ├── include/iris_sdk/
│   ├── src/
│   └── tests/
└── .claude/
```

**Goals:**
- Real-time iris detection using MediaPipe
- AR lens overlay rendering with OpenGL
- Cross-platform C++ SDK (Desktop → Android → iOS)

**Performance Targets:**
- FPS: > 30
- Detection Latency: < 33ms
- Memory Usage: < 100MB
- SDK Size: < 20MB

**Tech Stack:**
- C++17
- MediaPipe (TensorFlow Lite)
- OpenCV 4.x
- OpenGL ES 3.0
- GoogleTest

## Integration with wshobson/agents

This skill automatically calls these plugins when available:

- `/unit-testing:test-generate` - Test generation
- `/code-review-ai:ai-review` - Code review
- `/application-performance` - Performance profiling
- `/debugging-toolkit:smart-debug` - Advanced debugging
- `/code-documentation:doc-generate` - Documentation

If plugins not installed, provide manual instructions.

## Notes

- Always read task document first
- Follow project coding standards (C++17, RAII, thread-safety)
- Performance is critical for real-time processing
- Maintain consistency with existing codebase
- Provide actionable next steps in completion summary

---

**Version:** 1.0.0
**Project:** IrisLensSDK Phase 1
**Updated:** 2025-01-08
