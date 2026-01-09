# P1-W4-01: LensRenderer 기본 구현

**태스크 ID**: P1-W4-01
**상태**: ✅ 완료
**시작일**: 2026-01-09
**완료일**: 2026-01-09

---

## 1. 계획

### 목표
검출된 홍채 위치에 가상 렌즈 텍스처를 오버레이하는 렌더러를 구현한다.

### 산출물
| 파일 | 설명 | 상태 |
|------|------|------|
| `cpp/include/iris_sdk/lens_renderer.h` | LensRenderer 클래스 선언 | ✅ 완료 |
| `cpp/src/lens_renderer.cpp` | LensRenderer 구현 | ✅ 완료 |
| `shared/textures/sample_lens.png` | 샘플 렌즈 텍스처 | ⏳ 별도 태스크 |

### 검증 기준
- [x] 텍스처 로딩 성공 (PNG, JPEG, Grayscale 지원)
- [x] 정적 이미지에 렌즈 오버레이 성공 (API 구현 완료)
- [x] 양눈 독립 렌더링 지원 (renderLeftEye, renderRightEye)
- [ ] 렌더링 시간 10ms 이하 (통합 테스트 필요)
- [ ] 시각적 품질 확인 (통합 테스트 필요)

### 선행 조건
- P1-W3-02: 데이터 구조 정의 ✅
- P1-W3-03: MediaPipeDetector 구현 ✅

---

## 2. 분석

### 2.1 렌더링 파이프라인

```
IrisResult (검출 결과)
    │
    ▼
┌─────────────────┐
│ 좌표 변환       │ 정규화 좌표 → 픽셀 좌표
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 영역 계산       │ 중심, 반지름, 회전 각도
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 텍스처 변환     │ 리사이즈 + 회전 (warpAffine)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 마스크 생성     │ 원형 페더링 마스크
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 블렌딩          │ 알파 블렌딩 (Normal/Multiply/Screen/Overlay)
└─────────────────┘
    │
    ▼
Output Frame (렌즈 적용된 이미지)
```

### 2.2 구현된 클래스 설계

```cpp
class IRIS_SDK_EXPORT LensRenderer {
public:
    LensRenderer();
    ~LensRenderer();
    LensRenderer(LensRenderer&&) noexcept;
    LensRenderer& operator=(LensRenderer&&) noexcept;

    // 초기화/해제
    bool initialize();
    void release();
    bool isInitialized() const;

    // 텍스처 관리
    bool loadTexture(const std::string& texture_path);
    bool loadTexture(const uint8_t* data, int width, int height);
    void unloadTexture();
    bool hasTexture() const;

    // 렌더링
    bool render(cv::Mat& frame, const IrisResult& iris_result, const LensConfig& config);
    bool renderLeftEye(cv::Mat& frame, const IrisResult& iris_result, const LensConfig& config);
    bool renderRightEye(cv::Mat& frame, const IrisResult& iris_result, const LensConfig& config);

    // 통계
    double getLastRenderTimeMs() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

### 2.3 핵심 구현 세부사항

**2.3.1 홍채 영역 구조체**
```cpp
struct IrisRegion {
    cv::Point2f center;     // 홍채 중심 (픽셀 좌표)
    float radius;           // 홍채 반지름 (픽셀)
    float angle;            // 회전 각도 (도)
    bool valid;             // 유효성 플래그
};
```

**2.3.2 좌표 변환**
- IrisLandmark의 정규화 좌표(0~1)를 픽셀 좌표로 변환
- 프레임 크기에 따라 동적 스케일링

**2.3.3 반지름 계산**
- landmark[0]: 중심점
- landmark[1~4]: 상/하/좌/우 경계점
- 4개 경계점의 평균 거리로 반지름 계산
- 최소 5px ~ 최대 200px 범위 검증

**2.3.4 회전 각도 계산**
- landmark[3] (좌측)과 landmark[4] (우측) 사용
- atan2로 수평 기준 회전 각도 계산

**2.3.5 텍스처 변환**
- cv::resize로 홍채 크기에 맞게 리사이즈
- cv::warpAffine으로 회전 적용
- BORDER_CONSTANT로 회전 시 빈 영역 처리

**2.3.6 페더링 마스크**
- 코사인 보간으로 부드러운 가장자리 생성
- 내부 원: 완전 불투명 (alpha=1.0)
- 외부 가장자리: 점진적 페이드아웃

**2.3.7 블렌드 모드**
| 모드 | 수식 | 용도 |
|------|------|------|
| Normal | dst = src*a + dst*(1-a) | 기본 오버레이 |
| Multiply | dst = src*dst/255 | 어둡게 |
| Screen | dst = 255 - (255-src)*(255-dst)/255 | 밝게 |
| Overlay | if d<0.5: 2*s*d else: 1-2*(1-s)*(1-d) | 대비 강화 |

### 2.4 텍스처 요구사항

| 항목 | 요구사항 |
|------|----------|
| 포맷 | PNG (알파 채널 포함 권장), JPEG, Grayscale |
| 크기 | 256x256 ~ 512x512 권장 |
| 형태 | 원형, 중심 정렬 |
| 알파 | 경계 그라데이션 권장 |

### 2.5 성능 최적화

| 전략 | 구현 상태 |
|------|----------|
| ROI 기반 처리 | ✅ 홍채 영역만 렌더링 |
| 경계 클리핑 | ✅ 프레임 경계 자동 처리 |
| 조건부 컴파일 | ✅ IRIS_SDK_HAS_OPENCV |
| 렌더링 시간 측정 | ✅ getLastRenderTimeMs() |

---

## 3. 실행 내역

### 3.1 헤더 파일 작성 ✅

```bash
# 생성됨: cpp/include/iris_sdk/lens_renderer.h
# - Pimpl 패턴 적용
# - cv::Mat 전방 선언
# - IRIS_SDK_EXPORT 매크로 사용
# - Doxygen 주석 완료
```

### 3.2 구현 파일 작성 ✅

```bash
# 생성됨: cpp/src/lens_renderer.cpp
# - 조건부 컴파일 (IRIS_SDK_HAS_OPENCV)
# - 4가지 블렌드 모드 구현
# - 페더링 마스크 생성
# - ROI 기반 블렌딩
```

### 3.3 CMake 업데이트 ✅

```bash
# 수정됨: cpp/CMakeLists.txt
# - IRIS_SDK_SOURCES에 lens_renderer.cpp 추가
# - OpenCV imgcodecs 컴포넌트 추가
```

### 3.4 빌드 검증 ✅

```bash
cd cpp/cmake-build-debug
cmake --build . --target iris_sdk
# 결과: 성공적으로 컴파일됨
```

### 3.5 단위 테스트 작성 ✅

```bash
# 생성됨: cpp/tests/test_lens_renderer.cpp
# - 총 57개 테스트 케이스
# - 초기화, 텍스처, 렌더링, 경계 조건, 성능 테스트
# - 모든 테스트 통과
```

### 3.6 코드 리뷰 및 최적화 ✅

**리뷰 결과:**
- 1 Critical, 11 Warning, 11 Info 이슈 발견
- 주요 수정 사항:
  - 텍스처 크기 최대값 검증 (4096px)
  - frame 유효성 검사 추가
  - frame_width/height 검증
  - impl_ nullptr 체크 (이동 후 안전성)
  - noexcept 지정자 추가

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 컴파일 | ✅ 성공 | CMake 빌드 성공 |
| 텍스처 로딩 API | ✅ 구현 | PNG, JPEG, Grayscale, RGBA |
| 좌표 변환 | ✅ 구현 | normalizedToPixel() |
| 홍채 영역 계산 | ✅ 구현 | calculateIrisRegion() |
| 텍스처 변환 | ✅ 구현 | transformTexture() |
| 페더링 마스크 | ✅ 구현 | createFeatherMask() |
| 알파 블렌딩 | ✅ 구현 | 4가지 블렌드 모드 |
| 양눈 렌더링 | ✅ 구현 | renderLeftEye(), renderRightEye() |
| 단위 테스트 | ✅ 57개 통과 | 초기화, 텍스처, 렌더링, 경계 조건, 성능 |
| 코드 리뷰 | ✅ 완료 | Critical/Warning 이슈 수정 완료 |
| 렌더링 시간 | ⏳ 대기 | 통합 테스트 필요 (목표: 10ms) |
| 시각적 품질 | ⏳ 대기 | 통합 테스트 필요 |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| in-place 렌더링 | 메모리 복사 최소화, cv::Mat& 참조 사용 |
| Pimpl 패턴 | OpenCV 헤더 의존성 분리, ABI 안정성 |
| 원형 페더링 마스크 | 홍채 형태에 가장 적합, 자연스러운 경계 |
| 코사인 보간 | 선형 보간보다 부드러운 페이드아웃 |
| 조건부 컴파일 | OpenCV 없이도 컴파일 가능하도록 |
| 이동 생성자 지원 | std::vector 등에서 효율적 사용 |

### 학습 내용

1. **cv::warpAffine 경계 처리**: BORDER_CONSTANT로 회전 시 빈 영역을 투명하게 처리
2. **알파 블렌딩 최적화**: ROI 기반 처리로 전체 프레임 순회 방지
3. **경계 클리핑**: 홍채가 프레임 경계에 걸릴 때 안전하게 처리
4. **블렌드 모드 수식**: Photoshop 스타일 블렌드 모드 구현

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 렌더링 파이프라인 설계 완료 |
| 2026-01-09 | LensRenderer 구현 완료, 빌드 검증 성공 |
