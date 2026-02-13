# P3 GPU 렌더링 파이프라인 및 LUT 필터 구현 리뷰

- **작성일**: 2026-02-11
- **관련 커밋**: `feature/P3-beauty-enhancement` 브랜치 작업 내역
- **검토 대상**:
    - `CameraGLRenderer.kt`: Android OpenGL 렌더링 파이프라인
    - `gpu_beauty_backend.cpp`: Native Beauty Filter 구현
    - `GpuRenderActivity.kt`: UI 및 제어 로직

## 1. 개요 (Overview)

본 문서는 Phase 3의 핵심 목표인 **"GPU 기반 렌즈 오버레이와 뷰티 필터의 통합"** 구현 상태를 분석하고, 새롭게 도입된 **LUT(Look-Up Table) 컬러 그레이딩** 시스템의 기술적 가치와 향후 최적화 방향을 제시한다.

현재 시스템은 MediaPipe의 CPU 추론(얼굴/홍채 감지)과 OpenGL ES 기반의 GPU 렌더링이 비동기로 작동하는 **하이브리드 아키텍처**를 성공적으로 구축하였으며, 특히 메모리 관리와 렌더링 안정성 측면에서 높은 완성도를 보인다.

## 2. 렌더링 파이프라인 분석 (Pipeline Analysis)

현재 구현된 프레임 처리 흐름은 다음과 같다.

```mermaid
graph TD
    CAM[Camera Input (OES)] -->|Step 1: Convert| RGBA[RGBA Texture\n(Mirror/Rotate)]
    
    subgraph "Lens Processing (Kotlin/GLSL)"
    RGBA -->|Texture + Iris Coords| LENS_SHADER[Lens Overlay Shader]
    LENS_SHADER -->|Masking/Blending| LENS_FBO[Lens Output FBO]
    end
    
    subgraph "Beauty Processing (C++ Native)"
    LENS_FBO -->|JNI Call| SMOOTH[Smoothing Pass\n(Bilateral)]
    SMOOTH -->|Ping-Pong| COLOR[Combined Pass\n(Bright/White/Balance)]
    COLOR -->|Ping-Pong| SOFT[Soft Focus Pass]
    SOFT --> BEAUTY_TEX[Beauty Result Texture]
    end

    subgraph "Post Processing"
    BEAUTY_TEX -->|Step 4: LUT| LUT_SHADER[LUT Color Grade]
    LUT_SHADER -->|Step 5: Draw| SCREEN[Device Screen]
    end
```

### 2.1 주요 단계별 특징
1.  **OES to RGBA**: OES 텍스처를 표준 2D 텍스처로 변환하며, 전면 카메라 미러링/회전/상하반전(Flip-Y)을 일괄 처리하여 후속 단계의 복잡도를 낮췄다.
2.  **Lens Overlay**: `IrisResult` 좌표를 기반으로 렌즈를 합성한다.
    - **Eyelid Clipping**: MediaPipe 랜드마크를 활용해 렌즈가 눈꺼풀 위로 튀어나오지 않도록 마스킹 처리됨.
    - **Feathering**: 렌즈 가장자리 부드러움 처리.
3.  **Beauty Filter (Native)**: C++ 백엔드에서 수행되며, **Ping-Pong 버퍼**를 활용해 텍스처 생성 비용을 최소화했다.
4.  **LUT Filter**: 3D Texture Lookup을 통해 복잡한 색상 보정을 단일 패스로 처리한다.

## 3. 핵심 기술 심층 분석

### 3.1 LUT (Look-Up Table) 컬러 그레이딩
현재 구현된 LUT 시스템은 상용 뷰티 앱(Instagram, VSCO 등)과 동일한 방식의 **산업 표준 기술**이다.

*   **동작 원리**: 입력 색상(RGB)을 3D 좌표(XYZ)로 사용하여 `uLutTexture`에서 결과 색상을 샘플링한다.
*   **구현 방식**: `LutTextureLoader`를 통해 비트맵을 3D 텍스처로 변환하여 업로드한다.
*   **장점**:
    - **무한한 확장성**: 셰이더 코드 수정 없이 `.png` LUT 이미지 교체만으로 필터 스타일을 무한히 확장 가능하다.
    - **고성능**: 복잡한 색상 연산(Curve, HSL, Tint 등)을 단 한 번의 텍스처 조회(Texture Lookup)로 대체한다.
*   **Identity LUT**: 현재 기본값으로 사용되는 Identity LUT는 입력과 출력이 동일한(변화 없는) 상태를 의미하며, 이를 기준으로 디자이너가 제작한 LUT를 적용하면 즉시 필터 효과를 낼 수 있다.

### 3.2 Smoothing (Bilateral Filter)
피부 보정의 핵심인 스무딩 효과는 **Bilateral Filter** 알고리즘을 사용한다.
*   **공간 가중치(Space Weight)** + **색상 차이 가중치(Range Weight)**를 결합하여 작동한다.
*   **효과**: 피부와 같이 색상 차이가 적은 영역은 뭉개고(Blur), 눈/코/입 등 색상 차이가 큰 경계선(Edge)은 보존하여 "선명하면서도 매끈한" 피부 표현이 가능하다.

## 4. 구현 평가 (Review)

### ✅ 우수한 점 (Pros)
1.  **메모리 안정성**: C++ 레벨에서 `TexturePool`을 도입하여 매 프레임 텍스처 할당/해제를 방지했다. 이는 GC 오버헤드와 메모리 파편화를 막는 핵심 설계다.
2.  **Double-free 방지**: Android와 Native 간의 텍스처 소유권을 명확히 하여, 고질적인 문제였던 검은 화면 이슈를 근본적으로 해결했다.
3.  **셰이더 최적화**: 기존에 분리되어 있던 Brightness, Color Balance, Whitening을 `CombinedColorPass`로 통합하여 Draw Call을 줄였다.

### ⚠️ 개선 가능성 (Cons)
1.  **Draw Call 중복**: 현재 **기본 보정(Brightness/Whitening)** 패스와 **LUT 필터** 패스가 별도로 수행되어 텍스처 쓰기(Draw Call)가 1회 더 발생한다.
2.  **LUT 에셋 부재**: 기능은 구현되었으나 실제 사용할 수 있는 예쁜 필터(LUT PNG) 파일이 없어 시각적 확인이 어렵다.

## 5. 최적화 제안 (Optimization Proposal)

### 5.1 Super Color Shader 도입 (강력 권장)
현재 분리된 **슬라이더 보정(Brightness/Whitening)**과 **LUT 필터**를 하나의 셰이더로 통합할 것을 제안한다.

*   **현재**: `Combined Pass` (Draw) → `LUT Pass` (Draw) = **2 Pass**
*   **제안**: `Combined Pass` 내부에서 LUT 연산까지 수행 = **1 Pass**

**통합 셰이더 로직 예시:**
```glsl
void main() {
    vec4 color = texture(uTexture, vTexCoord);
    
    // 1. 슬라이더 보정 (동적 튜닝)
    vec3 corrected = applyBrightness(color.rgb, uBrightness);
    corrected = applyWhitening(corrected, uWhitening);
    
    // 2. LUT 적용 (스타일링)
    // 보정된 색상(corrected)을 기준으로 LUT를 조회
    vec3 lutColor = texture(uLutTexture, corrected).rgb;
    
    // 3. 최종 믹스
    fragColor = vec4(mix(corrected, lutColor, uLutIntensity), color.a);
}
```
이 방식은 **기능(슬라이더 조절 + 필터 적용)을 모두 유지하면서 성능을 2배 가까이 향상**시킬 수 있다.

### 5.2 추천 LUT 에셋 추가
데모 앱의 완성도를 위해 다음 3가지 스타일의 LUT 파일을 `assets/luts/`에 추가하는 것을 권장한다.
1.  **Juno Style**: 인물 피부톤을 생기 있게 (Daily용)
2.  **Film Style (Kodak Portra)**: 감성적인 필름 룩
3.  **High Contrast B&W**: 느와르 스타일 흑백

---
**종합 의견**: 현재 구현 상태는 상용 수준의 품질을 갖추고 있으며, 제안된 셰이더 통합 최적화만 적용된다면 성능 면에서도 매우 우수할 것으로 판단됨.
