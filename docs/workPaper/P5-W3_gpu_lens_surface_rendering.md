# P5-W3: GPU Lens Surface Rendering (렌즈 표면 렌더링)

## 작업 개요
- **Phase**: P5 (경쟁사 대비 품질 갭 해소)
- **기간**: P5-W1 완료 후 (W2와 병렬 가능)
- **상태**: ⏳ 대기
- **선행 조건**: P5-W1-01 (스무딩 전략 결정). W3-00 (데모 기존 구현 분석) 선행 필수.
- **근거**: Codex — "Full PBR can wait. Lid occlusion plus corneal specular cannot."

## 핵심 전제: Android 데모에 이미 GPU 렌즈 렌더링 구현됨

**`CameraGLRenderer.kt`에 이미 존재하는 기능:**
- GLES 3.1 렌즈 오버레이 셰이더 (`:160`)
- 비대칭 타원 Eye Mask + `asymmetricEllipseMask()` (`:364-366`)
- 눈꺼풀 클리핑 (Y-slab + 타원 모드) (`:358-372`)
- 8종 블렌드 모드 (Normal~ColorReplace) (`:159-180`)
- Sclera Protection (기하학적 + 색상 기반) (`:377-384`)
- OneEuroFilter 기반 타원 파라미터 스무딩 (`:597-606`)
- Eyelid hold frames (`:47`, `EYELID_HOLD_FRAMES = 5`)
- 타원 캐시 (`:588-594`, `ellipseCacheValidFrames`)

**따라서 W3의 핵심은 "신규 구현"이 아니라 "데모 Kotlin/GLES → SDK C++ 코어 포팅 + 확장"이다.**
이중 구현을 방지하기 위해, 데모 코드를 레퍼런스로 사용하되 C++ 포팅 범위를 먼저 결정해야 한다.

## 목표

Android 데모의 검증된 GPU 렌즈 렌더링을 **SDK C++ 코어로 이관**하고, 추가로 각막 하이라이트/림발 다크닝/Normal Map을 확장한다.

### 현재 상태 정리

| 항목 | C++ SDK 코어 (`lens_renderer.cpp`) | Android 데모 (`CameraGLRenderer.kt`) |
|------|-----------------------------------|--------------------------------------|
| 메시 형태 | 평면 사각형 + cv::resize | 타원 마스크 (비대칭) |
| 눈꺼풀 오클루전 | ❌ | ✅ (Y-slab + 타원 모드) |
| 블렌드 모드 | 8종 (일부 CPU 폴백) | 8종 (전부 GPU 셰이더) |
| 반사광/하이라이트 | ❌ | ❌ |
| Sclera Protection | ❌ | ✅ (기하학적 + 색상) |
| 스무딩 | ❌ | ✅ (OneEuro 26개+) |

### 경쟁사 대비 비교

| 항목 | 피팅몬스터 | Perfect Corp | **Android 데모 (현재)** | **W3 목표 (SDK 코어)** |
|------|-----------|-------------|----------------------|---------------------|
| 메시 형태 | 3D Face Mesh | PBR 곡면 | 비대칭 타원 | 비대칭 타원 (포팅) |
| 눈꺼풀 오클루전 | ✅ | ✅ | ✅ | ✅ (포팅) |
| 각막 반사 | ❌ | ✅ (IBL) | ❌ | **✅ (신규)** |
| 림발 다크닝 | 미확인 | 추정 | ❌ | **✅ (신규)** |
| 블렌드 모드 GPU | MediaPipe | Venus 24+ | 8종 GPU | 8종 GPU (포팅) |
| Sclera Protection | 미확인 | 미확인 | ✅ | ✅ (포팅) |

---

## W3-00: 데모 CameraGLRenderer 기능 분석 + C++ 포팅 범위 결정

### 상태: ⏳ 대기 (W3 선행 필수)

### 작업 내용

1. **데모 기존 구현 기능 목록화**
   - `CameraGLRenderer.kt` 전체 분석 (~1500줄)
   - 셰이더 코드 추출 및 GLSL 버전 확인 (GLES 3.1)
   - `fitEyeEllipse()` 알고리즘 분석
   - `asymmetricEllipseMask()` 셰이더 로직 분석
   - Sclera Protection (`calcScleraFactor`) 분석

2. **C++ 포팅 범위 결정**
   - 포팅 대상: 타원 마스크, 눈꺼풀 클리핑, 8종 블렌드 셰이더, sclera protection
   - 제거 (SDK 코어로 대체): Kotlin OneEuroFilter 26개+ (W1의 TemporalStabilizer로 대체)
   - 비포팅 (Android 전용): 카메라 회전 보정, CameraX 연동
   - 신규 추가: 각막 하이라이트, 림발 다크닝, Normal Map

3. **LensRenderer API 재설계 (핵심 선결)**
   현재 `LensRenderer`는 cv::Mat 기반 OpenCV 렌더러이고 `IRenderContext` 주입 지점이 없다.
   GPU 렌즈 셰이더를 넣으려면:
   - `LensRenderer` 공개 API에 `IRenderContext` DI 추가 (BeautyProcessor 패턴 참조)
   - GL 컨텍스트 수명주기 관리 (`onSurfaceCreated`/`onSurfaceDestroyed`/`isContextLost`)
   - GPU 경로와 CPU 폴백 경로 분기

4. **플랫폼 추상화 방향**
   - `IRenderContext` 추상화 (`GLESRenderContext` / `CPURenderContext`) 활용
   - 셰이더 소스를 C++ 문자열로 이관
   - CPU 폴백: 기존 OpenCV 경로 유지 (일부 블렌드가 Normal로 폴백되는 현 동작 포함)

### 산출물

| 항목 | 내용 |
|------|------|
| 기능 대조표 | 데모 기능 ↔ SDK 코어 포팅 여부 |
| 셰이더 이관 목록 | GLSL 코드 추출 + C++ 래핑 방식 |
| 포팅 난이도 추정 | 항목별 작업량 |

---

## W3-01: 데모 타원 마스크/눈꺼풀 클리핑 → C++ 포팅

### 상태: ⏳ 대기

### 배경

Android 데모 `CameraGLRenderer.kt`에 이미 검증된 구현이 있다:
- `fitEyeEllipse()` (`:1091`): Face Mesh 16점 → 비대칭 타원 피팅
- `asymmetricEllipseMask()` (`:364-366`): 내안각/외안각 반경 분리
- Y-slab + 타원 듀얼 모드 눈꺼풀 클리핑 (`:358-372`)

### 작업 내용

1. **타원 피팅 알고리즘 C++ 포팅**
   - `fitEyeEllipse()` Kotlin → C++ 변환
   - Face Mesh 랜드마크 인덱스 동일하게 사용
   - `IRenderContext` 기반 GPU 렌더링 경로에 통합

2. **눈꺼풀 클리핑 셰이더 포팅**
   - GLES 3.1 프래그먼트 셰이더 → C++ 셰이더 문자열로 이관
   - `ShaderManager`에 렌즈 전용 프로그램 등록
   - 기존 `lens_renderer.cpp`의 CPU 경로는 폴백으로 유지

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/include/iris_sdk/lens_renderer.h` | 공개 API에 `IRenderContext` DI 추가, GPU/CPU 분기 |
| `cpp/src/lens_renderer.cpp` | GPU 렌더링 경로 추가, GL 컨텍스트 수명주기, CPU 폴백 유지 |
| `cpp/include/iris_sdk/gpu/lens_shader.h` (신규) | 렌즈 셰이더 프로그램 (타원 마스크 + 눈꺼풀 클리핑) |
| `cpp/src/gpu/lens_shader.cpp` (신규) | 셰이더 소스 + 컴파일 (데모 GLSL 포팅) |

---

## W3-02: 데모 8종 블렌드 + Sclera Protection → C++ GPU 셰이더 이관

### 상태: ⏳ 대기

### 배경

데모 `CameraGLRenderer.kt:159-180`에 8종 블렌드 모드가 GPU 셰이더로 구현되어 있다. C++ `lens_renderer.cpp:505`에서는 일부가 CPU Normal로 폴백된다. Sclera Protection(`:377-384`)도 데모에만 존재.

### 작업 내용

1. **데모 블렌드 셰이더 → C++ GLSL 문자열 이관**
   - Normal, Multiply, Screen, Overlay, LuminanceTint, LuminanceTintLinear, SoftLight, ColorReplace
   - `ShaderManager`에 uniform으로 blend_mode 전달

2. **Sclera Protection 포팅**
   - 기하학적 감쇄 (홍채 경계 0.75~1.0 smoothstep)
   - 색상 기반 감쇄 (`calcScleraFactor`)
   - 흰자위 번짐 방지

3. **CPU 폴백 경로 현상 유지**
   - GPU 불가 환경에서만 CPU 경로 사용
   - CPU 경로는 현재 상태 유지: Normal/Multiply/Screen/Overlay만 동작, LuminanceTint/LuminanceTintLinear/SoftLight/ColorReplace는 Normal로 폴백 (`lens_renderer.cpp:518`)
   - CPU 블렌드 8종 parity는 P5 범위 밖 (필요 시 별도 작업)

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/src/gpu/lens_shader.cpp` | 블렌드 + sclera 셰이더 |
| `cpp/src/lens_renderer.cpp` | CPU 폴백 정리 |

---

## W3-03: 각막 하이라이트 + 림발 다크닝

### 상태: ⏳ 대기

### 배경

실제 눈은 각막 반사(corneal specular)와 림발 다크닝(limbal darkening, 홍채-공막 경계의 어두운 고리)이 있다. 이 두 요소가 없으면 렌즈가 "매트하고 평면적"으로 보인다.

### 작업 내용

1. **각막 하이라이트 (Corneal Specular)**
   ```glsl
   // 간단한 방식: 고정 위치에 작은 원형 하이라이트
   vec2 highlight_offset = vec2(-0.15, -0.2); // 좌상단 (조명 방향)
   float highlight = smoothstep(0.08, 0.0, 
       distance(uv, center + highlight_offset));
   color = mix(color, vec3(1.0), highlight * 0.6);
   ```
   - 향후 개선: 카메라 밝기 분석하여 하이라이트 위치/강도 동적 조절

2. **림발 다크닝 (Limbal Darkening)**
   ```glsl
   // 홍채 경계(r=1.0)에서 안쪽으로 어두운 고리
   float limbal = smoothstep(0.7, 1.0, r); // r = 중심에서의 정규화 거리
   color = mix(color, color * 0.4, limbal * 0.8);
   ```
   - 홍채-공막 경계를 자연스럽게 정의
   - 렌즈의 "깊이감" 생성

3. **홍채 내부 그라데이션 (선택적)**
   ```glsl
   // 동공 주변은 살짝 밝게, 외곽은 어둡게
   float inner_glow = 1.0 - smoothstep(0.0, 0.4, r);
   color = mix(color, color * 1.2, inner_glow * 0.3);
   ```

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| 셰이더 파일 (신규 또는 기존 확장) | 하이라이트 + 림발 셰이더 |
| `cpp/include/iris_sdk/types.h` | LensConfig에 highlight/limbal 파라미터 추가 |

---

## W3-04: 선택적 Normal Map + IBL

### 상태: ⏳ 대기 (선택적)

### 배경

Perfect Corp은 `eye_normal.png` (512×512) + `eye_ibl.hdr` (1.8MB)로 PBR 렌더링을 한다. 풀 PBR은 과도하지만, 간단한 Normal Map은 곡률감을 크게 향상시킨다.

### 작업 내용

1. **홍채 Normal Map 생성**
   ```
   // 구면(sphere)을 기반으로 분석적 노말 생성
   // 별도 텍스처 에셋 없이 셰이더에서 계산 가능
   vec3 normal;
   normal.x = (uv.x - 0.5) * 2.0;
   normal.y = (uv.y - 0.5) * 2.0;
   normal.z = sqrt(1.0 - clamp(dot(normal.xy, normal.xy), 0.0, 1.0));
   ```

2. **간단한 Diffuse + Specular 라이팅**
   ```glsl
   vec3 light_dir = normalize(vec3(0.3, 0.4, 1.0)); // 고정 조명
   float diffuse = max(dot(normal, light_dir), 0.0);
   float specular = pow(max(dot(reflect(-light_dir, normal), view_dir), 0.0), 32.0);
   
   color = color * (0.6 + 0.4 * diffuse) + vec3(1.0) * specular * 0.3;
   ```

3. **(향후) 텍스처 기반 Normal Map + HDR IBL**
   - `eye_normal.png` 에셋 추가 (~100KB)
   - Low-res HDR 환경맵 (~200KB)
   - Perfect Corp 수준에 근접

### 결정 필요 사항

- [ ] 분석적 노말 (셰이더 계산, 에셋 0) vs 텍스처 노말 (에셋 추가)
- [ ] IBL 도입 시점 (W3에서 or Phase 6)

---

---

## 작업 순서 및 의존성

```
W3-00 (데모 분석) ← 최선행 필수
    ↓
W3-01 (타원/눈꺼풀 포팅) ← W3-00 후
W3-02 (블렌드/Sclera 포팅) ← W3-00 후 (W3-01과 병렬 가능)
    ↓
W3-03 (하이라이트/림발) ← W3-01 후 (신규)
W3-04 (Normal Map) ← W3-03 후 (선택적, 신규)
```

## 예상 효과

- **렌즈 사실감**: "2D 스티커" → "실제 컨택트 렌즈" 전환
- **눈꺼풀 오클루전**: 가장 큰 시각적 개선 (Codex: "cannot wait")
- **각막 하이라이트**: 렌즈에 "생기" 부여
- **림발 다크닝**: 렌즈-공막 경계 자연스러움
- **경쟁사 대비**: 피팅몬스터와 동등, Perfect Corp의 70~80% 수준 달성
- **이중 구현 방지**: 데모 검증 코드를 SDK 코어 C++로 정립. Android 데모는 W1 완료 후 Kotlin 스무딩 제거, W3 완료 후 Kotlin 셰이더도 SDK 코어 호출로 점진 전환
