# P4-W3-04 성능 및 확장성 분석 보고서

| 항목 | 내용 |
|------|------|
| **대상 커밋** | 9fd5bdc (feature/P4-W3-04) |
| **분석 범위** | Temporal Stability (One Euro Filter) + DeviceTier 분기 + Half-Res FreqSep 파이프라인 |
| **프레임 버짓** | 33ms (30fps 목표) |
| **분석일** | 2026-03-04 |

---

## 1. 요약

P4-W3-04 변경은 전반적으로 잘 설계되었으며, 실시간 33ms 프레임 버짓을 충족할 수 있다.
MID tier 하이브리드 해상도 파이프라인은 대역폭 절감 효과가 크며, LOW tier Bilateral fallback 분기는 안전한 성능 보장을 제공한다.
아래에 발견된 이슈를 심각도별로 정리한다.

**심각도 요약**:
- Critical: 0건
- High: 1건
- Medium: 3건
- Low: 4건

---

## 2. GPU 파이프라인 성능

### 2.1 [Medium] 중복 glTexParameteri 호출

**위치**: `gpu_beauty_backend.cpp:1310-1318`

```cpp
// Pass 3: Composite (full-res)
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, smoothedLow_half->texture_id);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // 중복
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // 중복

glActiveTexture(GL_TEXTURE1);
glBindTexture(GL_TEXTURE_2D, lowFreq_half->texture_id);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // 중복
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // 중복
```

**분석**: `TexturePool`은 텍스처 생성 시 이미 `GL_LINEAR`를 설정한다 (`texture_pool.cpp:331-332`).
풀에서 획득한 텍스처는 해제 후 재사용되더라도 필터 모드가 변경되지 않으므로,
이 4개의 `glTexParameteri` 호출은 **동일한 값을 재설정하는 중복 호출**이다.

**성능 영향**: 프레임당 4회의 불필요한 GL 상태 변경 명령. 드라이버 레벨에서 no-op으로 최적화될 가능성이 높으나, 일부 모바일 GPU 드라이버(특히 Mali)는 glTexParameteri 호출 시 내부 해시 검증을 수행하여 미세한 오버헤드가 발생할 수 있다.

**추정 영향**: 프레임당 0.01~0.05ms (대부분 무시 가능)

**권장 조치**: Phase 1 컨텍스트에서 지적된 "GL_LINEAR 텍스처 필터 미복원" 이슈는 **실제로는 문제가 아니다**. TexturePool의 기본 필터가 GL_LINEAR이므로 복원할 필요가 없다. 다만 의도를 명확히 하기 위해 두 가지 선택지가 있다:

1. (권장) 중복 호출 제거 + 주석으로 TexturePool 기본값 의존 명시
2. (대안) 방어적으로 유지하되, TexturePool에 `getDefaultFilter()` 상수 추가하여 일관성 보장

### 2.2 [Low] 중복 glUseProgram 호출

**위치**: `gpu_beauty_backend.cpp:1282`

```cpp
// Pass 2a 시작부
glUseProgram(freq_sep_gaussian_program_);  // Pass 1에서 이미 바인딩됨
```

**분석**: Pass 1a(line 1254)에서 `freq_sep_gaussian_program_`이 이미 활성화되어 있고,
Pass 1b까지 동일 프로그램을 사용한다. Pass 2a에서 다시 `glUseProgram`을 호출하는 것은 중복이다.
동일 프로그램 ID로의 `glUseProgram` 호출은 대부분의 드라이버에서 no-op으로 처리된다.

**추정 영향**: 프레임당 <0.01ms

**권장 조치**: 제거해도 되나, 코드 가독성(각 패스의 독립성 명시)을 위해 유지해도 무방하다.
full-res `executeFreqSepPipeline`(line 1079)에서도 동일 패턴을 사용하므로 일관성 측면에서 유지가 합리적이다.

### 2.3 [Low] Shader Switching 비용

**분석**: Half-res 파이프라인은 2개의 셰이더 프로그램을 사용한다:
- `freq_sep_gaussian_program_` (Pass 1a, 1b, 2a, 2b)
- `freq_sep_composite_program_` (Pass 3)

셰이더 전환은 Pass 2b → Pass 3 사이에 **1회만** 발생한다.
이는 full-res 파이프라인과 동일한 전환 횟수이며, 최적의 배치 패턴이다.

**추정 영향**: 전환 1회 = 약 0.05~0.1ms (모바일 GPU 기준)

**권장 조치**: 현재 상태로 최적. 추가 조치 불필요.

---

## 3. 메모리 관리

### 3.1 [Medium] Half-Res 텍스처 풀 압력

**분석**: MID tier는 프레임당 3개의 half-res 텍스처를 acquireRenderTarget으로 할당한다.

**1080p 기준 메모리 비교**:

| 항목 | 해상도 | RGBA 크기 | 수량 | 총 메모리 |
|------|--------|-----------|------|-----------|
| HIGH tier 중간 버퍼 | 1920x1080 | 8.29MB | 3장 | 24.88MB |
| MID tier 중간 버퍼 | 960x540 | 2.07MB | 3장 | 6.22MB |
| **MID tier 절감량** | - | - | - | **-18.66MB (-75%)** |

**풀 압력 분석**: TexturePool은 해상도별로 텍스처를 관리하므로, half-res와 full-res 텍스처는
별도의 슬롯을 차지한다. MID tier 파이프라인이 half-res 텍스처를 반복적으로 acquire/release하면,
풀에 half-res 텍스처가 캐싱되어 재할당 오버헤드가 없다.

**잠재적 이슈**: 다른 파이프라인 단계(Combined Color, Masking 등)가 full-res 텍스처를 사용하면서
동시에 half-res 텍스처가 풀에 상주하면, 총 풀 크기가 증가할 수 있다.
다만 half-res 3장의 추가 메모리(6.22MB)는 절대적으로 작은 규모이다.

**추정 영향**: 추가 6.22MB GPU 메모리 (허용 범위)

**권장 조치**: TexturePool의 `max_textures` 설정이 half-res 텍스처를 수용할 수 있는지 확인.
필요 시 `PoolStats` 로깅을 추가하여 운영 중 풀 사용률을 모니터링.

### 3.2 [Low] One Euro Filter 메모리 영향

**분석**: 3개의 `OneEuroFilter` 인스턴스가 추가되었다.

```cpp
OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};
OneEuroFilter mask_center_x_filter_{1.0f, 0.02f, 1.0f};
OneEuroFilter mask_center_y_filter_{1.0f, 0.02f, 1.0f};
```

각 `OneEuroFilter`는 내부에 2개의 `LowPassFilter` + 파라미터 float 변수를 포함한다.
추정 인스턴스 크기: ~48 bytes x 3 = **~144 bytes**.

**추정 영향**: 무시 가능 (144 bytes)

**권장 조치**: 조치 불필요.

---

## 4. CPU-Side 오버헤드

### 4.1 [Low] detectDeviceTier() 문자열 파싱

**위치**: `gpu_beauty_backend.cpp:1155-1211`, 호출 위치: `initialize()` line 162

**분석**: `detectDeviceTier()`는 **`initialize()` 내에서 1회만 호출**된다.
결과는 `device_tier_` 멤버 변수에 캐싱되어 이후 프레임에서 재사용된다.

```cpp
// initialize() 내부 (line 162)
device_tier_ = detectDeviceTier();
```

문자열 파싱 비용:
- `glGetString(GL_RENDERER)`: GL 드라이버 호출 1회
- `std::string` 생성 + `find()` 최대 5회 + digit 추출
- 총 비용: ~0.01ms (1회성)

**추정 영향**: 초기화 시 0.01ms 미만 (프레임 성능 무관)

**권장 조치**: 조치 불필요. Phase 1 컨텍스트의 우려는 해소됨.

### 4.2 [Low -> 무시] One Euro Filter 연산 비용

**위치**: `gpu_beauty_backend.cpp:1614-1633`

**분석**: 프레임당 3회의 `filter()` 호출. 각 호출은 ~10개의 부동소수점 연산
(exp, sqrt 없이 단순 곱셈/덧셈)을 수행한다.

**추정 영향**: 프레임당 ~0.001ms (CPU). 33ms 버짓의 0.003%.

**권장 조치**: 조치 불필요.

---

## 5. 프레임 버짓 분석

### 5.1 MID Tier Half-Res 파이프라인 예상 타이밍

1080p (1920x1080) 기준, Adreno 6xx급 GPU 추정:

| 패스 | 해상도 | 추정 시간 | 비고 |
|------|--------|-----------|------|
| Pass 1a: Gaussian H | 960x540 | ~1.5ms | full-res 대비 ~4x 빠름 |
| Pass 1b: Gaussian V | 960x540 | ~1.5ms | |
| Pass 2a: Low Smooth H | 960x540 | ~1.2ms | low_radius < blur_radius |
| Pass 2b: Low Smooth V | 960x540 | ~1.2ms | |
| Pass 3: Composite | 1920x1080 | ~3.5ms | 4 텍스처 입력 |
| **FreqSep 합계** | | **~8.9ms** | |

| 후속 패스 | 추정 시간 |
|-----------|-----------|
| Combined Color | ~1.5ms |
| Masking | ~1.5ms |
| Soft Focus (선택) | ~2.0ms |
| ROI/기타 오버헤드 | ~1.0ms |
| **전체 파이프라인** | **~14.9ms** |

**결론**: 33ms 버짓 대비 약 45% 사용. **충분한 여유**가 있다.
작업 문서의 "MID ≤12ms" 목표는 FreqSep 단독으로는 달성 가능할 것으로 보인다.

### 5.2 HIGH vs MID 비교

| 항목 | HIGH (Full-Res) | MID (Hybrid Half-Res) | 절감률 |
|------|-----------------|----------------------|--------|
| Gaussian 4패스 픽셀 수 | 4 x 2.07M = 8.29M | 4 x 0.52M = 2.07M | **-75%** |
| Composite 픽셀 수 | 2.07M | 2.07M (동일) | 0% |
| 중간 버퍼 메모리 | 24.88MB | 6.22MB | **-75%** |
| 추정 FreqSep 시간 | ~14ms | ~8.9ms | **-36%** |

### 5.3 대역폭 절감 추정

1080p RGBA 텍스처 기준 (read + write per pass):

| 경로 | 읽기 대역폭 | 쓰기 대역폭 | 총 대역폭 |
|------|------------|------------|-----------|
| HIGH: 4 blur @ 1080p | 4 x 8.29MB = 33.2MB | 4 x 8.29MB = 33.2MB | 66.4MB |
| MID: 4 blur @ 540p | 4 x 2.07MB = 8.3MB | 4 x 2.07MB = 8.3MB | 16.6MB |
| Composite (동일) | ~16.6MB (4 input) | 8.29MB | 24.9MB |
| **HIGH 총합** | | | **91.3MB** |
| **MID 총합** | | | **41.5MB** |
| **절감** | | | **-54.6%** |

모바일 GPU의 메모리 대역폭이 25~50 GB/s 범위인 것을 감안하면,
프레임당 ~50MB 절감은 **~1~2ms 시간 절감**에 해당한다.

---

## 6. OpenGL 상태 관리

### 6.1 [Medium] Viewport 전환 패턴 검증

**위치**: `gpu_beauty_backend.cpp:1259, 1303`

```cpp
glViewport(0, 0, half_w, half_h);   // line 1259: half-res 전환
// ... Pass 1a, 1b, 2a, 2b ...
glViewport(0, 0, width, height);     // line 1303: full-res 복원
```

**분석**: Viewport 전환은 올바르게 관리되고 있다. half-res 진입 시 축소, Composite 전에 복원.
다만 `executeFreqSepPipelineHalfRes`가 실패하여 조기 반환하는 경우,
line 1235-1240의 에러 경로에서 **viewport가 복원되지 않는다**.

현재 에러 경로는 텍스처 획득 실패 시 발생하며, 이 시점에서는 아직 viewport가 변경되지 않았으므로
(glViewport 호출이 line 1259에 있음) 실제 문제는 없다. 하지만 미래 코드 변경 시 위험할 수 있다.

**추정 영향**: 현재 0 (에러 경로가 viewport 변경 전)

**권장 조치**: 방어적으로 RAII 패턴의 viewport 복원 guard 추가를 고려.
```cpp
struct ViewportGuard {
    GLint prev[4];
    ViewportGuard() { glGetIntegerv(GL_VIEWPORT, prev); }
    ~ViewportGuard() { glViewport(prev[0], prev[1], prev[2], prev[3]); }
};
```
다만 `glGetIntegerv` 호출의 오버헤드(~0.01ms)를 감안하면, 현재 상태 유지도 합리적이다.

### 6.2 Scissor 상태 관리

**위치**: `gpu_beauty_backend.cpp:1660-1682`

```cpp
if (scissor_active) {
    glDisable(GL_SCISSOR_TEST);   // FreqSep 전에 비활성화
}
// ... FreqSep 실행 ...
if (scissor_active) {
    glEnable(GL_SCISSOR_TEST);    // FreqSep 후 복원
}
```

**분석**: Scissor 상태의 save/restore가 올바르게 구현되어 있다.
FreqSep이 실패하여 `runBilateralFallback()`으로 대체되는 경우에도
scissor 복원이 정상적으로 수행된다.

**추정 영향**: 정상. 추가 조치 불필요.

---

## 7. 확장성 분석

### 7.1 [High] 정적 DeviceTier 판정의 한계

**위치**: `gpu_beauty_backend.cpp:1155-1211`

**분석**: 현재 `detectDeviceTier()`는 `GL_RENDERER` 문자열만으로 판정하며,
`initialize()` 시 1회 결정 후 변경되지 않는다. 다음의 엣지 케이스에서 문제가 발생할 수 있다:

1. **열 스로틀링(Thermal Throttling)**: 기기가 과열되면 GPU 클럭이 동적으로 하강한다.
   HIGH tier로 판정된 Adreno 730이 지속 사용 시 MID급 성능으로 떨어져도
   full-res 파이프라인을 계속 실행하여 프레임 드롭이 발생한다.

2. **경계 디바이스(Borderline Devices)**: Adreno 640 (MID)과 Adreno 650 (MID)은
   실제 성능 차이가 크지만 동일 tier로 분류된다.

3. **미인식 GPU**: 새로운 GPU 벤더나 알려지지 않은 렌더러 문자열은
   무조건 LOW로 분류되어 최적 경로를 사용하지 못한다.

4. **Mali GPU 파싱 정확도**: Mali-G710은 `num=710`으로 HIGH,
   Mali-G78은 `num=78`로 MID. 이 분류는 실제 성능과 대체로 일치하나,
   Mali-G715(=715, HIGH)과 Mali-G610(=610, 파싱 결과 없음 -> find("Mali-G") 후
   digit 추출 시 610 -> 조건에 없으므로 LOW)에서 의도치 않은 결과가 나올 수 있다.

   **Mali 파싱 오류 확인**: Mali-G610은 `num=610`이 되고, 조건이 `>=710 → HIGH, >=70 → MID`이므로
   610 >= 70은 true → MID로 올바르게 분류된다. 문제없음.

**추정 영향**: 열 스로틀링 시 HIGH tier에서 프레임 드롭 가능 (33ms 초과)

**권장 조치** (P4-W3-05에서 구현 가능):
1. **런타임 적응형 tier 전환**: GPUProfiler의 FreqSep 측정 시간을 누적하여,
   연속 N프레임이 목표 시간을 초과하면 자동으로 한 단계 하향.
   ```cpp
   // 의사 코드
   if (freqsep_avg_ms > 12.0f && consecutive_slow_frames > 10) {
       if (device_tier_ == DeviceTier::HIGH) device_tier_ = DeviceTier::MID;
       else if (device_tier_ == DeviceTier::MID) device_tier_ = DeviceTier::LOW;
   }
   ```
2. **히스테리시스**: tier 하향 후 성능이 안정되면 일정 시간(예: 5초) 후 상향 시도.
3. **벤치마크 기반 초기 판정**: initialize() 시 더미 텍스처로 1회 벤치마크를 실행하여
   실제 GPU 성능을 측정하는 방식 (작업 문서 §7에서도 언급).

### 7.2 [Low] roi_ptr 직접 변이(Mutation)

**위치**: `gpu_beauty_backend.cpp:1631-1632`

```cpp
roi_ptr->face_rect.x += dx;
roi_ptr->face_rect.y += dy;
```

**분석**: One Euro Filter가 `face_rect`의 중심 좌표를 안정화한 후,
원본 `roi_ptr`의 값을 직접 수정한다. 이는 호출자의 BeautyROI 데이터를 변이시킨다.

현재 `applyTextureId()` 내에서 `roi_ptr`은 이 시점 이후에만 사용되므로
기능적으로는 문제가 없다. 하지만 미래에 ROI를 다른 용도로 재사용하거나,
멀티스레드 환경에서 동일 ROI를 참조하는 경우 부작용이 발생할 수 있다.

**추정 영향**: 현재 0. 잠재적 유지보수 리스크.

**권장 조치**: 로컬 복사본을 사용하거나, 오프셋을 별도 변수로 관리.
```cpp
float mask_offset_x = stable_cx - cx;
float mask_offset_y = stable_cy - cy;
// uploadSkinMask() 또는 Composite 셰이더에서 오프셋 적용
```

---

## 8. 발견 사항 종합

| # | 심각도 | 영역 | 이슈 | 추정 영향 | 권장 조치 |
|---|--------|------|------|-----------|-----------|
| 1 | **High** | 확장성 | 정적 DeviceTier 판정 - 열 스로틀링 미대응 | 과열 시 프레임 드롭 | 런타임 적응형 tier 전환 (P4-W3-05) |
| 2 | Medium | GPU 상태 | 중복 glTexParameteri(GL_LINEAR) 4회/프레임 | <0.05ms | 제거 + 주석으로 TexturePool 기본값 명시 |
| 3 | Medium | 메모리 | Half-res 텍스처 풀 혼재(full+half) 모니터링 부재 | 추가 6.22MB | PoolStats 로깅 추가 |
| 4 | Medium | GL 상태 | executeFreqSepPipelineHalfRes 에러 경로의 viewport 안전성 | 현재 0 (잠재적) | RAII guard 고려 |
| 5 | Low | GPU 상태 | 중복 glUseProgram 1회/프레임 | <0.01ms | 유지 (가독성) |
| 6 | Low | 메모리 | One Euro Filter 3 인스턴스 | 144 bytes | 조치 불필요 |
| 7 | Low | CPU | detectDeviceTier() 문자열 파싱 | 0.01ms (1회) | 조치 불필요 |
| 8 | Low | 정확성 | roi_ptr 직접 변이 | 잠재적 리스크 | 로컬 복사본 고려 |

---

## 9. 긍정적 설계 패턴

다음 항목들은 성능 관점에서 올바르게 설계되었다:

1. **detectDeviceTier()의 1회 호출**: `initialize()`에서만 호출되고 결과를 캐싱.
   Phase 1 컨텍스트에서 우려한 "반복 호출 시 성능 저하"는 해당사항 없음.

2. **Gaussian weights CPU 사전 계산**: `computeGaussianWeights()`가 셰이더 실행 전에
   CPU에서 가중치를 계산하여 `glUniform1fv`로 일괄 전송. GPU 연산 절감.

3. **셰이더 전환 최소화**: Gaussian 4패스를 연속 실행 후 Composite 1회 전환.

4. **Half-res 텍스처의 bilinear 업샘플링**: 저주파 성분(lowFreq, smoothedLow)에
   GL_LINEAR를 적용하여 별도 업샘플링 패스 없이 하드웨어 보간 활용.
   이론적으로 저주파 성분의 bilinear 품질 손실은 미미하다.

5. **TexturePool 기반 중간 버퍼 관리**: 프레임마다 allocate/free하지 않고
   풀에서 acquire/release하여 GPU 메모리 할당 오버헤드 제거.

6. **Bilateral fallback 안전망**: FreqSep 실패 시(셰이더 미컴파일, 텍스처 획득 실패, LOW tier)
   일관된 fallback 경로(`runBilateralFallback`)를 제공하여 프레임 드롭 방지.

7. **GL_LINEAR 필터 미복원 이슈 (Phase 1 컨텍스트)**: TexturePool 기본 필터가 GL_LINEAR이므로
   half-res 텍스처 반환 시 원래 필터 모드 미복원은 **문제가 아니다**. 확인 완료.

---

## 10. 결론

P4-W3-04의 변경은 **33ms 프레임 버짓을 안정적으로 충족**할 수 있다.
MID tier 하이브리드 파이프라인은 대역폭 54.6% 절감, GPU 메모리 75% 절감이라는
의미 있는 성능 이득을 제공하면서도 비피부 영역의 원본 품질을 보존한다.

**즉시 조치가 필요한 사항은 없으며**, 가장 우선순위가 높은 개선은
P4-W3-05에서 계획된 **런타임 적응형 tier 전환** 구현이다.
이를 통해 열 스로틀링 환경에서의 프레임 안정성을 확보할 수 있다.
