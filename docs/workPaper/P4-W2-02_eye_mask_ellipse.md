# P4-W2-02: 비대칭 타원 Eye Mask (선택적)

## 작업 개요
- **Phase**: P4 (시각적 리얼리즘 — Visual Fidelity)
- **기간**: TBD
- **상태**: ⏳ 대기
- **선행 조건**: P4-W2-01 완료 + 렌즈 경계 누출이 여전히 거슬릴 경우 진행
- **근거**: 브레인스토밍 Section 3, 7, 10, 11, 13 합의

## 목표

현재 Y축 상/하 2개 경계(slab) 클리핑을 **16점 랜드마크 기반 비대칭 타원 SDF 마스킹**으로 교체하여, 좌우/대각 방향 렌즈 누출을 방지하고 내안각(Caruncle) 영역을 보호한다.

### 현재 문제

```
현재 마스크: Y축 slab (eyeTop, eyeBottom)
├─ 좌우(외안각/내안각) 방향 누출 발생
├─ 대각 방향 누출 발생
└─ 눈꺼풀의 곡면 형태 미반영

제안 마스크: 비대칭 타원 (Asymmetric Ellipse)
├─ 아몬드형 눈 윤곽에 근접
├─ 내안각 쪽 반경 축소로 붉은 살 보호
└─ fragment당 연산: 16점 polygon SDF 대비 ~10배 적음
```

## 구현 사항

### 1. FaceMesh 눈 윤곽 랜드마크 (16점)

```
왼쪽 눈: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
오른쪽 눈: [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
```

내/외안각 코너 랜드마크:
- 좌안: **33** (내안각, 코 쪽), **133** (외안각, 귀 쪽)
- 우안: **263** (내안각, 코 쪽), **362** (외안각, 귀 쪽)

### 2. CPU 측: 16점 → 타원 피팅

런타임에 16점 polygon을 바운딩 타원으로 피팅:

```kotlin
data class EyeEllipse(
    val cx: Float,      // 중심 X
    val cy: Float,      // 중심 Y
    val rxInner: Float, // 내안각 쪽 반경 (짧음)
    val rxOuter: Float, // 외안각 쪽 반경 (길음)
    val ry: Float,      // Y 반경
    val rotation: Float // 회전 (라디안)
)

fun fitEyeEllipse(landmarks: List<PointF>, innerCornerIdx: Int, outerCornerIdx: Int): EyeEllipse {
    // 1. 16점의 바운딩 박스로 중심/반경 초기 추정
    // 2. 내안각(innerCornerIdx) vs 외안각(outerCornerIdx) x좌표 차이로 비대칭 반경 결정
    //    rxInner = center - inner.x (코 쪽, 짧게)
    //    rxOuter = outer.x - center (귀 쪽, 길게)
    // 3. 회전은 inner→outer 벡터의 각도
    // 4. One Euro Filter 적용 (center, radii 각각)
}
```

**Uniform 전달 (눈 1개당 6개 값):**
```kotlin
glUniform2f(uLeftEyeCenter, cx, cy)
glUniform3f(uLeftEyeRadii, rxInner, rxOuter, ry)
glUniform1f(uLeftEyeRotation, rotation)
```

### 3. GLSL: 비대칭 타원 SDF

```glsl
float asymmetricEllipseMask(vec2 uv, vec2 center, vec3 radii, float rotation, float feather) {
    // radii = (rxInner, rxOuter, ry)
    vec2 d = uv - center;

    // 회전 적용
    float cosR = cos(rotation);
    float sinR = sin(rotation);
    d = vec2(d.x * cosR + d.y * sinR, -d.x * sinR + d.y * cosR);

    // 내안각(d.x < 0) vs 외안각(d.x >= 0) 비대칭 반경
    float rx = (d.x < 0.0) ? radii.x : radii.y;
    float ry = radii.z;

    float ellipseDist = length(vec2(d.x / rx, d.y / ry));
    return smoothstep(1.0, 1.0 - feather, ellipseDist);
}
```

**장점**:
- Fragment당 연산: `cos`, `sin`, `length`, `smoothstep` — polygon SDF(16점 루프) 대비 **~10배 경량**
- 아몬드형 눈 윤곽에 시각적으로 유사한 마스킹 결과

**주의사항**:
- `d.x < 0.0` 판정은 눈의 좌우 방향(회전)에 의존
- 내안각/외안각 방향: 좌안 기준 33(내안각, 코 쪽) → 133(외안각, 귀 쪽) 벡터로 결정
- 우안은 263(내안각) → 362(외안각) 동일 로직
- 측면 뷰에서도 올바르게 동작하려면 방향벡터를 uniform으로 전달 필요

### 4. 기존 Y-slab 마스킹과의 전환

```glsl
// feature flag로 전환
if (uUseEllipseMask == 1) {
    alpha *= asymmetricEllipseMask(vTexCoord, uLeftEyeCenter, uLeftEyeRadii, uLeftEyeRotation, feather);
} else {
    // 기존 Y-slab 마스킹 유지
    alpha *= smoothstep(minY, minY + feather, vTexCoord.y)
           * (1.0 - smoothstep(maxY - feather, maxY, vTexCoord.y));
}
```

기존 마스킹을 **제거하지 않고** feature flag로 전환 가능하게 유지. 타원 마스킹이 문제를 일으킬 경우 즉시 rollback 가능.

### 5. Caruncle 보호 (내안각 보호)

내안각 쪽 반경(`rxInner`)을 외안각(`rxOuter`)보다 짧게 설정:

```
rxInner = distance(center, innerCorner) × 0.85  // 15% 축소
rxOuter = distance(center, outerCorner) × 1.0   // 그대로
```

내안각은 살색/붉은색이라 렌즈가 덮이면 "좀비 눈"처럼 보이는 문제를 방지.

## 수정 대상 파일

| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `CameraGLRenderer.kt` | GLSL 비대칭 타원 함수 + uniform 추가 + feature flag |
| 2 | `CameraGLRenderer.kt` | CPU 측 16점 → 타원 피팅 로직 |

## 검증 체크리스트

- [ ] 정면 뷰: 렌즈가 눈 윤곽 밖으로 새지 않음
- [ ] 측면(30°) 뷰: 내안각 쪽 렌즈 누출 방지 확인
- [ ] 깜빡임: 타원 경계가 자연스럽게 따라감
- [ ] 성능: 기존 Y-slab 대비 fragment shader 추가 비용 < 0.5ms
- [ ] Rollback: feature flag OFF 시 기존 마스킹으로 즉시 복귀 확인

## 스모크 테스트 (권장 25~40분, 선택 작업)

### 목적
- Ellipse mask를 켰을 때 누출 개선이 실제로 보이는지, 성능/롤백이 안전한지 확인한다.

### 실행 순서

1. 빌드/실행
   - 명령: `cd android && ./gradlew :demo-app:assembleDebug`
   - 확인: 앱 실행 및 기본 렌더 정상

2. ON/OFF 비교 (동일 조건)
   - 동작: `uUseEllipseMask=0` 상태에서 10초 관찰 후, `uUseEllipseMask=1`로 전환
   - 확인: ON에서 좌우/대각 누출이 눈에 띄게 감소

3. 각도 시나리오 확인
   - 동작: 정면 → 좌/우 30° 회전
   - 확인: 내안각(Caruncle) 영역 침범 감소, 경계 깨짐 없음

4. 깜빡임 추적 확인
   - 동작: 자연 깜빡임 10회
   - 확인: 경계가 눈꺼풀 움직임을 무난하게 따라가며 떨림/끊김 없음

5. 성능/롤백 확인
   - 확인:
     - ON 시 렌더 시간 증가가 0.5ms 이내
     - OFF로 즉시 되돌렸을 때 기존 Y-slab 동작 정상 복귀

### 내가 확인해야 하는 핵심
- Ellipse mask의 목적은 "클리핑 미세 개선"이다.  
  개선이 체감되지 않거나 성능 비용이 큰 경우 이 작업은 스킵하는 것이 맞다.

### 최종 판정
- 필수 통과: 1, 2, 3, 5(롤백)
- 권장 통과: 4, 5(성능)
- 필수 항목 실패 시 해당 기능은 기본 OFF 유지

## 진행 조건

이 작업은 **선택적(optional)**입니다.
- P4-W2-01(Sclera + Shadow) 완료 후 렌즈 경계 누출이 **여전히 거슬릴 경우**에만 진행
- P4-W2-01의 Sclera Protection이 충분히 경계 오차를 보정하면 **스킵 가능**

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-13 | 작업 계획 문서 작성 |
| 2026-02-13 | Codex 리뷰 반영: 역순 smoothstep UB 수정, 내/외안각 인덱스 충돌 해소 |
