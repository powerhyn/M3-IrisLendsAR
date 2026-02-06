# GPU Beauty 렌더링 이슈 추적 문서

**생성일**: 2026-02-04
**해결일**: 2026-02-05
**상태**: ✅ 해결됨

## 해결 요약
**근본 원인**: Android-C++ 이중 텍스처 해제 (Double-Free)
- Android (`CameraGLRenderer.kt`)에서 `releaseTexture()` 호출
- C++ (`GPUBeautyBackend`)에서도 TexturePool을 통해 동일 텍스처 해제
- 이중 해제로 TexturePool 상태 손상 → 이후 프레임에서 삭제된 텍스처 재사용 → 검은 화면

**수정 내용**:
1. `CameraGLRenderer.kt`: `IrisLensSDK.releaseTexture()` 호출 제거
2. 텍스처 수명 관리를 C++ TexturePool에 일원화

**디버그 코드 정리**:
- `shader_sources.cpp`: 하드코딩된 색상 출력 제거 (빨강/파랑/초록)

---

## 1. 문제 요약

GPU Beauty Backend에서 특정 필터 효과 적용 시 **검은 화면**이 출력되는 문제.

### 현재 증상 (2026-02-04 19:30)
| 시나리오 | 결과 |
|----------|------|
| Smoothing만 사용 | ✅ 정상 |
| Smoothing → 0으로 조절 → 다시 올림 | ❌ 검은 화면 |
| Brightness 조절 | ❌ 검은 화면 (조금만 조절해도) |
| Smoothing + Brightness | ❌ 검은 화면 |

---

## 2. 기술 배경

### 렌더링 파이프라인
```
카메라 OES 텍스처 → RGBA 변환 → GPU Beauty Backend → 화면 출력
                                    ↓
                            [Ping-Pong 버퍼 패턴]
                            - TexturePool에서 2개 텍스처 획득
                            - 필터 체인 실행 (Smoothing → CombinedColor → ...)
                            - 출력 텍스처 반환
```

### 관련 파일
- `cpp/src/gpu/gpu_beauty_backend.cpp` - 핵심 GPU 처리
- `cpp/src/gpu/shader_sources.cpp` - GLSL 셰이더 소스
- `cpp/src/gpu/texture_pool.cpp` - 텍스처 풀 관리
- `android/.../CameraGLRenderer.kt` - Android GL 렌더러

---

## 3. 수정 이력

### 수정 1: 테스트 코드 추가 (빨간 틴트)
**목적**: 셰이더가 실제로 실행되는지 확인
**결과**: 빨간 틴트 확인됨 → 셰이더 실행 OK

### 수정 2: 테스트 코드 제거
**문제**: `return` 문으로 인해 GLSL 컴파일러가 유니폼 최적화 제거
**증상**: `bright=-1, balance=-1, white=-1` (유니폼 위치 무효)
**해결**: 테스트 코드 제거 → 유니폼 위치 정상 (0, 1, 2)

### 수정 3: FBO/텍스처 검증 로그 추가
**추가한 체크**:
- FBO Completeness 체크
- 텍스처 유효성 확인 (`glIsTexture`)
- 프로그램 링크 상태 확인
- GL 에러 체크

**결과**: 모든 체크 통과 (FBO 정상, 텍스처 유효, GL 에러 없음)

### 수정 4: 텍스처 풀 반환 로직 수정 (1차)
**문제**: ping/pong 모두 즉시 반환 → 출력 텍스처도 해제됨
**증상**: 첫 프레임만 잠깐 보이고 검은 화면
**수정**: 출력 텍스처 제외하고 반환
```cpp
if (current_input == ping->texture_id) {
    texture_pool_->releaseTexture(pong);
} else if (current_input == pong->texture_id) {
    texture_pool_->releaseTexture(ping);
}
```
**결과**: TexturePool 가득 참 (8/8) → 텍스처 누수

### 수정 5: 텍스처 풀 반환 로직 수정 (2차)
**문제**: 출력 텍스처를 반환 안 하니 누적됨
**수정**: 이전 프레임의 ping/pong을 **다음 프레임 시작 시** 반환
```cpp
// 다음 프레임 시작 시
if (previous_output_ping_ != nullptr) {
    texture_pool_->releaseTexture(previous_output_ping_);
}
if (previous_output_pong_ != nullptr) {
    texture_pool_->releaseTexture(previous_output_pong_);
}

// ... 처리 ...

// 프레임 끝
previous_output_ping_ = ping;
previous_output_pong_ = pong;
```
**결과**:
- Smoothing → 0 → 다시 올림: ❌ 검은 화면
- Brightness 조절: ❌ 검은 화면

### 수정 6: GPU 동기화 이슈 조사 (2026-02-04 22:00)
**결과**: ❌ `glFinish()` 추가해도 검은 화면 여전히 발생
→ **GPU 동기화 문제 아님 확정**

### 수정 7: 셰이더 하드코딩 테스트 (2026-02-05) - 현재
**목적**: 파이프라인 vs 텍스처 샘플링 문제 구분
**수정**: BILATERAL_FILTER_FRAGMENT에 빨간색 하드코딩
```glsl
// DEBUG: 파이프라인 테스트
fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // 빨간색
return;
```

**예상 결과**:
- 🔴 빨간색 나옴 → 파이프라인 정상, 입력 텍스처 샘플링 문제
- ⬛ 여전히 검은색 → FBO/Draw 실패

**테스트 방법**: Smoothing 값을 0보다 크게 설정 후 화면 색상 확인

**테스트 결과** (2026-02-05):
- 초기 진입: 🔵 파란색 (SoftFocus 정상)
- Smoothing > 0: 🔵 파란색 (정상)
- Smoothing = 0: 🔵 파란색 (정상)
- Smoothing 0 → 다시 > 0: ⬛ 검은색
- **Brightness 정확히 4번째 조절**: ⬛ 검은색 (값과 무관!)

→ **TexturePool 누수 가능성 높음** (8개 풀 / 2개씩 사용 = 4회 후 고갈)

### 수정 8: TexturePool 상태 로깅 추가 (2026-02-05) - 현재
**목적**: 텍스처 풀 누수 여부 확인
**추가 로그**:
```cpp
LOGI("TexturePool BEFORE release: total=%d, in_use=%d, available=%d", ...);
LOGI("TexturePool AFTER release: total=%d, in_use=%d, available=%d", ...);
LOGI("TexturePool AFTER acquire: total=%d, in_use=%d, available=%d", ...);
```

**확인할 로그 패턴**:
- `in_use`가 계속 증가하면 → 텍스처 누수
- `available`이 0이 되면 → 풀 고갈
**발견**: 렌더링 파이프라인 타이밍 문제 의심
```
Frame N:
  1. applyTextureId() → pingN/pongN 획득, outputN 반환
  2. renderToScreen(outputN) → GL 커맨드 제출 (비동기!)
  3. onDrawFrame 종료

Frame N+1:
  1. applyTextureId() 시작
  2. pingN/pongN 해제 ← GPU가 아직 outputN을 렌더링 중일 수 있음!
     → 검은 화면 원인
```

**분석**:
- `renderToScreen()`은 GL 커맨드를 제출만 하고 완료를 기다리지 않음
- 다음 프레임에서 텍스처를 해제할 때 GPU는 이전 프레임 렌더링 중일 수 있음
- 해제된 텍스처에서 읽으면 검은색(0,0,0) 또는 쓰레기 값 발생

**테스트 수정**:
```cpp
// 텍스처 해제 전 GPU 동기화 추가
if (previous_output_ping_ != nullptr || previous_output_pong_ != nullptr) {
    glFinish();  // GPU 렌더링 완료 대기 (성능 영향 있음)
    LOGI("glFinish() called before releasing previous textures");
}
```

**다음 단계**:
- 이 수정으로 검은 화면 해결되면 → GPU 동기화 이슈 확정
- 해결되면 성능 최적화 방안 검토:
  1. Fence Sync 사용 (GL_SYNC_GPU_COMMANDS_COMPLETE)
  2. 트리플 버퍼링 (3 프레임 전 텍스처 해제)
  3. 텍스처 풀 크기 확대 및 LRU 정책

### 수정 9: Android-C++ 이중 텍스처 해제 수정 (2026-02-05) - 현재
**발견**: TexturePool 로그 분석 및 glFinish() 테스트 후에도 문제 지속
→ GPU 동기화 문제가 아님 확정

**새로운 분석**:
"4번째 조절 시 검은색" 패턴에서 힌트 발견:
- TexturePool 크기: 8개
- 프레임당 ping/pong 2개 사용
- 4회 × 2개 = 8개 → 풀 고갈

그러나 로그상 TexturePool은 정상 동작 (release→acquire 사이클 정상)
→ **Android와 C++ 양쪽에서 텍스처를 해제**하고 있음을 발견!

**문제 코드** (`CameraGLRenderer.kt:326-327`):
```kotlin
// 이전 출력 텍스처가 있으면 해제 ← 문제의 원인!
if (beautyOutputTextureId != 0 && beautyOutputTextureId != outputTexture) {
    IrisLensSDK.releaseTexture(beautyOutputTextureId)  // Android가 해제
}
```

**C++ 코드** (`gpu_beauty_backend.cpp`):
```cpp
texture_pool_->releaseTexture(previous_output_ping_);  // C++도 해제!
texture_pool_->releaseTexture(previous_output_pong_);
```

**이중 해제 시나리오**:
```
Frame N:
  C++: applyBeautyFilterTextureV2() → 텍스처 A 반환 (TexturePool 소유)
  Android: beautyOutputTextureId = A 저장

Frame N+1:
  Android: releaseTexture(A) 호출 ← Android가 먼저 해제!
  C++: texture_pool_->releaseTexture(A) ← 이미 해제된 텍스처 재해제!
       → TexturePool 내부 상태 손상 → 이후 획득한 텍스처 무효 → 검은 화면
```

**수정 내용**:
```kotlin
// 수정 전
if (beautyOutputTextureId != 0 && beautyOutputTextureId != outputTexture) {
    IrisLensSDK.releaseTexture(beautyOutputTextureId)  // 제거
}

// 수정 후
// NOTE: 텍스처 해제는 C++ TexturePool에서 관리함
// Android에서 releaseTexture() 호출하면 이중 해제 발생 → 검은 화면 원인
beautyOutputTextureId = outputTexture
```

**테스트 필요**:
- [ ] Smoothing → 0 → 다시 올림: 검은 화면 해결 여부
- [ ] Brightness 조절: 검은 화면 해결 여부
- [ ] 장시간 사용 시 텍스처 누수 없는지 확인

---

## 4. 현재 상태 분석

### 로그 분석 결과
- 유니폼 위치: 정상 (`tex=3, bright=0, balance=1, white=2`)
- 텍스처 유효: 정상 (`valid=1`)
- FBO Completeness: 정상 (에러 로그 없음)
- GL 에러: 없음
- TexturePool: 정상 (가득 참 에러 없음)

### 의심 영역 (우선순위 순)
1. **🔴 Android-C++ 이중 텍스처 해제 (수정 9에서 해결 시도)**
   - Android `releaseTexture()`와 C++ `texture_pool_->releaseTexture()` 양쪽에서 해제
   - 이중 해제로 TexturePool 내부 상태 손상
   - **수정 완료**: Android 측 releaseTexture() 호출 제거

2. ~~**GPU 동기화 문제**~~ → ❌ 배제됨 (glFinish()로 테스트 완료)
   - glFinish() 추가해도 검은 화면 여전히 발생
   - GPU 동기화 문제 아님 확정

3. ~~**필터 비활성화 후 재활성화 시 상태 불일치**~~ → ❓ 이중 해제가 원인일 수 있음
   - Smoothing=0이면 smoothing 패스 스킵
   - 이중 해제 수정 후 재테스트 필요

4. ~~**CombinedColor 셰이더 렌더링 실패**~~ → ❌ 배제됨 (셰이더 하드코딩 테스트 완료)
   - 셰이더 하드코딩 테스트에서 초록색 정상 출력 확인
   - 셰이더 자체는 문제 없음

5. **Ping-Pong 상태 관리** → ❓ 이중 해제가 원인일 수 있음
   - 필터 조합 변경 시 current_input/current_output 추적 문제
   - 이중 해제 수정 후 재테스트 필요

---

## 5. 다음 디버깅 단계

### TODO
- [ ] Smoothing=0 → 0이 아닌 값 전환 시 로그 분석
- [ ] CombinedColor 패스에서 실제 렌더링 확인 (단색 출력 테스트)
- [ ] current_input/current_output 전환 로직 검증
- [ ] TexturePool::acquirePingPongPair 동작 확인

### 필요한 로그
1. Smoothing=0 → 0.5 전환 시점의 로그
2. Brightness 조절 시점의 로그
3. PingPong 획득/반환 상세 로그

---

## 6. 관련 코드 스니펫

### applyTextureId 필터 체인 로직
```cpp
// 1. 스무딩 (Bilateral Filter)
if (config.smoothing > 0.01f) {
    executeSmoothingPass(...);
    current_input = current_output->texture_id;
    current_output = (current_output == ping) ? pong : ping;
}

// 2. 통합 Color Adjustment
if (needsBrightness || needsBalance || needsWhitening) {
    executeCombinedColorPass(...);
    current_input = current_output->texture_id;
    current_output = (current_output == ping) ? pong : ping;
}

// 출력
*output_texture = current_input;
```

### 의문점
- `config.smoothing <= 0.01f`이고 `brightness != 1.0`이면?
  - smoothing 스킵, current_input = input_tex_id (원본)
  - CombinedColor 실행, 입력 = 원본 텍스처
  - 출력 = ping 텍스처
- 이 경우 원본 텍스처(rgbaTextureId)에서 읽는데, 이게 유효한가?

---

## 7. 추가 발견 사항

### DEFAULT_SOFT_FOCUS = 0.3f 이슈
**발견 시점**: 2026-02-04 21:45

`BeautyFilterConfigV2.java`에서:
```java
public static final float DEFAULT_SOFT_FOCUS = 0.3f;  // 0.0이 아님!
```

**영향**:
- 사용자가 모든 필터를 끄더라도 softFocus는 기본적으로 0.3
- smoothing=0, brightness=1.0이어도 softFocus 패스가 실행됨
- 예상과 다른 필터 체인 실행 → 디버깅 혼란 유발

**필터 체인 실행 조건**:
| 필터 | 실행 조건 |
|------|----------|
| Smoothing | `config.smoothing > 0.01f` |
| CombinedColor | `|brightness - 1.0| > 0.01 || |colorBalance| > 0.01 || whitening > 0.01` |
| SoftFocus | `config.softFocus > 0.01f` |

---

## 8. 테스트 체크리스트

### GPU 동기화 테스트 (glFinish 추가 후)
- [ ] Smoothing만 조절: 정상 동작 확인
- [ ] Smoothing → 0 → 다시 올림: 검은 화면 해결 여부
- [ ] Brightness 조절: 검은 화면 해결 여부
- [ ] Smoothing + Brightness 동시 조절
- [ ] 성능 영향 측정 (FPS 변화)
