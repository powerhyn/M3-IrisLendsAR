# P3-W2-01: P3 안정화 — 좌표계 정렬 + 텍스처 소유권 + ROI 연동 + Stub 정리

## 작업 개요
- **Phase**: P3 (안정화 및 완성도 제고)
- **기간**: 2026-02-11 ~
- **상태**: ✅ 완료
- **근거**: Codex 분석 보고서에 대한 Claude, Gemini, Codex 3개 AI의 공통 지적 사항 + 리뷰 코멘트 반영

## 목표
Super Color Shader + LUT 통합(`54b6f50`) 완료 후, 세 AI가 합의한 핵심 리스크와 ISS-003 좌표계 이슈를 해소하여 P3 안정성을 확보한다.

### 합의된 우선순위

| 순위 | 항목 | 합의 수준 | 핵심 근거 |
|------|------|-----------|-----------|
| 0 | ISS-003 좌표계 정렬 (Mesh 불일치 + 렌즈 왜곡) | 리뷰 추가 | P0 시각적 결함, 렌즈 타원 왜곡 직접 재현 |
| 1 | 텍스처 소유권 규약 명문화 | 3/3 동의 | 실제 검은 화면 크래시 이력 |
| 2 | Detection Handle → ROI 활성화 | 3/3 동의 | 전체 프레임 GPU 낭비 + 품질 저하 |
| 3 | Stub 경로 정리 | 3/3 동의 | apply() CPU fallback, applyFaceWarp() 미구현 |

---

## 수행 내역

### Phase 0: ISS-003 좌표계 정렬 (Critical — P0)

> 리뷰 코멘트 #1: ISS-003 핵심 범위(Mesh 불일치/렌즈 세로 왜곡)가 빠져 있었음.
> 리뷰 코멘트 #6: 검증 항목이 좌표/왜곡 회귀를 커버하지 못함.

**관련 문서**: `docs/workPaper/ISS-003_gpu_lens_mesh_coordinate_alignment_plan.md`

#### 0-1. 렌즈 셰이더 좌표 기준 통일
- **파일**: `android/demo-app/.../CameraGLRenderer.kt` (line 618, 661)
- **문제**: `fboWidth/fboHeight` 기준 반경 정규화와 `uFrameAspect` 계산이 검출 좌표계(`result.frameWidth/frameHeight`)와 불일치
- **수정**:
  - `result.frameWidth/frameHeight`를 렌즈 계산 1순위 기준으로 사용
  - `uFrameAspect`를 동일 기준으로 산출
  - 회전(0/90/270) 시 width/height swap을 `resolveCoordinateSpace()` 유틸로 캡슐화

#### 0-2. Overlay 좌표 매핑 GL 정책 통일
- **파일**: `android/demo-app/.../OverlayView.kt` (line 476)
- **문제**: Overlay는 `fill-center(max)` 기준 매핑, GL 출력은 `fit` 기준 → Mesh 디버그 점과 GL 영상 불일치
- **수정**:
  - GPU 모드에서는 `fit` 기반 매핑 사용 (GL 화면 출력과 동일)
  - `computeScreenTransform(imageW, imageH, viewW, viewH, mode)` 공통 함수 추출

#### 0-3. IrisResult 불변 스냅샷 전달
- **파일**: `android/demo-app/.../GpuRenderActivity.kt`
- **문제**: 단일 `irisResult` 객체를 GL/UI 스레드가 공유하여 프레임 경합
- **수정**: Analyzer에서 `IrisResult` 깊은 복사 스냅샷을 GL/UI에 전달

---

### Phase 1: 텍스처 소유권 규약 명문화 (Critical)

#### 1-1. GPUBeautyBackend::releaseTexture() 안전성 보강
- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 1099-1126)
- **문제**: `releaseTexture()`가 TexturePool 관리 여부 미확인, `glDeleteTextures()` 직접 호출 (line 1120)

> 리뷰 코멘트 (추가 #1): `isManaged(GLuint)` 만으로는 불충분.
> TexturePool의 `releaseTexture()`는 `TextureInfo*`를 받으므로, texture_id → TextureInfo* 조회 API가 필요.

> 리뷰 코멘트 (추가 #7): `findByTextureId` 반환 포인터를 외부에서 보관하면 `trim`/`release` 시 댕글링 위험.
> 리뷰 코멘트 (추가 #8): 문서 전반에서 `isManaged`와 `findByTextureId` 혼용 → 용어 통일 필요.

- **수정** — API를 `releaseTextureById(GLuint)` 단일 메서드로 통일 (조회+반환 원자화):
  - TexturePool에 `releaseTextureById(GLuint texture_id)` 추가:
    ```cpp
    // texture_pool.h
    bool releaseTextureById(GLuint texture_id);

    // texture_pool.cpp — 조회+반환을 lock 안에서 원자적 수행
    bool TexturePool::releaseTextureById(GLuint texture_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& info : textures_) {
            if (info->texture_id == texture_id) {
                if (!info->in_use) return false;  // 이미 반환됨
                releaseTextureLocked(info.get());  // in_use=false
                return true;  // 풀 관리 텍스처였음
            }
        }
        return false;  // 풀 미관리 텍스처
    }
    ```
  - `GPUBeautyBackend::releaseTexture(uint32_t texture)` 수정:
    ```cpp
    GLuint tex_id = static_cast<GLuint>(texture);
    if (texture_pool_ && texture_pool_->releaseTextureById(tex_id)) {
        LOGI("Released pool-managed texture %u", tex_id);
    } else {
        glDeleteTextures(1, &tex_id);  // 외부 텍스처만 직접 삭제
        LOGI("Released external texture %u", tex_id);
    }
    ```
  - **포인터 노출 없음**: `TextureInfo*`를 외부에 반환하지 않으므로 댕글링 리스크 제거
  - **용어 통일**: 문서 전반에서 ~~`isManaged`~~, ~~`findByTextureId`~~ → **`releaseTextureById`** 로 통일
  - 이중 해제 방지: `in_use` false 체크 내장

#### 1-2. sdk_api_v2.cpp 관리 텍스처 정합성
- **파일**: `cpp/src/sdk_api_v2.cpp` (line 262-275)

> 리뷰 코멘트 #2: `g_managed_textures` 개별 GL 해제 시 이중 해제 리스크.
> `gpu_beauty_backend->release()` → `TexturePool::release()` (line 62-68)에서 이미 모든 GL 텍스처를 삭제함.
> 여기에 `g_managed_textures`를 추가로 `glDeleteTextures()`하면 이미 삭제된 텍스처를 다시 삭제하게 됨.

- **현황**: `g_managed_textures` set이 SDK 생성 텍스처 추적 (line 38, 371, 427)
- **수정**: `iris_sdk_release_gpu_beauty()`에서 `g_managed_textures.clear()` **만** 호출 (set 정리만).
  `g_gpu_beauty->release()`가 TexturePool 경유로 실제 GL 리소스를 이미 해제하므로 개별 `glDeleteTextures()` 금지.
- **해제 순서**: `g_gpu_beauty->release()` (GL 리소스 해제) → `g_managed_textures.clear()` (추적 set 정리) → `g_gpu_beauty.reset()` (객체 소멸)

#### 1-3. 소유권 규약 문서화
- **파일**: 신규 `docs/TEXTURE_OWNERSHIP.md`
- **내용**:
  - C++ TexturePool 소유 텍스처: Android에서 `glDeleteTextures()` 호출 금지
  - Android 소유 텍스처 (OES, RGBA FBO, LUT 3D): C++에서 삭제 금지, ID 참조만
  - SDK 텍스처 해제 경로: `IrisLensSDK.releaseTexture()` → C API → TexturePool
  - 종료 순서: Android 리소스 → SDK 텍스처 → GPU Backend
  - **이중 해제 방지 규칙**: `release()` 호출 후 `g_managed_textures` 내 ID를 개별 삭제하지 않음

---

### Phase 2: Detection Handle 전달 및 ROI 활성화 (High)

**좌표 계약**: 본 Phase에서 다루는 ROI(`face_rect`)는 모두 **normalized (0.0~1.0) top-left 원점** 기준이다. 픽셀 좌표 변환은 소비 측(`applyTextureId`, `glScissor`)에서 수행한다.

#### 2-1. Java API Detection Handle overload 추가
- **파일**: `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java` (line 724, 746)
- **현재**: 모든 `applyBeautyFilterTextureV2()` overload에서 `detectionPtr = 0L` 하드코딩
- **수정**: `detectionHandle` 파라미터를 받는 새 overload 추가, 기존 overload(0L) 하위 호환 유지

#### 2-2. IrisResult 네이티브 핸들 관리 — 더블 버퍼 재사용 방식

> 리뷰 코멘트 #4: per-frame `new/delete` 대신 재사용(슬롯/풀) 방식 필요.

- **파일**: `IrisLensSDK.java`, `iris_jni.cpp`
- **현재**: `IrisResult`는 순수 Java POJO, 네이티브 포인터 없음
- **수정**:
  - JNI에 **더블 버퍼** 방식 채택: C++ 측 `DetectionSlot slot_[2]` 사전 할당
  - `nativeUpdateDetectionSlot(IrisResult)` → 비활성 슬롯에 덮어쓰기 후 atomic swap (heap alloc 없음)
  - `nativeGetDetectionSlotPtr()` → 현재 활성 슬롯 포인터 반환 (읽기 전용)
  - `nativeReleaseDetectionSlot()` → 종료 시 1회 해제
  - **수명주기**: SDK 초기화 시 슬롯 2개 할당 → 매 프레임 비활성 슬롯 업데이트 → SDK 해제 시 삭제
  - per-frame heap alloc/free 제거로 GC 압력 및 메모리 단편화 방지

> 리뷰 코멘트 (추가 #2): Analyzer 스레드(갱신)와 GL 스레드(소비) 동시 접근 시 경합 방지 규칙 필요.
> 리뷰 코멘트 (추가 #4): 더블 버퍼만으로는 stale read/ABA 가능. valid + generation 필요.

#### Detection Slot 동시성 계약 (필수)

Detection slot은 `valid`와 `generation`을 함께 사용한다.

- **슬롯 구조**:
  ```cpp
  struct DetectionSlot {
      IrisResult data;
      std::atomic<bool> valid{false};
      std::atomic<uint64_t> generation{0};
  };
  DetectionSlot slot_[2];
  std::atomic<int> active_index_{-1};  // -1 = 초기 미설정
  ```

- **초기 상태**: 모든 슬롯 `valid=false`, `generation=0`, `active_index_=-1`

- **Writer (Analyzer 스레드)** — JNI `nativeUpdateDetectionSlot`:
  1. 비활성 슬롯 인덱스 결정: `int write_idx = (active_index_.load(acquire) == 0) ? 1 : 0`
  2. `slot_[write_idx].data`에 `IrisResult` 복사
  3. `slot_[write_idx].generation.fetch_add(1, release)`
  4. `slot_[write_idx].valid.store(true, release)`
  5. `active_index_.store(write_idx, release)` — swap

- **Reader (GL 스레드)** — `nativeGetDetectionSlotPtr`:
  1. `int read_idx = active_index_.load(acquire)`
  2. `read_idx == -1`이면 → **fallback** (detection 없음, passthrough)
  3. `slot_[read_idx].valid.load(acquire)` 확인 → `false`이면 → **fallback**
  4. `uint64_t gen_before = slot_[read_idx].generation.load(acquire)`
  5. `slot_[read_idx].data` 소비 (읽기 전용)
  6. `uint64_t gen_after = slot_[read_idx].generation.load(acquire)`
  7. `gen_before != gen_after`이면 → **해당 프레임 폐기** (passthrough)

- **정책**: `valid=false` 또는 generation 불일치 시 해당 프레임은 뷰티 passthrough (원본 유지)
- **보장**: 락 없는 wait-free 읽기, 초기화 직후 빈 슬롯 접근 방지, ABA 문제 해소
- **대안 기각**: mutex 락은 GL 프레임 드랍 유발 가능

#### 2-3. ROI 데이터 경로 완성 (sdk_api_v2 → applyTextureId)

> 리뷰 코멘트 #3: ROI가 `sdk_api_v2`에서 계산되지만 `applyTextureId` 호출 시 전달되지 않음.
> 현재 `sdk_api_v2.cpp:319-354`에서 ROI를 계산하지만, `applyTextureId()` 호출(line 358)에는 ROI를 넘기지 않음.
> `applyTextureId` 내부(line 870-912)에서 detection→ROI를 다시 계산하지만 line 1032에서 `(void)roi_ptr`로 무시.

- **파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` (line 182-190), `cpp/src/gpu/gpu_beauty_backend.cpp`
- **현재 시그니처**: `applyTextureId(input, output, w, h, config, detection, lut_id, lut_intensity)`
- **문제**: detection에서 ROI를 내부 계산하지만 실제 적용 코드가 없음 (`(void)roi_ptr`)
- **수정 방안**:
  - `applyTextureId` 내부에서 이미 ROI 계산 로직이 있으므로 시그니처 변경 불필요
  - `(void)roi_ptr` TODO를 실제 구현으로 교체

> 리뷰 코멘트 (추가 #3): `glScissor` 단독 사용 시 직사각형 경계로 품질 한계.
> 얼굴형 경계 품질을 위한 최종 선택 기준을 미리 결정.

- **ROI 적용 방식 결정**: **2단계 전략** (성능 우선 → 점진적 품질 개선)

- **1단계 (P3 구현)**: `glScissor` 기반 직사각형 ROI
  - ROI face_rect 영역에 마진 적용 후 `glScissor()` 설정
  - 뷰티 필터를 ROI 내부에서만 실행 → GPU 처리량 감소 효과 확보
  - ROI 외부는 원본 텍스처 passthrough blit
  - 경계가 직사각형이지만, 뷰티 필터 자체가 피부색 기반이라 배경 변형 리스크 낮음

- **2단계 (P4 이후, 선택)**: soft mask 텍스처 기반
  - `BeautyROIManager::computeROI()`에서 생성된 face mesh 기반 마스크를 텍스처로 업로드
  - 최종 블렌딩 셰이더에서 `mix(original, beauty, mask)` 적용
  - 구현 시점은 P3 실기기 테스트에서 경계 아티팩트가 시각적으로 문제되는 경우에 한함

- **선택 근거**: P3 목표는 "ROI 미적용(전체 프레임) → ROI 적용(얼굴 영역)" 전환이며, 직사각형이라도 전체 프레임 대비 GPU 부하 40-60% 감소 기대. 곡면 마스크는 추가 텍스처 업로드 + 블렌딩 패스가 필요하여 P3 스코프 초과.

> 리뷰 코멘트 (추가 #5): GL(좌하단 원점)과 Android/MediaPipe(좌상단 원점) 좌표계 차이로 ROI 오적용 가능.

#### ROI → glScissor 좌표 변환 규약 (필수)

입력 ROI는 normalized top-left 기준 `(x, y, w, h)`, 프레임 크기 `(W, H)`:

```
sx = floor(x * W)
sy = floor((1.0 - (y + h)) * H)    // Y 뒤집기 핵심
sw = ceil(w * W)
sh = ceil(h * H)
```

Clamp 규칙:
```
sx = clamp(sx, 0, W - 1)
sy = clamp(sy, 0, H - 1)
sw = clamp(sw, 1, W - sx)
sh = clamp(sh, 1, H - sy)
```

주의사항:
- detector 결과가 이미 display-rotated space이면 회전 보정을 중복 적용하지 않는다.
- `applyTextureId` 내부에서 ROI를 계산할 때 (line 870-912), `detection->face_rect`의 좌표계가 normalized top-left임을 전제한다.

#### 2-4. CameraGLRenderer.kt Detection Handle 전달
- **파일**: `android/demo-app/.../CameraGLRenderer.kt`
- **수정**: `applyGpuBeautyFilter()`에서 `irisResult` → `nativeUpdateDetectionSlot()` 호출 후 슬롯 포인터 전달

---

### Phase 3: Stub 경로 정리 (Medium)

> 리뷰 코멘트 #5: Stub 정책 통일 필요. 패스스루 유지는 "성공처럼 보이면서 미적용"이라 추적 어려움.

**정책 결정**: 모든 미구현 GPU 경로는 `IRIS_SDK_ERROR_NOT_SUPPORTED` 반환으로 통일.
호출부에서 에러 코드를 확인하고 fallback 또는 skip을 결정하도록 함.

#### 3-1. CPU 버퍼 경로 `apply()` — 명시적 비활성화
- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 364-405)
- **현재**: `LOGW("CPU buffer processing not yet implemented")` + `IRIS_SDK_OK` 반환
- **수정**: `IRIS_SDK_ERROR_NOT_SUPPORTED` 반환 + 로그 `LOGW("GPUBeautyBackend::apply() CPU buffer path not supported. Use CPUBeautyBackend instead.")`
- **이유**: CPU 경로 성능이 낮아 실용성 없음, GPU 미지원 기기는 `CPUBeautyBackend` 사용

#### 3-2. `applyFaceWarp()` — 명시적 미지원 에러 반환
- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 1043-1097)
- **현재**: `LOGW("Face Warp not yet implemented, returning input texture")` + `IRIS_SDK_OK` 반환
- **기존 계획**: 패스스루 유지 → **변경**: `IRIS_SDK_ERROR_NOT_SUPPORTED` 반환
- **호출부 보호**: `sdk_api_v2.cpp:402-403`에서 이미 `slim_face/thin_chin/enlarge_eyes` 값=0 체크로 bypass → 실제 호출 도달 불가
- **이유**: "성공 반환 + 미적용"은 디버깅 시 혼란 유발. 명시적 에러로 추적 용이. P4에서 구현 시 에러 반환 제거.

---

### Phase 4: ISS-004 홍채 반경 불일치 수정 (High)

**관련 문서**: `docs/workPaper/ISS-004_lens_iris_radius_mismatch_report.md`

**문제**: 디버그 화면에서 홍채 랜드마크(보라색 점)는 홍채 내부에 위치하지만, 렌즈 반경 디버그 원(녹색)이 눈 양 끝에 가까운 영역을 덮어 과대 렌더링.

#### 4-1. Fix-A (P0): 디버그 표기 분리
- **파일**: `android/demo-app/.../OverlayView.kt`
- **수정**:
  - `rawIrisPaint` 추가 (파란색 점선, `#4488FF`, `DashPathEffect`)
  - `drawIrisMarker()` — 원 2개 분리: raw(파란 점선) + effective(녹색 실선)
  - 디버그 텍스트에 `rawR`, `effR` 분리 출력

#### 4-2. Fix-B (P1): GPU 반경 정규화 좌표계 수정
- **파일**: `android/demo-app/.../CameraGLRenderer.kt`
- **수정**: `normalizedRadius = radius / detW` → `radius / detH`
- **근거**: 셰이더가 `adjustedCoord = vec2(texCoord.x * aspectRatio, texCoord.y)` 사용 → isotropic height 단위 공간이므로 detH로 정규화해야 함
- **효과**: Portrait(1080x1920) 기준 ~1.78x 과대 렌더링 해소

#### 4-3. Fix-C (P2): V2 홍채 보정 로직 개선
- **파일**: `cpp/src/mediapipe_detector.cpp`
- **수정**:
  - `validateAndFixIrisCoordinates()` snap-to-center → lerp 보간으로 변경
  - 임계값: 고정 `0.05` → 눈폭의 50% 비례로 동적 조정
  - 시선 추적 정보 보존

---

## 수정 대상 파일 요약

| 파일 | Phase | 변경 유형 |
|------|-------|-----------|
| `android/demo-app/.../CameraGLRenderer.kt` | 0, 2, 4 | 좌표계 통일, Detection handle 전달, 반경 정규화 detW→detH |
| `android/demo-app/.../OverlayView.kt` | 0, 4 | 매핑 정책 GL 통일, 디버그 원 분리(raw/effective) |
| `android/demo-app/.../GpuRenderActivity.kt` | 0 | IrisResult 불변 스냅샷 |
| `cpp/src/mediapipe_detector.cpp` | 4 | V2 홍채 보정 snap→lerp, 동적 임계값 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | 1, 2, 3 | releaseTexture 안전성, ROI 마스킹 구현, stub 에러 반환 |
| `cpp/include/iris_sdk/gpu/texture_pool.h` | 1 | `releaseTextureById()` 선언 |
| `cpp/src/gpu/texture_pool.cpp` | 1 | `releaseTextureById()` 구현 |
| `cpp/src/sdk_api_v2.cpp` | 1 | 관리 텍스처 해제 순서 수정 (clear only) |
| `android/iris-sdk/.../IrisLensSDK.java` | 2 | Detection handle overload |
| `android/iris-sdk/src/main/cpp/iris_jni.cpp` | 2 | Detection 슬롯 JNI (update/get/release) |
| `docs/TEXTURE_OWNERSHIP.md` | 1 | 소유권 규약 문서 (신규) |

---

## 검증 방법

1. **좌표계 정렬 (Phase 0)**:
   - 회전 0°/90°/270° + 전면 미러 조합별 렌즈 원형도(가로/세로 비율) 확인
   - Mesh 디버그 점이 얼굴 실루엣에 정확히 대응하는지 시각 확인
   - Overlay 마커와 GL 렌더 좌표가 일치하는지 확인
   - `resolveCoordinateSpace()` 유닛 테스트: 각 회전값별 기대 width/height 검증

2. **텍스처 소유권 (Phase 1)**:
   - `release()` 호출 후 GL 에러 로그 없음 확인
   - **이중 해제 테스트**: `release()` 2회 연속 호출 시 크래시 없음
   - 반복 초기화/해제 사이클 메모리 누수 없음 확인 (Android Profiler)
   - C++ 빌드: `cd cpp/cmake-build-debug && cmake --build . --parallel && ctest`

3. **ROI 연동 (Phase 2)**:
   - Detection 슬롯 업데이트 시 ROI 생성 로그 확인
   - ROI 전후 GPU 시간 비교 (FPS 표시 활용)
   - 얼굴 영역만 뷰티 적용, 배경 원본 유지 시각 확인
   - 슬롯 재사용 시 메모리 증가 없음 확인 (heap alloc 0회/frame)
   - **Scissor 시각 검증** (리뷰 코멘트 추가 #6):
     - 회전 `0°/90°/270°` + 전면 미러 조합별 ROI 디버그 박스 렌더링
     - Pass 기준:
       - ROI 박스가 얼굴 중심 영역에서 벗어나지 않음
       - 배경 영역에 뷰티가 적용되지 않음
       - 렌즈 원형도(가로/세로 비율) 0.95~1.05 유지
     - 구현: `glScissor` 적용 후 디버그 모드에서 scissor rect를 빨간 와이어프레임으로 표시

4. **Stub 정리 (Phase 3)**:
   - CPU `apply()` → `IRIS_SDK_ERROR_NOT_SUPPORTED` 반환 확인
   - `applyFaceWarp()` 직접 호출 → `IRIS_SDK_ERROR_NOT_SUPPORTED` 반환 확인
   - `applyFaceWarp(slim_face=0, thin_chin=0, enlarge_eyes=0)` → sdk_api_v2에서 bypass (호출 도달 안 함) 확인

5. **실기기 통합 테스트**:
   - Android 빌드 → 실기기 → 뷰티+LUT+렌즈 동시 활성화 → 크래시 없음
   - **회전 회귀**: 0°/90°/270° 회전 + 전면 미러 + 렌즈 원형도 확인

---

## 리뷰 코멘트 반영 이력

| # | 심각도 | 코멘트 요약 | 반영 내용 |
|---|--------|------------|-----------|
| 1 | Critical | ISS-003 좌표계 정렬 누락 | Phase 0 신설 (0-1, 0-2, 0-3) |
| 2 | High | Phase 1-2 이중 해제 리스크 | `g_managed_textures.clear()` only, 개별 GL 삭제 금지로 수정 |
| 3 | High | ROI 인터페이스 데이터 경로 미완성 | Phase 2-3 신설: `applyTextureId` 내부 ROI 적용 구현 계획 상세화 |
| 4 | Medium | per-frame Detection handle alloc 오버헤드 | Phase 2-2: 슬롯 재사용 방식으로 변경 |
| 5 | Medium | Stub 정책 불일치 (패스스루 vs 에러) | Phase 3: 모든 미구현 경로 `IRIS_SDK_ERROR_NOT_SUPPORTED` 통일 |
| 6 | Medium | 검증에 회전/미러/원형도 테스트 누락 | 검증 방법 1, 5에 회전/미러/원형도 항목 추가 |
| 7 | Medium | TexturePool `isManaged` API 구체화 부족 | Phase 1-1: `releaseTextureById()` 원자화 API로 재설계 |
| 8 | Medium | Detection 슬롯 스레드 경합 규칙 누락 | Phase 2-2: valid + generation 동시성 계약으로 강화 |
| 9 | Medium | ROI `glScissor` 품질 한계 판단 기준 부재 | Phase 2-3: 2단계 전략 (P3: glScissor, P4: soft mask) 및 선택 근거 명시 |
| 10 | High | Detection Slot stale read/ABA 방지 | Phase 2-2: valid + generation 프로토콜, 초기 빈 슬롯 방어 |
| 11 | High | glScissor Y 좌표 변환 수식 누락 | Phase 2-3: normalized top-left → GL bottom-left 변환 규약 명시 |
| 12 | Medium | Scissor 시각 검증 항목 누락 | 검증 방법 3에 ROI 디버그 박스 렌더링 + pass 기준 추가 |
| 13 | Medium | TexturePool API 용어 불일치 | `isManaged`/`findByTextureId` → `releaseTextureById` 통일, 파일 표 반영 |
| 14 | Medium | `findByTextureId` 포인터 생존성 리스크 | `releaseTextureById`로 조회+반환 원자화, 외부 포인터 노출 제거 |

## 이슈 및 메모
- CI 테스트 자동화는 2/3 AI만 언급, 본 작업 범위에서 제외 (후속 작업)
- LUT 관련 코드는 `54b6f50`에서 C++ 파이프라인에 통합 완료
- ISS-003 상세 원인 분석은 별도 문서 참조: `docs/workPaper/ISS-003_gpu_lens_mesh_coordinate_alignment_plan.md`

## 변경 이력
| 날짜 | 변경 내용 |
|------|-----------|
| 2026-02-11 | 작업 문서 초안 작성 (Claude, Gemini, Codex 합의 기반) |
| 2026-02-11 | 리뷰 코멘트 6건 반영 (Phase 0 신설, 이중 해제 수정, ROI 경로 상세화, 슬롯 방식, Stub 정책 통일, 검증 보강) |
| 2026-02-11 | 리뷰 코멘트 3건 추가 반영 (releaseTextureById API 구체화, 더블 버퍼 스레드 정책, ROI 2단계 전략) |
| 2026-02-11 | 리뷰 코멘트 5건 추가 반영 (valid+generation 동시성 계약, glScissor Y변환 수식, Scissor 시각 검증, releaseTextureById 원자화, 포인터 노출 제거) |
| 2026-02-11 | 전체 구현 완료: Phase 0~3 모두 구현, C++ 빌드 검증 완료, TEXTURE_OWNERSHIP.md 작성 |
| 2026-02-11 | Phase 4 추가: ISS-004 Fix-A/B/C 구현 (디버그 원 분리, GPU 반경 정규화 수정, V2 보정 lerp 변경) |
