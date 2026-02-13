# GPU 렌더링 코드 리뷰 리포트 (Codex)

- 작성일: 2026-02-12
- 범위: 워킹트리 변경분 코드 리뷰
- 대상 파일:
`android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt`,
`android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`,
`cpp/src/gpu/gpu_beauty_backend.cpp`,
`android/demo-app/src/main/java/com/irislenssdk/demo/beauty/BeautyPresetFactory.kt`
- 검증 방식: 정적 코드 리뷰 (빌드/런타임 테스트 미포함)

---

## 요약

현재 수정은 핵심 방향이 맞고, 샘플러 충돌 방지/ROI 교집합 처리/기본 ROI 정책 정리는 긍정적입니다.  
다만 LUT 상태 전환 경로의 큐잉 구조와 pending 텍스처 수명 관리에 실질적인 리스크가 남아 있습니다.

---

## 주요 발견 사항

### 1) [High] LUT 적용/해제 경로의 이중 큐잉으로 상태 적용 순서 비결정성

- 근거 코드:
`android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:551`,
`android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:575`,
`android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLView.kt:222`
- 문제 설명:
`GpuRenderActivity`에서 이미 `queueEvent { ... }` 블록 내부인데, 그 안에서 `cameraGLView.setLut3dTexture()` / `setLutEnabled()`를 호출하면서 내부에서 다시 `queueEvent`가 걸립니다.
- 영향:
빠른 프리셋 전환 시 이벤트 순서가 뒤섞일 수 있어, LUT 교체 실패/멈춤/적용 누락이 재현될 여지가 있습니다.
- 권장 조치:
GL 스레드에서 실행 중일 때는 `CameraGLRenderer`의 직접 setter를 호출하는 단일 경로로 통일하거나, `CameraGLView` API 자체를 “항상 1회 큐잉” 구조로 리팩터링합니다.

### 2) [Medium] `pendingLut3dTextureId` overwrite 시 중간 텍스처 누수 가능성

- 근거 코드:
`android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:992`,
`android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:997`,
`android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:821`
- 문제 설명:
pending 슬롯이 단일 정수라서, 업로드 전에 새 프리셋이 들어오면 이전 pending ID가 덮어써집니다.
- 영향:
기존 pending 텍스처가 delete 경로 없이 유실될 수 있어 장시간 사용 시 메모리 누수 및 불안정성 위험이 있습니다.
- 권장 조치:
pending 교체 시 이전 pending ID를 즉시 `glDeleteTextures` 하거나, garbage queue를 두고 안전 시점에 일괄 정리합니다.

### 3) [Medium] ROI 교집합 empty 시 `roiOnly` 의도와 불일치한 동작

- 근거 코드:
`cpp/src/gpu/gpu_beauty_backend.cpp:1006`,
`cpp/src/gpu/gpu_beauty_backend.cpp:1019`
- 문제 설명:
교집합이 비어 scissor가 skip되어도 필터 체인은 계속 진행되어 사실상 전체 프레임 처리로 폴백됩니다.
- 영향:
`roiOnly=true` 정책의 의미가 깨지고, 경계 프레임에서 결과가 튀는 현상이 나타날 수 있습니다.
- 권장 조치:
`roiOnly` + intersection empty이면 pass-through 또는 prefill 결과 반환으로 정책을 명시적으로 고정합니다.

### 4) [Low] LUT OFF 시 `GL_TEXTURE_3D`에 0 바인딩의 드라이버 의존 리스크

- 근거 코드:
`cpp/src/gpu/gpu_beauty_backend.cpp:756`,
`cpp/src/gpu/gpu_beauty_backend.cpp:760`
- 문제 설명:
현 수정으로 sampler unit 충돌은 해소됐지만, 일부 드라이버에서 3D sampler에 texture 0 바인딩이 불안정할 가능성은 남습니다.
- 영향:
특정 기기에서 간헐적 블랙/GL 오류 가능성.
- 권장 조치:
LUT OFF에서도 neutral 3D LUT(예: 1x1x1 identity)를 바인딩해 샘플러 경로를 완전 고정합니다.

---

## 긍정적 변경점

- `uLutTexture`를 항상 texture unit 1로 고정하도록 수정하여 sampler 충돌 리스크를 크게 줄였습니다.
- ROI scissor 계산이 교집합 기반으로 바뀌어 프레임 경계 안정성이 개선되었습니다.
- 기본 설정/프리셋에서 `roiOnly`를 전체화면 처리 정책에 맞게 정리했습니다.

---

## 우선순위 제안

1. P0: LUT 큐잉 경로 단일화(이중 queue 제거).
2. P1: pending LUT 텍스처 수명 관리(덮어쓰기 누수 제거).
3. P1: `roiOnly` + intersection empty 정책 확정 및 코드 반영.
4. P2: neutral 3D LUT fallback 도입.

