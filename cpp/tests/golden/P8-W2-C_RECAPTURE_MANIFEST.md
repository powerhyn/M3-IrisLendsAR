# 골든 베이스라인 재캡처 manifest — P8-W2-C (CPU 색보정 곁가지 제거)

> 무언의 재기준선(silent re-baseline) 차단용 강제 기록 (ADR §12 / REFACTOR-4_plan §3).
> 이 manifest 없이 `cpp/tests/golden/baseline/`의 beauty.png를 변경하는 PR은 거부한다.

## 재캡처 사유

P8-W2-C에서 CPU 백엔드(`cpu_beauty_backend.cpp`)의 곁가지 색보정/스무딩 EFFECT를 제거했다:
- 제거: `applyWhitening`(2 오버로드), `applyColorBalance`, `applySoftFocus`(+V2), `applySkinSmoothing`(Bilateral, 2 오버로드, +V2), `applyWrinkleRemoval` + orphan 헬퍼(`overlayBlend`/`detectSkinTone`/`createWrinkleRegionMasks`).
- 보존: `applyBrightness`(생존 — 곁가지 아님). `applyFullFrame`/`applyWithROI`는 brightness-only로 축소.

기존 beauty.png baseline은 `golden_capture.cpp`의 `fillBeautyConfig`가 **`smoothing=0.5`(Bilateral) + `skin_quality=0.5`** 로 캡처했는데, 그 효과(Bilateral 스무딩)가 제거됐다. 곁가지 제거 후 CPU beauty의 유일 생존 효과는 **brightness**이므로, `fillBeautyConfig`를 `brightness=1.2`(곁가지 0)로 변경하고 재캡처해 골든이 생존 경로(ROI brightness)를 실측하도록 했다.

**핵심: 골든은 코어(CPU beauty 이펙트)를 "고정 478 랜드마크 입력"에 대해 결정적으로 검증한다.** 재캡처는 라이브 추적이 아니라 기존 baseline에 동결된 478 랜드마크를 그대로 재주입(`INJECT_BASELINE`)했다. 입력 랜드마크·ROI 산출은 불변이고, **CPU beauty 효과 config만** smoothing→brightness로 바뀌었다.

## 메타데이터

| 항목 | 값 |
|---|---|
| 재캡처 일자 | 2026-06-22 |
| 승인자 | 사용자 — P8-W2 곁가지 제거 착수(2026-06-20) + W2-C 진행 지시(2026-06-22) |
| 단계 | P8-W2-C (브랜치 `feature/p8-beauty`) |
| 산출 경로 | injection (`scripts/golden_capture_all.sh` + `INJECT_BASELINE`), CPU beauty = `iris_sdk_apply_beauty_v2_c` |
| 랜드마크 소스 | 기존 baseline JSON의 동결 478점 재주입 (불변) |
| config 변경 | `fillBeautyConfig`: `smoothing=0.5`·`skin_quality=0.5` 제거 → `brightness=1.2`. enabled=1/use_gpu=0/intensity=0.7/roiOnly=1(default) 유지 |

## before / after

| | before (smoothing=0.5) | after (brightness=1.2) |
|---|---|---|
| 효과 | Bilateral 스무딩 (ROI) | brightness ×1.2 (ROI) |
| face_closeup__rot0 전체 mean | 124.5 (=입력, 스무딩은 평균 보존) | 125.2 |
| ROI 변경 영역 | (텍스처 변화, 평균 무변) | 5.2% 픽셀, 변경 영역 mean diff 17.3, max 36 (×1.2 정합) |

## 불변식 증명 (격리)

재캡처(`/tmp/w2c_new`)를 기존 baseline과 파일별 `cmp`:
- **변경: beauty.png 4벌만** — `face_closeup__gamma05`, `face_closeup__rot0`, `screenshot_640a__rot0`, `screenshot_720a__rot0`.
- **byte-identical 33벌**: JSON 18 + render.png 15 (CPU 색보정 제거가 렌더/지오메트리에 무영향 입증).
- 파일 집합 동일(추가/삭제 0), 재캡처 스크립트 산출물 검증 PASS(37/37, KNOWN_RENDER_FAILURES 2건 = pre-existing 허용).
- brightness 정량 검증: 새 beauty.png ROI가 입력 대비 ×1.2 밝음(변경 영역 mean +17.3).

## 게이트

- 데스크톱 빌드: exit 0, C 변경 파일 신규 경고 0.
- ctest: 582개 중 581 통과, 유일 실패 `GPUBeautyBackendTest.FailsWithNullContext`(pre-existing, 곁가지 무관) — 신규 회귀 0.
- beauty.png는 ctest 자동 비교 대상 아님(`test_golden_injection_derive`는 `.result.json`만 비교) → 본 manifest가 수동 baseline 변경의 강제 기록.
</content>
