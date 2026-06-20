# 골든 베이스라인 재캡처 manifest — ④ W4-D

> 무언의 재기준선(silent re-baseline) 차단용 강제 기록 (Codex HIGH, ADR §12 / REFACTOR-4_plan §3).
> 이 manifest 없이 `cpp/tests/golden/baseline/`를 변경하는 PR은 거부한다.

## 재캡처 사유

W4-D에서 자체 추적 코어(mediapipe_detector/inference_thread/iris_detector + detect C API)를 물리 제거했다.
기존 baseline은 **detector 경로**(`iris_sdk_detect_with_rotation`)로 캡처돼 detector 전용 메타 필드를 포함했는데,
detector가 사라져 detect-mode 캡처가 불가능해졌다. 따라서 baseline을 **injection 경로**
(`iris_set_landmarks` → `iris_get_injected_result` → `deriveIrisResult`)로 재캡처했다.

**핵심: 골든은 코어(지오메트리+이펙트)를 "고정 478 랜드마크 입력"에 대해 결정적으로 검증한다.**
재캡처는 라이브 추적기(MediaPipe Tasks)로 새로 검출한 것이 **아니라**, 기존 baseline에 동결돼 있던
**구(舊) detector의 478 랜드마크를 그대로 재주입**한 것이다. 즉 입력 랜드마크는 불변(frozen)이고,
산출 경로만 detector→injection으로 바뀌었다. 라이브 추적기(Tasks) 품질은 골든이 아니라 **실기기 A/B**로 검증한다.

## 메타데이터

| 항목 | 값 |
|---|---|
| 재캡처 일자 | 2026-06-18 |
| 승인자 (gate-0) | 사용자 — '전환 확정(TASKS로 전환)' 선언 + W4-D 착수 승인 (2026-06-17, plan §1) |
| 산출 경로 | injection (`iris_set_landmarks`/`iris_get_injected_result`/`deriveIrisResult`) — detector 제거됨 |
| 랜드마크 소스 | 구 detector(mediapipe_detector) 478점, 기존 baseline JSON에 동결된 값 재주입 (불변) |
| 재캡처 도구 | `golden_capture --inject-from <baseline.json>` + `scripts/golden_capture_all.sh` (`INJECT_BASELINE` 모드) |
| 입력 corpus | 3장(face_closeup·screenshot_640a·screenshot_720a) × {rot0/90/180/270, mirror, gamma05, mirror_rot270, lensmirror} = JSON 18 + PNG 19 |
| 회전 치수 처리 | rot90/270은 upright 치수 W↔H 스왑 주입 (input_width↔height) — 미스왑 시 radius 0.04~0.34px 오차 (Codex 교차검증 실측) |
| 라이브 추적기 (런타임, 골든 무관) | MediaPipe Tasks tasks-vision:0.10.35 |
| 모델 SHA256 (런타임 추적, 골든 무관) | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` (face_landmarker.task) |
| 캡처 플랫폼 | macOS 데스크톱, CPU 렌더 (golden_capture `use_gpu=0`), 결정적 |

## before(구 detector baseline) → after(injection baseline) diff

geometry는 **불변(동일 랜드마크)**, detector 전용 필드만 변경:

| 필드 | before (detector) | after (injection) | 사유 |
|---|---|---|---|
| left/right_iris·radius·face_rect·face_mesh | — | **ε-동일** | 동일 478점, deriveIrisResult가 detector와 동일 수식 (test_golden_injection_derive 18/18 + 음성대조로 입증, radius Δ ~2e-4px) |
| `iris_quality_left/right` | 실측(대부분 0 — eye_refiner 비활성) | **필드 제거** | detector 전용 메타, W4-D 물리삭제 |
| `eye_refiner_used` | false (활성 불가 결함) | **필드 제거** | detector 전용 메타, W4-D 물리삭제 |
| `confidence` | 0.66~0.94 (detector 측정) | **1.0** | injection은 detected?1.0:0.0 (W4-B1 게이트 통과 상수) |
| `avg_iris_luma_left/right` | 실측(0.08~0.56) | **-1.0** | injection은 RGB 미보유 → 미측정 sentinel. **런타임 TASKS는 TasksToIrisResult.fillIrisLuma로 실측 유지**(골든만 -1, GPU 전용 소비라 렌더 무영향) |
| `frame_width`/`frame_height` | 없음 | **추가** | upright 치수 명시(회전 치수 모호성 제거) |
| render.png / beauty.png | — | **시각 ε-동일** | 픽셀 max Δ 4/255(AA 엣지), meanΔ≈0, >0.5px 0%. byte는 다름(sub-pixel AA가 압축 PNG 바이트 증폭) |

## invariant (불변식 — 회귀 아님 판정 근거)

- **geometry(478점·홍채중심·radius·face_rect) 불변**: test_golden_injection_derive 18골든 parametrized + 음성대조(MisswappedDimsDiverge) 통과.
- **render/beauty 시각 불변**: CPU 픽셀 max Δ 4/255 (AA 엣지 극소수), >0.5px 0%.
- **avg_iris_luma 런타임 보존**: 골든(injection)만 -1, Android TASKS 런타임은 fillIrisLuma+stabilizer로 실측 유지 (별도 실기기 검증).
- **timestamp**: 골든은 ts=0 주입(결정적).

## 검증 (재캡처 후)

- `test_golden_injection_derive` 19/19 PASS (코어 제거 후에도 — injection 경로 무손상).
- 골든 게이트 idempotency: 새 baseline 재주입 캡처 vs baseline → `golden_compare.py` **PASS, 불일치 0** (injection vs injection = byte-identical).
- 데스크톱 빌드 GREEN, 코어 .a TFLite-free, ctest 회귀 0(pre-existing 3), assembleDebug exit 0.

## 재생성 방법 (향후)

```bash
# detector 제거됨 — detect-mode 캡처 불가. injection 모드만 가능:
INJECT_BASELINE="$PWD/cpp/tests/golden/baseline" \
  bash scripts/golden_capture_all.sh /tmp/recap   # 동결 랜드마크 재주입
python3 scripts/golden_compare.py --baseline cpp/tests/golden/baseline --candidate /tmp/recap  # PASS여야
```

## 이월 (이 재캡처와 별개)

- **좌표 canonical left/right relabeling (ADR §7.3)**: 출력(좌표 의미) 변경이라 별도 슬라이스 + 별도 재캡처. 이번 재캡처는 **현 라벨(detector와 동일계) 유지** — relabeling 미포함.
- `cpp/third_party/tflite/` (32MB, git 미추적): CMake 참조 0(W4-D 제거). 물리 삭제는 선택적 정리.
