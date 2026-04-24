# P6-W2 브레인스토밍 R1 — Claude 응답

**모델:** claude-opus-4-7 (1M)
**라운드:** R1
**대상:** `docs/workPaper/P6-W2_blend_canonical.md` §6.1 ~ §6.7
**특수:** 숨김 파일(.claude_w2.md)로 저장 → Codex/Gemini 응답 완료 후 `claude_w2.md`로 rename (편향 방지).

---

## 6.1 ColorReplaceLinear 구현 시점

**추천: 옵션 B (W2에서 분기도 활성, 함수도 정의)**

근거:
- (1) W5 B1 벤치가 CRL을 테스트 대상으로 포함하는데 W2에서 함수와 분기 모두 미완성이면 W5에서 신규 구현 부담이 발생. W2 범위 내에서 정리하는 게 효율적.
- (2) B1 결과가 "CRL 제거" 판정이면 W5 단계에서 함수·분기 모두 제거. 롤백 비용은 소규모(shader 문자열·ID 상수 몇 줄) → 감수 가능.

## 6.2 ID 3/4/6 빈 슬롯

**추천: 옵션 A (유지 + blendNormal 또는 fallback default로 분기)**

근거:
- (1) Android demo/기존 클라이언트가 ID 3/4/6을 전송할 가능성 잔존. 재번호화는 호출자 전체 수정 유발하여 P6 범위를 넘어감.
- (2) 빈 ID 분기의 동작은 §6.3 fallback default 결정에 묶임 (blendNormal 또는 TintLinearV2 중 하나로 자동 fallback).

## 6.3 Fallback default

**추천: 옵션 B (TintLinearV2 fallback)**

근거:
- (1) TintLinearV2가 "기본 모드" 위치(99 §1.1 D6)인데 "빈 ID = Normal"은 의미적으로 이상. "빈 ID = 기본 렌더링(TintLinearV2)"가 일관됨.
- (2) blendNormal 자체는 B1 벤치(W5)에서 "유지/제거" 재평가 대상. fallback을 B1 대기 상태인 Normal에 묶으면 B1 결과에 따라 fallback도 깨짐. TintLinearV2는 확정 모드이므로 안정적.

## 6.4 LUMA_COEFFS 정합성

**추천: `LUMA_709` 매크로(또는 `kLuma709` 상수)로 shader 전체 통일. W1 CPU 측정도 동일 상수 사용.**

근거:
- (1) 계수 자체는 Rec.709 표준(0.2126, 0.7152, 0.0722) — 이견 없음. 문제는 **색공간 일관성**: `avg_iris_luma`가 sRGB 평균이면 LTL 정규화에서 non-linear 왜곡 발생. W1 §5.2.1 색공간 계약에 따라 **linear space에서 계수 곱하기** 강제.
- (2) W2 범위 내 확인: `shader_sources.cpp`에 `LUMA_709` 같은 공통 매크로가 있으면 재사용, 없으면 정의. CPU 측정(W1) 쪽도 동일 상수 참조. 테스트 케이스로 shader 계산 결과 ≈ CPU 계산 결과 (오차 1% 이내) 검증.

## 6.5 realSpec 복원 경로 — **가장 첨예 쟁점**

**추천: 완전 삭제 + `docs/archive/realSpec_history.md` 같은 문서에 수식 보존**

근거:
- (1) Claude 원안 "주석 유지"와 Codex 원안 "완전 삭제"의 타협. 주석 코드 냄새는 P6 이후 누적되면 전체 shader 파일 가독성 저하. Git history로도 복원 가능하지만 **수식 그 자체를 가독 가능한 형태로 문서화**해두면 실제 복원 작업이 "문서에서 긁어와 붙여넣기"로 단순해짐.
- (2) W4 B2 실패 확률(<30%) × 복원 비용(30분 이내) = 실효 비용 매우 낮음. 코드 주석을 수개월 유지하는 대가보다 합리적.
- **조건:** 삭제와 동시에 `docs/archive/realSpec_history.md`(또는 기존 archive 문서)에 수식 블록 + 관련 커밋 SHA 기록. 이게 없으면 삭제 보류.

## 6.6 기본값 블렌드 모드 확정

**추천: TintLinearV2 ID(=5)를 Android demo 초기 blendMode로 설정 + 동일 기본값을 SDK C API 문서화**

근거:
- (1) 99 §1.1 D6에 "TintLinearV2 기본값" 명시됨. Android demo가 다른 값으로 시작하면 기본값 검증 시 매번 수동 전환 필요 → 현장 테스트 노이즈.
- (2) SDK 쪽 C API도 `iris_sdk_set_blend_mode` 류 함수가 없다면 기본값이 적용되는 초기화 경로에 TintLinearV2 ID를 상수로 못박고, 해당 값을 헤더 주석 또는 `sdk_api.h`에 명시.

## 6.7 Android Demo UI — W9 이관 확정

**추천: W2 범위에서 UI 터치 금지. W9 integration 단계에서 drop-down 정리.**

근거:
- (1) 빈 ID 선택 시 fallback으로 동작(안 깨짐) → 긴급 수정 아님.
- (2) W2가 shader/C++ 범위를 넘어 Kotlin UI까지 손대면 리뷰 범위가 넓어지고 병합 위험 증가. W9 통합 테스트 시 demo UI 정리 일괄 처리가 효율적.

---

## 요약 표

| 쟁점 | Claude 추천 |
|------|-------------|
| 6.1 CRL 구현 시점 | **옵션 B (함수 + 분기 모두 W2에서 활성)** |
| 6.2 ID 3/4/6 빈 슬롯 | **옵션 A (유지, §6.3 fallback로 분기)** |
| 6.3 fallback default | **옵션 B (TintLinearV2 fallback)** |
| 6.4 LUMA_COEFFS | **Rec.709 매크로로 shader+CPU 통일, linear space 강제** |
| 6.5 realSpec | **완전 삭제 + `docs/archive/realSpec_history.md`에 수식 보존** |
| 6.6 기본값 blendMode | **TintLinearV2 ID=5, Android demo 초기값 설정 + C API 문서화** |
| 6.7 demo UI | **W9 이관 (W2 범위 밖)** |

---

## 자기비판 (편향 경계)

- 6.5는 Codex 원안(완전 삭제) 쪽에 기울어진 타협안. "Codex 편향" 경계 필요. Gemini가 "주석 유지"로 가면 2:1이 될 수 있고, 그 경우 Claude가 유지 쪽으로 돌아가야 할지 재검토.
- 6.1은 실패 시 W5 롤백 비용이 실제로 작은지 코드 확인 없이 "소규모"라 단정했음. Codex/Gemini가 "롤백 비용 큼"이라 지적하면 옵션 A로 후퇴.
- 6.4는 Claude가 실제 코드 그렙 없이 "LUMA_709 매크로 있을 것"으로 추정 — 실제 shader_sources.cpp 내용 확인 후에만 확정.
