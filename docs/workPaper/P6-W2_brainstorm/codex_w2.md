# P6-W2 브레인스토밍 — Codex 의견

대상: `docs/workPaper/P6-W2_blend_canonical.md` §6.1 ~ §6.7

## 6.1 ColorReplaceLinear 수식 구현 시점

- 추천: **옵션 A**. W2에서는 `blendColorReplaceLinear` **함수만 정의**하고, ID 7 정식 분기 등록은 W5 B1 결과 후로 미룬다.
- 근거: B1은 `Normal vs CRL` 채택 여부를 닫는 벤치라, 그 전에 ID 7을 정식 활성하면 결론 선반영처럼 보인다. W5 비교는 벤치용 토글/임시 활성 경로로 충분하다.

## 6.2 ID 3/4/6 빈 슬롯 유지 vs 재번호화

- 추천: **옵션 A (유지)**.
- 근거: 재번호화는 호출자와 UI까지 같이 흔들지만 W5 B1 판정에는 아무 이득이 없다. 빈 슬롯 + fallback이 호환성과 작업 범위 둘 다 더 안전하다.

## 6.3 fallback default를 Normal vs TintLinearV2

- 추천: **옵션 A (Normal fallback 유지)**.
- 근거: fallback은 "기본 모드"가 아니라 무효 ID에 대한 안전장치이므로 가장 단순한 baseline에 붙여 두는 편이 낫다. W5 B1이 끝나기 전까지는 invalid ID가 TintLinearV2로 조용히 흡수되는 것보다 Normal로 명확히 떨어지는 편이 해석이 쉽다.

## 6.4 LUMA_COEFFS 정합성

- 추천: **Rec.709 linear 계수 `vec3(0.2126, 0.7152, 0.0722)`로 전 경로 통일**.
- 근거: TintLinearV2, `avg_iris_luma`, CRL이 서로 다른 luma 기준을 쓰면 W5 B1에서 수식 차이와 계수 차이가 섞여 버린다. W2에서 하나로 고정해 두는 것이 맞다.

## 6.5 realSpec 복원 경로 (조건부)

- 추천: **완전 삭제**.
- 근거: 조건부 유지 여부는 문서와 Git history로 관리하면 되고, 셰이더 안에 주석 상태의 dead code를 남기는 것은 냄새다. W4 B2/W5 비교도 현재 파이프라인이 단일 진실원으로 남아야 판정이 깔끔하다.

## 6.6 기본값 블렌드 모드 확정

- 추천: **TintLinearV2를 기본값으로 확정**.
- 근거: 확정된 3종 중 기본값으로 합의된 것은 TintLinearV2이고, Normal은 W5 B1의 비교 대상이지 canonical default가 아니다. 기본값을 먼저 고정해야 W5 캡처도 같은 baseline에서 비교된다.

## 6.7 Android Demo UI 변화

- 추천: **W2에서는 UI 정리하지 말고 W9로 이관**.
- 근거: 이번 W의 핵심은 셰이더 canonicalization이지 드롭다운 정리가 아니다. W5 B1도 전체 UI 개편 없이 필요한 모드 선택만 가능하면 충분하다.
