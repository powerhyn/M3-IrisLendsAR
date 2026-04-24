# P6-W3 브레인스토밍 — Codex 의견

대상: `docs/workPaper/P6-W3_env_reflection_scaffold.md` §6.1 ~ §6.7

## 6.1 Fresnel 수식 확정

- 추천: **옵션 A (Schlick)**.
- 근거: W3 반사 계층의 핵심은 `dot(N, V)` 기반의 시점 의존 cue라서 거리 기반 가짜값보다 Schlick이 목적에 맞고 비용도 충분히 싸다. **옵션 C는 D1 재도입은 아니지만**, 시점 의존성을 버린 단순 게이트라 Fresnel 역할로는 약하다.

## 6.2 분석 노멀 계산의 W3 포함 여부

- 추천: **포함**. 단 `reflect(-V, N)`와 `dot(N, V)` 입력으로만 쓰고, D1의 고정 light/diffuse/spec 계산은 재도입하지 않는다.
- 근거: D1의 문제는 분석 노멀 자체가 아니라 eye-local 고정 조명 블록이었다. 반사 방향과 Fresnel 계산용 최소 노멀은 geometry 입력이라 D1 복귀로 볼 필요가 없다.

## 6.3 renderMask hook 활성 방식

- 추천: **`#ifdef`**.
- 근거: `renderMask hook`은 W8 조건부 트랙용 준비 코드라 기본 바이너리에서 완전히 비활성인 편이 맞다. 현재 동작을 `finalAlpha`와 동일하게 고정하고 추가 런타임 분기를 만들지 않는 것이 가장 안전하다.

## 6.4 sampleReflection 함수 추상화 방식

- 추천: **방식 A**. `sampleReflection()` 내부에서 `uSourceType` uniform으로 OFF / env-map / periphery를 스위치한다.
- 근거: W4 B2는 단일 셰이더에서 런타임 토글로 3 프로토타입을 비교해야 하므로 variant 분리보다 uniform 스위치가 맞다. W3에서는 no-op 기본 구현만 두고 W4에서 실제 소스만 꽂으면 된다.

## 6.5 reflectUV 계산 수식

- 추천: **옵션 B**. `reflect(-viewDir, normal)`로 `reflectDir`를 만든 뒤 `xy`를 UV로 매핑한다.
- 근거: Fresnel을 Schlick로 가면 반사 방향도 같은 `N/V` 관계를 쓰는 편이 일관적이다. iris local 좌표만 쓰는 옵션 C는 구현은 쉽지만 env reflection의 방향성을 너무 많이 잃는다.

## 6.6 env_map 에셋 크기 및 포맷

- 추천: **99 기준 그대로 `256×128` LDR 에셋으로 시작하고, 위치는 `android/demo-app/src/main/assets/env/`로 둔다**.
- 근거: W3/W4 단계는 실험과 교체 속도가 우선이라 SDK 내장보다 demo asset이 다루기 쉽다. 포맷도 벤치 전용 스캐폴드라 우선은 가장 단순한 LDR로 충분하다.

## 6.7 B2 벤치 매트릭스 draft

- 추천: **W3에서 draft를 미리 확정한다**. `4환경 × 2동작 × 3프로토타입` 골격은 99/ W4 문서와 동일하게 간다.
- 근거: W4는 구조 논의보다 촬영과 평가가 본체라, W3에서 매트릭스를 닫아 둬야 바로 프로토타입 구현과 캡처로 들어갈 수 있다. 반사 소스 비교가 목적이므로 매트릭스 변수를 더 늘릴 필요도 없다.
