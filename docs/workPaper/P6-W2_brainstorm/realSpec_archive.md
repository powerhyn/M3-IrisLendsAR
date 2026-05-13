# realSpec 수식 아카이브 (P6-W2 §5.11)

> **상태**: P5-W3-05 S1 D3에서 코드 삭제, P6-W2 §5.11에서 완전 폐기 확정.
> **목적**: W4 B2 벤치(환경 반사 분리) 실패 시 복원 경로 (Git history 외 이중 보존).

## 원본 수식 (LTL 내부 부착 형태)

```glsl
// 삭제 시점: P5-W3-05 S1 D3 (커밋 9aee86d)
// 부착 위치: blendLuminanceTintLinear() 내부, mix(baseL, tinted, opacity) 직후

float realSpec = smoothstep(0.7, 0.95, lum);
result = mix(result, baseL, realSpec);
```

## 기능 의도 (당시)

- 카메라 base의 lum이 높은 영역(밝은 픽셀)에서 tinted 결과를 baseL로 되돌려 specular(반사) 보호.
- "밝은 픽셀 보호 = 실반사 보호" 가정.

## 폐기 사유

1. **개념 오류**: "밝은 픽셀"과 "실반사"는 동치가 아님. 흰자위, 피부 하이라이트, 환경광 모두 밝지만 반사가 아님.
2. **계층 분리 부재**: 블렌드 함수 내부에 specular 보호를 묶으면 블렌드 수식 비교(W5 B1)가 specular 부작용까지 섞여 판정 불가.
3. **W4 B2 (환경 반사 벤치)에서 별도 가산 계층으로 처리** — env-map 또는 periphery 프로토타입.

## 복원 경로

- **Primary**: Git history.
  - `git show 9aee86d:cpp/src/gpu/shader_sources.cpp` (삭제 직전)
  - `git log -p --all -- cpp/src/gpu/shader_sources.cpp | grep -A 3 "realSpec"`
- **Secondary**: 이 아카이브.

## W4 B2 실패 시 복원 절차

1. W4 B2 결과 분석 → "환경 반사 분리 계층 모두 실패" 확정.
2. realSpec를 LTL 내부에 다시 넣을 가치가 있는지 재평가 (개념 오류 인지 후에도 시각 효과 우수 시).
3. 복원하되 **Linear 공간**에서 `lum`은 `dot(baseL, LUMA_709_LENS)` 사용 (Rec.601 → Rec.709 통일).

## 참조

- 99_final_decision.md §1.1 D3 (realSpec 폐기 결정)
- P6-W2 §5.11 (완전 삭제 확정)
- P6-W4_env_reflection_bench.md (대체 계층 벤치)
