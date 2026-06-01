# P6-W1 브레인스토밍 — Codex 의견

대상: `docs/workPaper/P6-W1_eye_render_packet.md` §6.1 ~ §6.7

## 6.1 측정 경로: CPU 1회 vs shader 내

- 추천: **옵션 A (CPU 1회, 1프레임 지연 허용)**. **Codex R2/R3 재확인**: R2는 `masked ROI 평균 + 1프레임 지연 허용`을 명시했고, R3도 중심 1샘플이 아니라 ROI 평균이어야 한다는 점을 다시 못 박았다.
- 근거: `avg_iris_luma`는 실시간 제어 신호라기보다 material 기준값이라 0-latency보다 single-pass 유지와 안정성이 우선이다. shader 내 다중 fetch는 매 프레임 비용만 늘고, compute는 이미 기각된 경로다.

## 6.2 ROI 마스크 구체: `inner iris r<0.65 AND eyelidMask`

- 추천: **`inner iris r<0.65 AND eyelidMask`로 간다**. `eyelidMask`는 packet의 `ellipse_*`/aperture 정보에서 파생하고, W1에서는 luminance outlier drop까지는 넣지 않는다.
- 근거: R3가 문제 삼은 핵심은 중심 1샘플 오류였지 ROI 평균 자체가 아니었다. 반경을 더 줄이면 pupil 혼입 감소보다 샘플 수 감소로 인한 분산 증가가 먼저 생긴다.

## 6.3 Fallback 중립 상수 값

- 추천: **옵션 γ `0.3`**.
- 근거: 장기 hold 뒤 최후 fallback의 실패 모드는 과보정보다 약보정이 안전하다. `0.35`는 prior 성격이 남고, `0.5`는 중립이라기보다 밝기 과상향 위험이 크다.

## 6.4 어댑터 레이어 위치

- 추천: **옵션 1 `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}`**.
- 근거: EyeRenderPacket은 GPU 렌더러 전용 계약이므로 `gpu/` 아래에 두는 것이 책임 경계가 가장 명확하다. `gpu_lens_renderer.cpp` 내부 static 함수보다 테스트와 추적도 쉽다.

## 6.5 시간적 스무딩 중복 체크

- 추천: **`avg_iris_luma`에는 EMA 적용**. **Codex R2/R3 재확인**: 좌표/반경 hold는 상위 `TemporalStabilizer`, 렌더러는 `avg_iris_luma` 같은 material-only temporal envelope만 담당한다.
- 근거: blink/occlusion 시 ROI 면적이 바뀌면 measured luma가 순간적으로 튀어 tint flicker가 생길 수 있다. R3가 문제 삼은 것은 임의 EMA 계수 확정이지, material EMA 원칙 자체는 아니다.

## 6.6 `head_pose_yaw_roll` / `reflection_dir` optional

- 추천: **optional로 스키마에만 추가하고 W1 렌더러에서는 무시**.
- 근거: W4 env reflection/rotation 쪽 확장 포인트는 지금 확보해 두는 편이 맞지만, 현재 활용 경로가 없으니 W1 코드 경로에 의미 없는 분기를 넣지 않는 것이 낫다.

## 6.7 `render_confidence` optional 설계

- 추천: **optional은 유지하되 W1에서는 기본 미채움으로 두고, 추후 필요 시 `TemporalStabilizer` 출력에서만 공급**한다.
- 근거: 현재 구현에는 `getConfidence()` 같은 전용 API가 없고, `TemporalStabilizer`가 외부로 명시적으로 내보내는 안정화 신호는 `visibility`다. W1에서 새 confidence 의미를 만들면 `visibility`와 역할이 겹친다.
