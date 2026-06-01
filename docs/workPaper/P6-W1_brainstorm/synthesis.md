# P6-W1 브레인스토밍 R1 — 종합

**작성일:** 2026-04-24
**참여:** Codex (gpt-5.4 xhigh), Gemini (gemini-3-flash), Claude (opus-4-7)
**대상 쟁점:** `docs/workPaper/P6-W1_eye_render_packet.md` §6.1 ~ §6.7 (7개)

---

## 1. 합의 현황 (3/3 동의 → 닫힘)

| 쟁점 | 확정 |
|------|------|
| **6.1** 측정 경로 | **옵션 A (CPU 1회, 1프레임 지연 허용)** |
| **6.4** 어댑터 위치 | **`cpp/src/gpu/eye_render_packet_adapter.{h,cpp}`** (옵션 1) |
| **6.5** EMA 적용 | **렌더러 레벨에서 적용. 공식: `L_t = α·measured_t + (1-α)·L_{t-1}`, α=0.3, 초기값 L_0 = fallback 상수**. TemporalStabilizer는 center/radius 담당, 렌더러는 `avg_iris_luma` material-envelope 담당 |
| **6.6** `head_pose_yaw_roll` / `reflection_dir` | **스키마만 optional로 추가, W1 렌더러는 무시**. W4 env reflection 착수 시 adapter가 채움 |

## 2. 다수결 결정 (2/3 동의)

### 6.3 Fallback 중립 상수 값 — **α=0.35 채택**
- Gemini, Claude: 0.35 (한국인 평균 홍채 luminance 근사)
- Codex: 0.3 (약보정 안전)
- **판정 근거:** 다수 + "한국인 평균 prior"이 현장 체감에 더 부합. Codex의 "prior가 남는다" 우려는 유효하나, fallback은 극히 짧은 구간(N프레임 측정 성공 전)에서만 타므로 과보정 리스크 낮음.
- **소수 의견 기록:** 실기기에서 초기 N프레임 동안 "살짝 과도한 밝기" 피드백 나오면 0.3으로 재조정 검토.

### 6.7 `render_confidence` optional 설계 — **TemporalStabilizer 재활용, `visibility` 신호로 구체화**
- Gemini, Claude: TemporalStabilizer의 기존 신뢰도 재활용
- Codex: W1에서는 미채움, `visibility`와 역할 중복 경고
- **판정 근거:** Codex의 "`visibility`와 중복" 지적이 타당 → **Codex 지적을 수용하여 "신규 confidence 필드 만들지 않고 기존 `visibility` 값을 packet에 그대로 포함"** 형태로 구체화. 새 파생 수식 발명 금지.
- **구현 요지:** `packet.render_confidence = std::optional<float>(temporal_stabilizer.visibility)` — W1 렌더러는 읽지 않음 (6.6과 동일하게 예약만).

## 3. 미결 / 실기기 이관 (3분립)

### 6.2 ROI 마스크 반경 — **초기값 r<0.60, 실기기 튜닝 레인지 [0.55, 0.65]**
- Codex: r<0.65 (R2 원값 유지)
- Claude: r<0.60 (중간값)
- Gemini: r<0.55 (+0.3~0.7 luma 클램핑)
- **3분립이며 논리만으로는 수렴 불가** → 실측 데이터 우선 (메모리 `feedback_real_data_first` 적용)
- **R1 결론:**
  1. **초기 구현은 r<0.60** (Claude 중간값).
  2. **eyelidMask 파생은 Codex 제안대로** `ellipse_*` / aperture 정보에서.
  3. **luma outlier drop은 W1 초기 구현에서 미포함**. 단, 평균 결과를 `[0.1, 0.9]` 범위로 final clamp (Gemini 제안 축소 적용, 과도한 이상치 방지 최소 안전망).
  4. **W1 구현 후 실기기 벤치에서 0.55 / 0.60 / 0.65 A/B 비교**. 시각 체감에서 tint flicker / 중앙 콩알 현상을 기준으로 재확정. 결과는 W4 정성 체감 데이터와 함께 기록.

## 4. 편향 체크

- **Codex 편향 경계:** Codex가 Claude 응답(`claude_w1.md`)을 "형식 참고용"으로 읽음 (팬 로그 확인). 실제 내용에 영향 있었는지 여부는 불명확하나, 6.3에서 Codex가 유일하게 0.3을 고수한 점은 오히려 편향 없음의 증거로 볼 수 있다.
- **Claude 편향 경계:** 6.2의 "중간값 0.60"은 안이한 타협일 수 있으나, 실기기 A/B 벤치로 최종 결정을 미뤘으므로 편향 영향 제한적.
- **3모델 공통 편향 리스크:** 6.5 EMA α=0.3은 구체 수치 검증 없이 합의됨. 실기기에서 "반응이 너무 느리다"는 피드백 있으면 0.4~0.5로 재조정 가능.

## 5. W1 문서 §5에 반영할 항목

### 새로 확정 (§5.7 ~ §5.12로 추가 제안)

- **§5.7** 측정 경로: CPU 1회 평균 계산, 1프레임 지연 허용.
- **§5.8** ROI 마스크: `dist < 0.60 * iris_radius` AND `eyelidMask` (packet `ellipse_*`/aperture 파생). 평균 결과 `[0.1, 0.9]` 최종 clamp. Outlier drop은 미포함 (W1 벤치 결과에 따라 추가 검토).
- **§5.9** Fallback 중립 상수: `L_fallback = 0.35`.
- **§5.10** EMA 공식: `L_t = 0.3·measured_t + 0.7·L_{t-1}`, 초기화 `L_0 = L_fallback`. 렌더러 내부 static/멤버 상태로 유지.
- **§5.11** 어댑터 위치: `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}`.
- **§5.12** Packet 스키마: `head_pose_yaw_roll`, `reflection_dir`, `render_confidence` 3개는 `std::optional<...>`로 예약. W1 렌더러 미활용. `render_confidence`는 신규 발명 없이 `TemporalStabilizer::visibility` 값을 그대로 전달.

### §6 미결 (W1 브레인스토밍 후 남는 것)

- **§6.2 ROI 반경 튜닝:** r<0.60 초기 → 실기기 A/B(0.55/0.60/0.65) → 최종 확정.

## 6. 다음 액션

1. 사용자 승인 후 W1 문서 §5/§6 업데이트.
2. Task #1 (W1 브레인스토밍) 완료 표시.
3. W2 브레인스토밍 착수.
4. W1 구현은 별도 스킬 `ar-lens-implement`로 이관.

## 7. R2 필요 여부

**R2 불필요**. Hard veto 없음. 3분립(6.2)은 실기기 벤치로 자연스럽게 해결. 나머지는 합의 또는 유효한 다수결.
