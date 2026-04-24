# P6-W1 브레인스토밍 R1 — Claude 응답

**모델:** claude-opus-4-7 (1M)
**라운드:** R1
**대상:** `docs/workPaper/P6-W1_eye_render_packet.md` §6.1 ~ §6.7
**원칙:** 한 모델의 독립 의견. 모더레이터 자임 금지. Codex 편향 경계 (R2/R3 "1샘플 오류" 반복 경계).

---

## 6.1 측정 경로 (CPU 1회 vs shader 내 vs compute)

**추천: 옵션 A (CPU 1회, 1프레임 지연 허용)**

근거:
- (1) 99 §1.2 C-F에서 compute shader는 이미 기각. GLES 3.1 compute 드라이버 품질이 Adreno 일부에서 불안정하여 "실기기 정성 체감" 기준 리스크. 옵션 C는 재공론화할 이유 없음.
- (2) 옵션 B의 매 프레임 9~16 fetch는 단일 블렌드 패스당 30fps 기준 총 bandwidth를 수 MB/s 추가하는데, avg_iris_luma는 **중앙값·느리게 변하는 material 속성**이라 실시간성이 낮아도 무방. 1프레임 지연은 블링크 같은 급변 상황에서만 보이는데 이는 6.5 EMA로 흡수됨.

## 6.2 ROI 마스크 구체 (r<0.65 vs r<0.55 vs +eyelid+outlier)

**추천: `r<0.60 * eyelid_aperture * outlier-drop`** (Claude 제안 0.55와 Codex R2 0.65의 중간, outlier는 채택)

근거:
- (1) r<0.65는 Codex R2 원문이지만, 실제 홍채 렌더 패스의 pupil 에지가 0.30~0.42 정도까지 퍼지기 때문에 0.65는 블링크/사광 시 pupil 혼입 위험. 0.55는 너무 공격적이어서 marginal pixel이 적어져 평균 분산 증가. **0.60이 안전 margin.**
- (2) outlier drop은 "specular highlight 1~2 픽셀"이 평균을 확 끌어올리는 현상 방지에 유효 (3σ clip으로 충분). 단, 구현 부담 낮으면 채택이고, 1차 단계에선 drop 없이도 시각 체감 거의 동일하면 skip 허용 (실기기 체감으로 결정).

## 6.3 Fallback 중립 상수

**추천: α=0.35 (Codex 원값 유지)**

근거:
- (1) fallback 경로는 측정 실패·초기 N프레임에만 타므로 "한국인 평균 홍채 luminance(~0.32~0.38)" 근사가 현장 체감에 가장 무난. 0.5(중간 회색)는 tint가 과도하게 밝아지고, 0.3은 살짝 어두운 쪽으로 기울어 현장 테스트에서 "탁함" 피드백 가능성 존재. δ(EMA 시동값)는 매력적이지만 초기 frame 없이 시작 불가능 → 이때도 α 필요하므로 중복.

## 6.4 어댑터 레이어 위치

**추천: 옵션 1 `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}`**

근거:
- (1) EyeRenderPacket은 **GPU 렌더러에 공급되는 계약**이지 검출기 출력이나 SDK 외부 API가 아님. 따라서 `gpu/` 네임스페이스가 의미적으로 맞고 향후 W4/W8의 material uniform 추가 시 어댑터에 자연스럽게 배선됨.
- (2) 옵션 3(static 함수)은 단위 테스트 어려움. 옵션 2(루트)는 GPU 외 레이어가 packet을 생성할 일이 없어 위치가 과도하게 상위.

## 6.5 시간적 스무딩 (EMA) 적용 여부 — Codex R2/R3 재확인 필요

**추천: 적용. 계수 α=0.3 (렌더러 레이어에서 적용)**

근거:
- (1) Codex R3가 "1샘플 오류" 지적한 대목이 바로 이 지점. R2에서는 "매 프레임 즉시 주입"으로 읽혔지만 R3 정정 후에는 "최근 N 프레임 평균"으로 수정됐음. 그 수정을 명확히 수식으로 고정: `L_t = α*measured_t + (1-α)*L_{t-1}`, 초기화 `L_0 = 0.35` (6.3과 정합).
- (2) 블링크 시 ROI 픽셀 수가 수십→수백으로 급변하여 measured luma가 순간 튀는 것이 시각적으로 "tint 번쩍임"으로 나타남. EMA 없으면 현장 체감 나쁨. TemporalStabilizer는 center/radius만 스무딩하므로 luma는 별도로 필요.
- (3) 위치는 **렌더러 레벨**이지 adapter가 아님. adapter는 "측정값 전달"만 하고 렌더러 material envelope 원칙(W1 경계)에 따라 렌더러에서 history 유지.

## 6.6 head_pose_yaw_roll / reflection_dir optional

**추천: 스키마만 추가, 현재 렌더러에서는 무시 (W4에서 활용)**

근거:
- (1) W4 B2 env reflection 벤치가 head_pose에 의존하는데, 그때 가서 packet 스키마를 다시 깨고 싶지 않음. 지금 optional 필드로 예약해두면 W4 구현 시 adapter만 채우고 렌더러가 쓰면 됨. 기본값 `(0,0,0)` / `nullopt`.
- (2) 현재 렌더러에서 읽지 않으면 실효 비용 0이고 미래 비용 감소.

## 6.7 render_confidence / TemporalStabilizer 연결

**추천: TemporalStabilizer에 `getConfidence()` 있으면 재사용, 없으면 신규 추가 말고 `iris_radius_stability` 기반 파생값 사용**

근거:
- (1) 새 confidence 수식을 이 W1에서 발명하면 W1 범위가 불필요하게 늘어남. 이미 있는 값이 최우선.
- (2) 없다면: `confidence ≈ 1 - clamp(|radius_t - radius_{t-1}| / radius_t, 0, 0.3) / 0.3` 같은 간단 파생으로 충분. iris_radius 스무딩 정도가 검출 신뢰도와 상관 높음.
- (3) **W1 스코프에서 render_confidence 활용처는 없음** (현재 렌더러 어떤 분기도 confidence를 읽지 않음) — 그러면 이것도 6.6처럼 "스키마만 예약, 미활용"으로 두는 것이 R1의 합리적 선택. 현장 필요 생기면 W4+ 때 쓰자.

---

## 요약 표

| 쟁점 | Claude 추천 |
|------|-------------|
| 6.1 측정 경로 | **옵션 A (CPU 1회)** |
| 6.2 ROI 마스크 | r<0.60 * eyelid_aperture * outlier-drop |
| 6.3 fallback 중립 | α=0.35 |
| 6.4 어댑터 위치 | `gpu/eye_render_packet_adapter.*` |
| 6.5 EMA 적용 | 적용, α=0.3, 렌더러 레이어 |
| 6.6 head_pose/reflection_dir | 스키마 예약, 미활용 |
| 6.7 render_confidence | TemporalStabilizer 재사용; 없으면 파생, 단 W1에선 미활용 |

---

## 자기비판 (편향 경계)

- 6.1/6.2/6.3은 Codex R2 원값(A / 0.65 / 0.35)에 가깝거나 일부만 수정한 값. Codex 편향 유의 지점. Codex가 R1에서 R2를 그대로 재확인하면 Claude도 동의한 셈이니 합의로 보되, Gemini가 이견 낸 항목은 Gemini 쪽도 실기기 근거 확인 필요.
- 6.5 EMA는 Claude가 Codex R2의 "1샘플" 해석을 걸고 넘어진 지점이라 **자기 입장이 Claude 쪽일 수 있음** — R2 응답에서 Codex/Gemini가 "불필요"라 반박하면 재검토.
