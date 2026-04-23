# R2 쟁점 목록 (Claude 모더레이터 작성)

> **작성**: Claude
> **시점**: 2026-04-22 R1 종료 직후
> **목적**: R1에서 합의된 것은 닫고, 갈린 것만 R2에서 정면 대결시킨다.

---

## A. 닫힌 쟁점 (R2에서 재논의 금지)

세 모델이 같은 결론을 냈거나, 한 모델의 강한 반대 없이 합의된 항목들.

| # | 항목 | 결론 |
|---|------|------|
| A1 | HDR IBL 에셋 도입 (~1.8MB) | **기각** — 세 모델 합의. 정적 조명이라 가치 낮음. |
| A2 | Full PBR (Cook-Torrance BRDF) | **기각** — 세 모델. 튜닝 비용만 늘고 체감 없음. |
| A3 | 3D Face Geometry 재도입 | **기각** — 세 모델. 결합도 증가. |
| A4 | Neural rendering / harmonization | **기각** — Claude/Codex/Gemini 세 모델 (Gemini의 절차적 노이즈는 별 항목). |
| A5 | Full corneal refraction 시뮬 | **기각** — 세 모델. 육안 체감 미미. |
| A6 | LTL 계열을 기본 블렌드로 | **채택** — 세 모델 합의. |
| A7 | Multiply 블렌드 유지 | **채택** — 세 모델. |
| A8 | 렌더러 추가 FBO pass 0개 (single-pass 절대 조건) | **채택** — 세 모델. |
| A9 | W1 TemporalStabilizer가 좌표 스무딩 소유. 렌더러는 material-only temporal envelope | **채택** — Claude/Codex 명시, Gemini 미반대. |
| A10 | 현 W3-04 고정 조명(`vec3(0.3, 0.4, 1.0)`)은 방향이 틀렸음 | **채택 — 폐기 확정.** 세 모델 모두 비판. |
| A11 | 림발 다크닝을 전역 비활성(`LIMBAL_ENABLED=false`)으로 두는 것은 잘못된 default | **채택 — 제거 확정.** Codex/Gemini 강하게 비판, Claude도 동의 전환. |
| A12 | `realSpec = smoothstep(0.7, 0.95, lum)` — 임계값이 틀렸거나 개념 자체가 틀림 | **채택 — 수정 확정.** 세 모델 동의. 수정 방향은 쟁점 I3에서. |

---

## B. 열린 쟁점 (R2에서 정면 대결)

각 쟁점에서 **자기 R1 입장을 유지할지, 상대 논리를 수용할지 명시**하세요. 얼버무림 금지.

---

### I1. 환경 반사 소스 — 카메라 프레임 vs 에셋 env map (⭐ 최대 쟁점)

| 진영 | 주장 | R1 근거 |
|------|------|---------|
| **A** (Claude, Gemini) | 카메라 프레임을 mip-down 또는 고휘도 샘플링해서 환경맵으로 쓰자 | "우리의 유일한 ace는 실시간 카메라. 정적 HDR 대비 환경 자동 반응" (Claude), "창문/형광등이 렌즈에 맺혀야 '그 자리에 있음' 성립" (Gemini) |
| **B** (Codex) | LDR/RGBM eye env map 에셋(256×128~512×256) 추가 + Fresnel + pose 회전 | "전면 카메라는 얼굴밖에 안 찍어 오프스크린 광원 부재. 얼굴/눈이 반사에 재귀적으로 섞여 '붙은 반짝이'가 됨" |

**R2에서 답할 것**:
- **A 진영**: Codex의 "재귀/광원 부재" 논리에 구체 반박하거나 수용. 전면 카메라 프레임에서 (a) 얼굴 제외한 영역(머리카락 위, 이마 위, 화면 경계)의 비율 + (b) 그 영역의 실제 휘도가 반사로 유효한지 실기기 통계 있나? 없다면 어떻게 검증?
- **B 진영 (Codex)**: LDR/RGBM env map 에셋의 "일반 템플릿" 한 장으로 얼마나 다양한 환경(실내 형광/창가/야간/실외 낮)을 커버 가능한가? 사용자가 실제 카메라 하이라이트 위치(예: 창문이 옆에 있을 때)와 env map의 표기 위치가 어긋나면 오히려 더 어색하지 않은가? pose 없는 detector가 대다수라 env 회전도 못 함.
- **공통**: 이 쟁점은 **실기기 2시간 A/B 벤치로 확정 가능**한가? 어떤 메트릭으로 판정할지.

---

### I2. 블렌드 최종 세트 — 3종 vs 4종, Normal 드롭 여부

| 진영 | 최종 세트 |
|------|----------|
| Claude | **Normal / LTL / Multiply** (3종) |
| Codex | **TintLinearV2 / Multiply / ScreenLinear / ColorReplaceLinear** (4종, Normal 드롭) |
| Gemini | **LTL / Circle (Normal+Overlay 하이브리드) / Vivid (SoftLight+Multi 하이브리드)** (3종) |

**서브쟁점**:

- **I2-a. Normal 단독 필요?**
  - Claude: "불투명 서클렌즈 전용. 단순·예측 가능" → 유지
  - Codex: "flat하다. ColorReplaceLinear가 더 잘 덮음" → 드롭
  - Gemini: Circle 하이브리드 안에 녹여서 유지
  - **R2 질문**: 불투명 서클렌즈에서 Normal과 ColorReplaceLinear의 **시각 차이를 구체 GLSL 수식 한 줄로 비교** 가능한가? 둘의 출력이 어떤 SKU 타입에서 다른가?

- **I2-b. Screen/ScreenLinear 별도 필요?**
  - Claude: "LTL이 밝은 톤까지 커버" → 드롭
  - Codex: "LTL만으론 dark iris 위 밝은 렌즈(그레이/블루/바이올렛)가 안 뜸" → ScreenLinear 유지
  - **R2 질문**: LTL이 dark iris × 밝은 렌즈 셀에서 실제로 부족한가? Codex는 실기기 근거 있나? 없으면 실기기 매트릭스 2×3로 검증 가능.

- **I2-c. 하이브리드 블렌드 (Circle, Vivid) — Gemini만 제안**
  - Claude/Codex 입장에서 보면 하이브리드는 "튜닝 파라미터 숨긴 단일 블렌드" 의심. Gemini의 근거 보강 필요.
  - **R2 질문 (Gemini)**: Circle/Vivid가 (Normal + opacity 튜닝) / (Multiply + opacity 튜닝) 대비 **어떤 셀에서 다른 결과**를 내는가? 구체 매트릭스 셀.

- **I2-d. ColorReplace 드롭 vs ColorReplaceLinear 도입**
  - Claude: sRGB ColorReplace 드롭 (디테일 손실)
  - Codex: sRGB ColorReplace 드롭하되 **선형 공간 버전 ColorReplaceLinear 신규 추가** — 불투명 렌즈 + 색 정확도 + detail 보존의 trade-off를 검증된 선에서 잡음
  - Gemini: 드롭 (디테일 손실)
  - **R2 질문**: ColorReplaceLinear의 `detail = clamp(pow(lum / avgLum, 0.7), 0.75, 1.25)` 수식이 실제로 Normal/LTL 대비 별도 모드로 유지할 가치가 있는가? 아니면 LTL의 `scale` 파라미터 튜닝으로 흡수 가능한가?

**R2에서 세 모델 모두 답할 것**: 최종적으로 수렴하려면 어떤 **실기기 매트릭스**로 판정하면 되는지 제안. (예: "3 SKU × 3 홍채 톤 × 4 블렌드 = 36 클립 블라인드 비교")

---

### I3. LTL 실반사 보호 — 모드 내부 수정 vs 반사 계층으로 분리

| 진영 | 접근 |
|------|------|
| Claude | 임계값 0.7/0.95 → **0.45/0.75로 튜닝** (모드 내부 유지) |
| Gemini | `spec > 0.7`에서 `mix(..., 1.0, ...)` 계단현상 → **sigmoid/smoothstep 부드러운 전이** (모드 내부 유지) |
| Codex | "realSpec은 spec 보호가 아니라 밝은 픽셀 보호 (개념 자체 틀림)" → **모드 밖으로 분리. 반사 계층에서 처리** |

**R2 질문**:
- **Codex**: 반사 계층 분리의 구체적 구조는? 반사 계산이 blend 이후에 가산적으로 들어가면, blend에서 이미 톤이 변한 픽셀에 반사가 덧씌워지는 순서 문제는 어떻게 해결?
- **Claude/Gemini**: Codex의 "개념이 틀렸다" 비판에 동의하는가? 동의하면 모드 내부 수정은 임시방편일 뿐인가?
- **합의 가능**: 반사(환경 샘플링)가 분리 계층으로 빠지면, LTL의 realSpec 자체가 필요 없어진다. I1과 묶음 처리 가능한지 검토.

---

### I4. 림발 링 — 기본값 OFF vs ON + 메타데이터 자동 조절

| 진영 | 접근 |
|------|------|
| Claude | 기본 OFF, opt-in 옵션. 텍스처 제작 가이드라인으로 해결 |
| Gemini | 기본 ON, **텍스처 밝기에 반비례로 강도 자동 조절** (셰이더에서 자동 억제) |
| Codex | 기본 ON, **SKU 메타데이터 `has_baked_limbal` 플래그**로 자동 off |

**R2 질문**:
- **Claude**: "SDK 경계 cue를 asset authoring에 떠넘기는 것은 SDK 설계 실패" (Codex)에 동의하는가? 동의하면 기본 OFF 입장 철회.
- **Gemini vs Codex**: 자동 억제(Gemini: 텍스처 밝기 분석) vs 메타데이터(Codex: SKU 플래그) 어느 쪽이 더 안전한가? 텍스처 밝기 분석은 림발 이외의 어두운 영역(동공, 테두리 섬세한 무늬)에도 반응할 위험이 있지 않나?

---

### I5. 블링크 처리 — 즉시 Off vs Ramp Fade

| 진영 | 페이드 시간 |
|------|------------|
| Claude | alpha ramp down (3~5프레임, ~50~80ms) |
| Gemini | **즉시 Off** ("페이드 = 눈꺼풀 위 투영 부자연") |
| Codex | ease-out 60~80ms + ease-in 100~120ms |

**R2 질문**:
- **Gemini**: ease-out 60~80ms 정도의 짧은 페이드도 "눈꺼풀 위 투영"으로 인지되는가? 실제 블링크는 100~150ms 지속이므로 페이드가 블링크보다 짧으면 문제없지 않나?
- **공통**: 페이드가 `eyelidMask`에 묶이면 (렌즈가 눈꺼풀 영역에서 자동 감쇄) 페이드 자체가 눈꺼풀 위로 새는 문제가 없어지지 않나? (Claude 관점)

---

### I6. 입력 계약 확장 — 최소 추가 vs 구조화된 패킷

| 진영 | 접근 |
|------|------|
| Claude | 기존 유지 + `pupil_center` (optional) + `render_confidence` (optional) |
| Gemini | `gaze_vector` 추가 (parallax용). pupil_center는 기각 |
| Codex | **EyeRenderPacket 구조체로 고정**: 필수(iris_center_norm, iris_radius_norm, aperture_mask, visibility, timestamp) + 선택(pupil_center, head_pose_yaw_roll 또는 reflection_dir, avg_iris_luma, eye_depth_mm) |

**R2 질문**:
- **Claude + Gemini**: Codex의 `EyeRenderPacket` 구조화에 동의하는가? pupil_center vs gaze_vector 어느 쪽이 parallax 효과에 더 유효한가? (pupil_center는 "동공 위치", gaze_vector는 "시선 방향" — 동공 위치만으로도 시선 방향은 iris_center-pupil_center로 파생 가능. 그럼 gaze_vector는 redundant?)
- **Codex**: `avg_iris_luma`를 optional로 뒀을 때, detector가 안 주면 fallback은? "cheap ROI 측정값"이 구체적으로 무슨 수식?

---

### I7. `uAvgIrisLum = 0.35` 하드코드 처리 (Codex 신규 제기)

Codex 지적: `gpu_lens_renderer.cpp:811-813`의 `uAvgIrisLum = 0.35` 고정값은 "개인화가 아니라 priors 덮어씌움". 사용자/조명/노출이 변하는데 한국인 평균 근사로 정규화하면 블렌드 결과가 망가진다.

**R2 질문**:
- **Claude, Gemini**: 이 비판에 동의하는가?
- **대안**: 프레임당 iris ROI 평균 휘도 실측? 1프레임 지연 OK? 측정 실패 시 폴백 전략?

---

### I8. calcScleraFactor 존폐 — 유지 수정 vs 폐기

| 진영 | 입장 |
|------|------|
| Claude | `brightFactor` 하한 0.3 추가 수정 유지 |
| Gemini | **폐기**. 조명 변화에 취약, 기하학적 마스크(`r>0.75`)만 사용 |
| Codex | 색상은 **veto 정도로만**. 주 판정은 기하학 기반 |

**R2 질문**:
- **Claude**: Gemini/Codex가 "색상 기반 취약"을 공통 지적. 하한 추가는 미봉책 아닌가? 기하학만 쓰는 안에 반대 근거 있나?
- **공통**: "색상 veto"의 구체적 수식 예시 (Codex 입장 보강 필요).

---

### I9. 홍채 디테일 재주입 — 원본 샘플링 vs 절차적 생성 vs 미언급

| 진영 | 접근 |
|------|------|
| Claude | **미언급** (축 B-보조 수준에서 다루지 않음) |
| Gemini | **절차적 노이즈로 방사형 섬유 생성** (축 G 와일드카드) |
| Codex | **원본 휘도 재주입** (`detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15)`, 축 B-보조) |

**R2 질문**:
- **Claude**: 디테일 재주입 필요성에 동의하는가? 동의하면 Codex 방식과 Gemini 방식 중 어느 쪽?
- **Gemini**: 절차적 생성은 사용자 본인 홍채와 불일치하는 가짜 디테일 아닌가? 원본 샘플링이 더 정직하지 않나?
- **Codex**: 저조도 노이즈가 detail로 증폭되는 문제 (자기도 "내가 확신 없는 부분"에서 언급) — 노이즈 gate 전략은?

---

### I10. W2와의 경계 — eye-only refiner를 W3에서 다룰지

| 진영 | 입장 |
|------|------|
| Claude/Gemini | W2 별도 트랙이므로 W3 브레인스토밍 밖 |
| Codex | W3 와일드카드에 포함. "pupil_center, occlusion, optional gaze"만 뽑는 소형 refiner가 렌더링 품질에 직접 기여 |

**R2 질문**:
- **Codex**: W2-W3 경계를 어떻게 설정해야 하나? W2에서 refiner를 하고 W3 렌더러는 그 산출물을 받는 구조가 아니라, 왜 W3가 refiner를 끌고 가야 한다고 보는가?

---

## C. R2 응답 규칙

1. **응답 파일**: `codex_r2.md`, `gemini_r2.md`, `claude_r2.md` (같은 폴더)
2. **포맷**: 각 열린 쟁점(I1~I10)에 대해
   - **내 R1 입장**: 유지 / 수정 / 철회 중 하나 명시
   - **상대 논리 평가**: 어느 논리는 맞고 어느 논리는 틀렸는지
   - **수렴 제안**: 합의 가능한 지점 또는 실기기 벤치 제안
3. **새 아이디어 금지**: R1에 없던 축/대안을 R2에서 처음 꺼내지 마세요. 쟁점 수렴에 집중.
4. **얼버무림 금지**: "둘 다 일리 있다" 대신 "A가 맞다. B는 X 때문에 틀렸다" 또는 "A를 철회하고 B 수용" 중 하나.

---

## D. R2 이후 예상 액션

R2가 끝나면 Claude가:
- 닫힌 쟁점을 **`final_decision.md` 초안**으로 수렴
- 남는 쟁점은 **실기기 A/B 벤치 계획**으로 이관 (W3-05 실제 구현 착수 전 검증 단계)
- R3 필요 여부 판단 (쟁점 3개 이하로 좁혀지면 종료, 이상이면 R3)
