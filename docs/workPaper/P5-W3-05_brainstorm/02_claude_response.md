# Claude 렌즈 렌더링 자연스러움 브레인스토밍 응답 (R1)

> **작성**: Claude Opus 4.7 (1M)
> **시점**: 2026-04-22 R1
> **입장**: 해체보다 재조율(refactor-vs-retune 원칙). 지금 구조에서 "환경 반응성"과 "선택지 축소"가 최대 ROI.

## 한 줄 결론

지금 해야 할 것: **블렌드 3종으로 압축 + 카메라 프레임 기반 환경 반사로 고정 조명 대체**.
하지 말아야 할 것: **3D FaceGeometry 재도입, full PBR, HDR IBL 에셋 추가, neural rendering**.

## 핵심 주장 (5개)

1. **우리의 유일한 ace는 "실시간 카메라가 있다는 것"이다.** Perfect는 사전 베이크된 `eye_ibl.hdr`로 환경광을 시뮬레이션하지만, 우리는 지금 이 순간의 카메라 프레임을 환경맵으로 쓸 수 있다. 이게 고정 조명(W3-04)보다 근본적으로 우월하다.
2. **블렌드 8종 중 5종은 중복·하위호환이거나 렌즈 도메인에 안 맞는다.** Normal / LuminanceTintLinear / Multiply 3종으로 거의 모든 SKU 커버 가능. 드롭 사유가 각각 명확하다.
3. **렌더러 입력 계약은 "타원 6파라미터 + iris center/radius + pupil center(optional)"에서 못 늘어난다.** head_pose, gaze, eye_depth_mm은 검출기가 바뀔 때마다 깨지는 필드 — 상위에서 파생하거나 아예 안 받는다.
4. **축 D 시간적 안정성은 렌더러 레벨에선 alpha hysteresis만 추가.** W1 TemporalStabilizer와 중복 스무딩하면 디버깅 지옥.
5. **지금 품질 문제는 "디테일 부족"이 아니라 "환경 반응성 부족"이다.** neural rendering, 3D geometry 같은 대안은 해결하려는 문제가 다르다.

## 축별 결론

### 축 A — 광학 리얼리즘

- **추천: 카메라 프레임 기반 저해상도 환경 반사 (Screen-Space Environment Reflection)**
  - **무엇**: 카메라 텍스처를 `textureLod(uCameraTexture, uv, 3.0)`으로 2~3단계 미립화해 환경맵 대체. `normal.xy` 기반 반사 방향에 해당하는 카메라 영역을 specular 기여로 합성. 고정 조명 제거.
  - **근거**: Perfect의 `eye_ibl.hdr(1.8MB)`이 하는 일은 "촬영 환경 시뮬레이션". 우리는 실제 환경을 이미 카메라로 들고 있으니 시뮬레이션 대신 **샘플링**하면 된다. 사용자가 실내·야외·야간 어느 환경에 있든 하이라이트가 자동으로 따라옴. Perfect가 이 접근을 못 쓴 이유는 그들의 Venus 엔진이 메이크업 전반용이라 눈 전용 최적화 여유가 없었기 때문일 것.
  - **비용**: 저. mip chain은 GL 내장 (`glGenerateMipmap`). 추가 ALU 수 줄, pass는 기존 LENS_OVERLAY_FRAGMENT 내부에서 해결 (추가 pass 0개). MID tier +0.2ms 예상.
  - **리스크**: W3-03에서 경험한 Adreno mipmap + dynamic branch 이슈. 방어: `textureLod` 명시 사용 (자동 gradient 금지). 그리고 반사 강도는 `finalAlpha`로 마스킹해 렌즈 영역 밖 영향 차단.
  - **기각한 대안**:
    - HDR IBL 에셋 추가 (Perfect 방식): +1.8MB APK, 정적 조명이라 환경 변화 반응 불가. 실시간 카메라를 안 쓰는 건 손해.
    - 현 W3-04 고정 조명 유지: `(0.3, 0.4, 1.0)` 하드코딩이라 실외 낮/야간/측면광에서 방향 불일치 → 어색함 증가.
    - 사전 베이크 텍스처 노멀맵(Perfect `eye_normal.png` 512×512): +500KB, 분석적 노멀과 시각 차이 미미 (구면 대칭이라 자유도 낮음).

- **기각 (표)**:

| 대안 | 기각 사유 |
|------|----------|
| Full PBR (Cook-Torrance BRDF) | 튜닝 파라미터만 늘고 체감 향상 미미. 렌즈는 구면 + 특정 반사 특성이라 BRDF 다양성이 의미 없음. |
| Neural rendering / super-res | SDK 크기·지연 예산 파괴. 품질 문제가 학습 데이터 부족이 아니라 셰이더 디자인 부족. |
| 실시간 굴절 (corneal refraction) | 육안 거의 안 보임. 효과 대비 비용 과다. |

### 축 B — 블렌드 모드 (4개 이하 최종 세트)

- **최종 세트 (3종)**:

| ID | 이름 | 수식 (GLSL) | 커버 시나리오 |
|----|------|------------|--------------|
| 0 | **Normal** | `mix(base, blend, a)` | 불투명 서클렌즈, 강한 색 무늬 |
| 1 | **LuminanceTintLinear** (디폴트) | 선형공간 휘도 스케일 + 실반사 보호 (임계값은 **0.55~0.85로 튜닝 필요**) | 자연 반투명 렌즈 전반, 헤이즐/그레이/블루/바이올렛 등 밝은 톤까지 |
| 2 | **Multiply** | `base * blend` | 다크 브라운/블랙 짙은 렌즈 (밝은 홍채에 올릴 때) |

  - **근거**: 기존 8종을 렌즈 SKU × 사용자 홍채 매트릭스로 분해하면 셀 대부분이 위 3종 중 하나의 opacity/intensity 튜닝으로 수렴. Screen은 LuminanceTintLinear의 하위 케이스, Overlay는 렌즈 도메인에 과장이라 자연스러움 저해.
  - **비용**: 저. 기존 코드 3종만 남기면 프래그먼트 분기 감소 → 레지스터 압력 완화.
  - **리스크**: 프로덕션 SKU 중 SoftLight/ColorReplace를 기본값으로 쓰는 것이 있다면 마이그레이션 필요. 이건 SKU 메타데이터 매핑 테이블로 흡수.

- **드롭 5종 + 사유**:
  - **Screen**: `Normal` + opacity 0.7~0.8으로 90% 흡수. "밝은 톤"은 LuminanceTintLinear가 이미 커버.
  - **Overlay**: 대비 강화가 렌즈 도메인에서는 "플라스틱 장난감" 느낌을 만듦. 드롭.
  - **LuminanceTint (non-linear)**: LuminanceTintLinear의 감마 오차 버전 — 하위호환이라 유지 의미 없음.
  - **SoftLight**: Normal과 체감 차이 미미. 드롭.
  - **ColorReplace**: 불투명 렌즈는 Normal로 충분. 디테일 손실이 원본 휘도 구조까지 파괴해서 자연스러움 저해.

- **신규 추가/대체**: **없음**. 3종으로 전 스펙트럼 커버. 굳이 추가한다면 `GrainMerge(base + blend - 0.5)`가 후보지만 LuminanceTintLinear와 수학적 중복.

- **기본값 1개**: **LuminanceTintLinear**. 가장 범용적이고 "실제 렌즈 쓴 느낌"에 제일 근접. UI의 첫 번째 선택지로 노출.

### 축 C — 경계·마스킹

- **추천**:
  - **림발 다크닝**: `uLimbalEnabled` uniform으로 **opt-in 유지, 기본 OFF**. 대신 텍스처 제작 가이드라인에 "림발 포함 권장" 문구. (향후 텍스처 픽셀 히스토그램 기반 자동 검출도 가능하지만 Phase 6.)
  - **속눈썹 가림**: 별도 알파 패스 불필요. 타원 마스크 `feather`를 현 `eyelidFeather * 3.0` → `* 4.0`으로 약간 증가하면 속눈썹 영역까지 자연스럽게 감쇄.
  - **Sclera Protection**: 현 `geomFactor(0.75~1.0) × (0.5 + 0.5 * colorFactor)`를 유지. 취약점은 **속눈썹 그림자가 "흰자 색상 기준"에 안 걸려 렌즈가 새는 것** — 그림자 영역(낮은 밝기)에서 color-based factor가 작아져 오히려 보호가 약해짐. 대응: `brightFactor` 하한 0.3 추가.
- **기각**:
  - 속눈썹 전용 세그멘테이션 모델: ROI 작고 가림도 가벼워 과잉.
  - 림발 기본 ON: 텍스처 이중 적용 리스크.
  - 흰자·홍채 경계 feather를 모든 블렌드에서 강제: ColorReplace를 드롭하므로 불필요.

### 축 D — 시간적 안정성

- **추천**:
  - **Alpha hysteresis (렌더러 전담)**: 검출 신뢰도 급락·블링크 시 3~5프레임 alpha ramp down. 신뢰도 급상승 시도 동일 램프 up. `uRenderConfidence`를 입력 계약에 추가.
  - **블링크 시 페이드**: `eyeOpening < threshold` → alpha ramp down. 현재 contact shadow만 disable되고 렌즈는 그대로라 부자연.
  - 그 외 좌표·타원·휘도 스무딩은 **W1 TemporalStabilizer에 전임**. 렌더러는 stabilized 좌표만 받는다.
- **기각**:
  - 렌더러에서 추가 좌표 스무딩: W1 이중 구현, 디버깅 악화.
  - "이전 프레임 렌더 결과 블렌딩" (temporal blur): 모션 블러 버그 발생. 드롭.

### 축 E — 입력 계약

- **추천 (계약을 최소 변경)**:
  - **기존 유지**: iris_center, iris_radius, 타원 6파라미터 (center·radii·rotation), 눈꺼풀 top/bottom, avg_iris_lum.
  - **추가 (1개, optional)**: **`pupil_center` (optional)**. 사용자가 사선을 볼 때 iris_center(홍채 중심)와 pupil_center(동공 중심)가 어긋남 → 렌즈의 투명 동공 영역을 pupil_center에 정렬. 검출기가 pupil을 안 주면 iris_center 재사용 (기존 동작).
  - **추가 (1개, optional)**: **`render_confidence` (optional)**. 축 D alpha hysteresis용. 상위가 신뢰도를 전달, 렌더러가 램프.
- **기각**:
  - **head_pose**: 검출기 교체 시 가장 깨지기 쉬운 필드. 필요하면 상위에서 파생 후 렌더러엔 타원 파라미터로만 전달.
  - **gaze_vector**: pupil_center로 이미 간접 표현 가능. 별도 필드는 복잡도만 증가.
  - **eyes_dp_length_mm** (피팅몬스터 방식): 물리 단위는 렌더러에 누수되면 안 됨. iris_radius(정규화)로 충분.
  - **검출 실패 폴백을 렌더러가 소유**: 정책 이동은 W1 TemporalStabilizer 책임. 렌더러는 null → 렌즈 미출력, hold-frames는 상위.

### 축 F — 성능

- **추천**:
  - **추가 pass 0개 정책**: 환경 반사도 기존 LENS_OVERLAY_FRAGMENT 내부에서 `textureLod(uCameraTexture, reflectUV, 3.0)`로 해결. 별도 mip-blur pass 추가 않음 (GL 내장 `glGenerateMipmap`으로 커버).
  - **MID tier 예산**: 현 W3-04 대비 +0.3ms 이하. ALU 연산 추가만 허용.
  - **Adreno/Mali 대응**: mipmap + dynamic branch 조합은 **`textureLod` 명시 사용**(자동 gradient 금지), `finalAlpha` 기반 마스킹으로 false branch에서 텍스처 액세스 발생 안 하도록.
- **기각**:
  - Compute shader: GLES 3.1 compute는 드라이버 이슈 많음. 프래그먼트 유지.
  - 멀티패스 블러 pass 추가: 예산 초과. mip chain으로 대체.

### 축 G — 와일드카드 (모두 기각)

| 대안 | 기각 사유 |
|------|----------|
| 3D Face Geometry 재도입 (피팅몬스터 방식) | 입력 계약 파괴 (메시 데이터 누수). 검출기 교체 시 렌더러가 동반 변경 필요. 결합도 증가. |
| Neural rendering (small UNet on eye ROI) | SDK 크기 +2~5MB, 지연 +10~30ms. 문제 진단이 틀림 (디테일 부족이 아닌 환경 반응성). |
| Procedural iris (셰이더 절차적 생성) | 렌즈 SKU 카탈로그가 이미지 기반이라 불일치. B2B 파트너 자산 포맷 깨짐. |
| Fluid refraction (각막 굴절 시뮬) | 육안 거의 안 보임. 효과 대비 비용 과다. |

## Claude가 W3-04까지 해놓은 것 중 틀렸다고 보는 것 (자기비판)

1. **분석적 노멀 + 고정 조명 자체가 방향이 틀렸다.** `vec3(0.3, 0.4, 1.0)` 하드코딩 조명은 사용자 촬영 환경과 틀어지면 **오히려 어색**. 축 A 환경 반사로 대체해야 함. 지금 토글이 "ON해도 큰 감동 없음, OFF도 평면적"인 건 이 근본 원인.
2. **`specular * 0.7` 강도**: ON/OFF 체감 테스트 목적으로 과다. 자연스러운 값은 **0.25~0.35**. 현재 값은 렌즈가 "유리구슬"처럼 보일 가능성.
3. **림발 다크닝을 하드코딩 `const bool LIMBAL_ENABLED = false`로 완전 비활성**: 근거는 맞지만 확장성 없음. `uLimbalEnabled` uniform으로 opt-in.
4. **LuminanceTintLinear 실반사 보호 `smoothstep(0.7, 0.95, lum)`**: 임계값 너무 높음. 대부분 눈 밝기는 linear 0.3~0.6 영역이라 보호가 거의 안 걸림 → 반사광이 렌즈 색에 물들어 비현실적. **0.45~0.75가 적절**.
5. **블렌드 8종 존재 자체가 의사결정 병목.** 많은 선택지는 SDK 통합 고객사 입장에서 "어느 걸 써야 하지?" 마비를 부름. 3종으로 축소는 단순 정리가 아니라 **제품 결정**.

## 내가 가장 확신 없는 부분

1. **카메라 프레임 기반 환경 반사가 실기기에서 "체감 가능한 수준"의 개선이 될지**. 수학적으로는 맞는데, 사용자가 알아챌 정도의 차이가 날지는 A/B 없이 단정 불가. 너무 미묘해서 OFF와 구분 안 될 가능성도 있음.
2. **블렌드 3종이 충분한가 vs 4종 필요한가**. 특히 "밝은 바이올렛/핑크 렌즈"를 LuminanceTintLinear 하나로 커버 가능한지. 실기기 매트릭스 돌려봐야 확정.
3. **pupil_center 추가가 검출기 측에 부담인지**. MediaPipe Iris는 동공 좌표를 명시 출력하지 않아 iris landmark에서 파생해야 함. 파생 비용이 지연을 얼마나 늘릴지 미지.
4. **`textureLod`의 Adreno/Mali 구현 차이**. 이론상 안전하지만 드라이버 버그 경험이 적은 영역.

## 다음 단계 제안

1. **실기기 A/B (최우선, 1~2시간)**: 현 W3-04(노멀+고정조명 ON) vs 제안(환경 반사) 동일 SKU로 5개 조도 환경(실내 형광등/창가/실외 낮/실외 해 질 녘/야간 실내) 비교. 체감 차이가 "한눈에 보이는" 수준이면 채택, 아니면 폐기.
2. **블렌드 매트릭스 캡처 (1시간)**: 3개 SKU(다크브라운/헤이즐/바이올렛) × 2 홍채 톤(짙음/밝음) × 3 블렌드 매트릭스. LuminanceTintLinear가 어느 셀에서 부족한지 확인. 부족 셀 0~1개면 3종 확정, 2개 이상이면 4번째 추가.
3. **환경 반사 프로토타입 (2시간)**: `textureLod(uCameraTexture, reflectUV, 3.0)` 프래그먼트 한 줄 추가 + `finalAlpha` 마스킹. 기존 토글 UI 재활용 ("Env Reflect" 버튼).
4. **블렌드 드롭 마이그레이션 (30분)**: 기존 8종 → 3종 매핑 테이블. 5종은 Normal 또는 LuminanceTintLinear로 auto-remap.

## 부록 — R2에서 Codex/Gemini에게 묻고 싶은 것

- Gemini/Codex 중 누군가 **HDR IBL 찬성** 입장을 내면: 카메라 프레임 샘플링 대비 HDR 에셋의 구체적 우위는? 정적 환경맵이 실시간 카메라 대비 나은 시나리오가 존재하나?
- **블렌드 4종 주장** 시: 내가 드롭한 5종 중 어느 것을 살려야 하는가? 살리는 근거가 "특정 SKU 매트릭스 셀 커버"인지 "사용자 선호도"인지?
- **3D Face Geometry 찬성** 시: 결합도 증가를 감수할 만한 구체 시나리오? 고개 회전 시 원근 변형 외에?
- **Neural rendering 찬성** 시: SDK 크기·지연 예산 파괴를 감수할 품질 향상의 최소 기준은?
