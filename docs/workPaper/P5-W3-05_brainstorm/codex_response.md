# Codex 렌즈 렌더링 자연스러움 브레인스토밍 응답

## 한 줄 결론
지금 당장 해야 할 것은 `고정광 기반 구면 하이라이트`를 버리고 `움직이는 환경 반사 + 4종 이하 블렌드 체계 + 홍채 detail 재주입`으로 렌즈를 다시 정의하는 것이고, 하지 말아야 할 것은 W3-04의 fixed-light 미봉책을 더 정교하게 다듬는 것이다.

## 핵심 주장 (3~5개, 각 1문장)
- 지금 결과물이 어색한 1순위 원인은 노멀 디테일 부족이 아니라 `환경 반사 cue 부재`다.
- 블렌드 모드 8개는 표현력이 아니라 의사결정 비용만 늘리고 있고, 실제로는 4개도 많다.
- 경계 자연스러움은 `색상 기반 sclera 보호`보다 `림발/내경계 설계`가 더 크게 좌우한다.
- W1이 좌표를 이미 안정화하므로 렌더러는 geometry를 다시 붙잡지 말고 `alpha/반사 강도`만 부드럽게 다뤄야 한다.
- 렌더러 계약은 raw detector 산출물이 아니라 `2.5D eye packet`으로 고정해야 모델을 바꿔도 셰이더가 안 흔들린다.

## 축별 결론

### 축 A — 광학 리얼리즘
- **추천**: `고정 diffuse/spec`를 버리고, 분석적 구면 노멀은 유지하되 `LDR/RGBM eye env map + Fresnel reflection` 1층으로 교체한다.
- **무엇**: `R = reflect(-V, N)`로 소형 eye env map을 샘플하고, `face_rotation` 또는 optional `reflection_dir`로 env를 회전시켜 각막 반사만 더한다. diffuse는 빼고, highlight는 blend alpha가 아니라 Fresnel과 visibility에 묶는다.
- **근거**: 경쟁사 관찰 + 렌더링 이론. Perfect가 `eye_ibl.hdr`를 별도 자산으로 들고 있는 이유는 곡면감의 핵심 cue가 노멀 디테일이 아니라 환경 반사이기 때문이다. 실제 눈은 diffuse보다 corneal reflection 위치와 이동이 더 강하게 "진짜 같다"를 만든다.
- **비용**: 구현 중, 런타임 저~중. 추가 pass 0개, 추가 texture fetch 1개, 추가 asset 1개(256x128~512x256 수준)로 충분하다.
- **리스크**: env가 너무 generic하면 실내/역광에서 반사 성격이 과장될 수 있다. pose 없이 쓰면 움직임 설득력이 약해진다.
- **기각한 대안**: `고정 diffuse/spec 유지`, `베이크 eye normal 선행`, `카메라 프레임/SSR env`, `굴절/별도 pupil layer 즉시 도입`.
- **기각**:
  - `고정 diffuse/spec 유지`: 지금 W3-04 방식은 반사가 아니라 "렌즈 중앙 밝기 보정"에 가깝다. 위치가 월드/스크린이 아니라 eye-local에 고정되어 실제 각막 반사처럼 안 움직인다.
  - `베이크 eye normal 먼저 추가`: 환경광이 없는데 노멀 디테일만 늘리면 emboss된 플라스틱처럼 보일 뿐이다. 지금 병목은 노멀 품질이 아니라 광원 모델이다.
  - `카메라 프레임/SSR env`: 전면 카메라 프레임은 reflection source로 부적절하다. 오프스크린 광원이 없고, 얼굴/눈 자체가 반사에 재귀적으로 섞여 "붙은 반짝이"가 된다.
  - `굴절/별도 pupil layer`: perceptual payoff가 반사보다 작다. `pupil_center` 계약도 아직 없어서 바로 들어가면 coupling만 늘어난다.
- **결정 트리**: `reflection_dir` 또는 `face_rotation`이 있으면 env를 회전한다. 둘 다 없으면 정적 env를 낮은 강도로만 쓴다. `pupil_center`가 들어오기 전까지 굴절/시차는 보류한다.

### 축 B — 블렌드 모드 (4종 이하 최종 세트)
- **추천**: 최종 세트는 4종으로 끊는다. 핵심은 `TintLinearV2`를 기본값으로 두고, 나머지는 `어두운 렌즈`, `밝은 렌즈`, `불투명 렌즈`를 덮는 보조 모드만 남기는 것이다.
- **최종 세트** (n = 4):

| 이름 | 수식/의사코드 | 커버 시나리오 |
|---|---|---|
| `TintLinearV2` | `baseL=base*base; lensL=lens*lens; lum=dot(baseL,w); out=sqrt(mix(baseL, lensL * lum * scale, a));` | 반투명 자연렌즈, 중간색(헤이즐/올리브), 투명톤, dark/bright iris 공통 |
| `Multiply` | `out = mix(base, base * lens, a);` | 다크 브라운/블랙, 테두리 강조, 밝은 홍채 위 짙은 렌즈 |
| `ScreenLinear` | `out = sqrt(mix(baseL, 1-(1-baseL)*(1-lensL), a));` | 그레이/블루/바이올렛 같이 밝은 렌즈를 dark iris 위에 올릴 때 |
| `ColorReplaceLinear` | `detail = clamp(pow(lum / avgLum, 0.7), 0.75, 1.25); out = sqrt(mix(baseL, lensL * detail, a));` | 불투명 서클렌즈, 화이트/그래픽 렌즈, 색 정확도가 detail 보존보다 중요한 SKU |

- **드롭**:
  - `Normal`: flat하다. `ColorReplaceLinear`가 불투명 렌즈 역할을 더 잘 덮고, 자연렌즈는 `TintLinearV2`가 더 낫다.
  - `Overlay`: 대비를 과장해서 동공 경계와 저조도 눈에서 바로 가짜가 난다.
  - `LuminanceTint`: `TintLinearV2`의 하위호환이다. 감마 공간 계산을 유지할 이유가 없다.
  - `현재 LuminanceTintLinear`: 아이디어는 맞지만 구현이 틀렸다. `realSpec` 밝기 보호 해킹은 mode 안에 있으면 안 되고, 반사 계층으로 분리되어야 한다.
  - `SoftLight`: 체감 차이가 너무 약하다. opacity 튜닝 가능한 `TintLinearV2`와 역할이 겹친다.
  - `현재 Screen`: 밝은 렌즈 전용 모드는 필요하지만, 감마 공간 Screen은 하이라이트가 너무 쉽게 뜬다. `ScreenLinear`로 대체해야 한다.
  - `현재 ColorReplace`: 방향은 맞지만 sRGB + 평균밝기 휴리스틱이라 불투명 렌즈에서 색 정확도와 detail이 둘 다 불안정하다. `ColorReplaceLinear`로 교체한다.
- **신규/대체**:
  - `TintLinearV2`: 현재 ID 4/5를 흡수하는 새 기본 모드다. 핵심은 선형 공간 tint만 남기고, `spec 보호` 같은 광학 해킹은 모드 밖으로 뺀다는 점이다.
  - `ScreenLinear`: 밝은 렌즈 전용 모드를 유지하되 감마 공간 과장을 줄인다.
  - `ColorReplaceLinear`: 현재 `Normal`과 `ColorReplace`의 불투명 렌즈 용도를 하나로 합친다.
- **기본값 1개**: `TintLinearV2`.
- **이유**: SKU 수가 많아질수록 디폴트는 "대부분의 눈에서 덜 망하는" 쪽이어야 한다. 자연렌즈, 중간색 렌즈, 밝기 조건 변화 대응까지 합치면 `TintLinearV2`가 가장 안전한 기본값이다.

### 축 B-보조 — 홍채 디테일 보존
- **추천**: 블렌드 후에 `detail reinjection` 1층만 추가하고, `Pattern × Palette`는 렌더러 내부 추상화까지만 준비한다.
- **무엇**: 렌즈 합성 후 `detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15)` 같은 고주파 detail을 iris core에만 다시 곱한다. 적용 영역은 `홍채 내부`, `pupil 주변 제외`, `spec/reflection 제외`로 제한한다.
- **근거**: 렌더링 이론 + 경쟁사 관찰. 렌즈가 자연스럽게 보일 때 남아 있는 것은 "원본 색"이 아니라 "원본 미세 구조"다. Perfect의 `Pattern × Palette × Style`도 본질적으로 패턴과 색을 분리한다.
- **비용**: 구현 중, 런타임 중. 추가 camera fetch 4개 내외로 single-pass에서 끝낼 수 있다.
- **리스크**: 저조도 노이즈, 속눈썹 그림자, 아이라인까지 detail로 재주입하면 오히려 더 지저분해질 수 있다.
- **기각한 대안**: `블렌드 수식만으로 detail 보존`, `Pattern × Palette API 즉시 공개`, `원본 detail 100% 유지`.
- **기각**:
  - `블렌드 수식만으로 detail 보존`: global color mapping과 local structure 보존은 다른 문제다. mode 수를 늘려도 해결이 안 된다.
  - `Pattern × Palette API 즉시 공개`: 최종 제품 방향으론 맞지만, 지금 단계에서 API까지 흔들면 검증 비용이 너무 커진다.
  - `원본 detail 100% 유지`: 불투명 서클렌즈의 SKU identity가 무너진다. 렌즈는 보이되, 미세 구조만 일부 돌려주는 게 맞다.
- **Pattern × Palette API 영향**: 공개 API는 당장 바꾸지 말고, 내부 material 모델만 `patternTex + palette + style`을 받을 수 있게 만든다. 기존 단일 PNG는 `baked pattern`으로 그대로 호환시킨다.

### 축 C — 경계·마스킹
- **추천**: 기본 경계 전략은 `셰이더 합성 림발 링 + 내경계(pupil) feather + geometry-first sclera 보호`로 정리한다.
- **무엇**: 외곽에는 parametric limbal ring을 두고, 안쪽에는 pupil feather를 분리한다. `sclera protect`는 색상 판정이 아니라 `iris/ellipse 경계 기반`을 주로 쓰고, 색상은 veto 정도로만 사용한다. 속눈썹은 별도 pass를 만들지 않는다.
- **근거**: 렌더링 이론 + 현재 구현 비판. 현재 결과물의 어색함은 "홍채 끝이 어디인지 모르는" 경계 문제다. `LIMBAL_ENABLED = false`로 전역 off를 걸어버리면 texture authoring 품질에 자연스러움이 전부 종속된다.
- **비용**: 구현 저, 런타임 저. `smoothstep` 몇 개와 uniform 2~3개 추가 수준이다.
- **리스크**: texture에 이미 강한 림발이 있으면 이중 적용으로 과한 ring이 생길 수 있다.
- **기각한 대안**: `텍스처 내 림발을 default`, `속눈썹 알파 pass 추가`, `현재 색상 기반 sclera protect 유지`.
- **기각**:
  - `텍스처 내 림발 default`: asset 품질 편차가 크다. 자연스러움의 핵심 cue를 PNG authoring에만 맡기는 것은 SDK 설계 실패다.
  - `속눈썹 알파 pass 추가`: detector coupling이 급격히 올라가고 pass 비용도 오른다. 현재 단계에서는 eyelid ellipse 개선이 훨씬 싸고 효과가 크다.
  - `현재 색상 기반 sclera protect 유지`: 밝고 저채도인 그레이/블루 렌즈, spec highlight, 메이크업 반사가 모두 sclera로 오인될 수 있다.
- **결정 트리**: SKU에 `has_baked_limbal` 메타데이터가 있으면 셰이더 림발 강도를 0에 둔다. 없으면 셰이더 림발을 기본값으로 켠다.

### 축 D — 시간적 안정성
- **추천**: 렌더러는 `material-only temporal envelope`만 가진다. geometry hold는 상위 계층(W1 또는 detector adapter)에서만 한다.
- **무엇**: eye별로 `render_alpha`, `reflection_strength`, `limbal_strength`만 EMA/hysteresis로 완만하게 움직인다. blink 시 즉시 off가 아니라 60~80ms ease-out, 재오픈 시 100~120ms ease-in을 둔다. 좌표/반경은 렌더러에서 다시 스무딩하지 않는다.
- **근거**: 구조 원칙 + 실시간 합성 이론. W1이 이미 좌표 안정화를 소유하고 있는데 렌더러가 geometry까지 붙잡으면 lag와 이중 스무딩이 생긴다. 사용자가 느끼는 flicker는 대부분 위치가 아니라 on/off, highlight pop, eyelid clamp 전환에서 나온다.
- **비용**: 구현 저, 런타임 저. eye별 상태 변수 몇 개면 충분하다.
- **리스크**: fade가 길면 실제 검출 상실 후 1~2프레임 ghost가 남을 수 있다.
- **기각한 대안**: `즉시 off`, `렌더러가 이전 좌표를 몇 프레임 더 유지`, `geometry도 다시 smoothing`.
- **기각**:
  - `즉시 off`: blink 때 렌즈가 깜빡여 가장 티 난다.
  - `렌더러가 이전 좌표를 몇 프레임 더 유지`: 계약이 꼬인다. 추적기와 렌더러가 서로 다른 truth를 들고 가면 디버깅이 지옥이 된다.
  - `geometry도 다시 smoothing`: 지연만 늘고, 정면/측면 전환에서 스냅을 더 키운다.
- **결정 트리**: detector/W1이 `visibility`를 내려주면 렌더러는 `visibility > 0.7` 정상, `0.3~0.7` 페이드, `< 0.3` 렌더 종료로만 처리한다. 검출 실패 hold는 상위 계층에서 최대 2프레임까지만 허용한다.

### 축 E — 입력 계약
- **추천**: 렌더러 입력을 `EyeRenderPacket`으로 고정하고, raw `IrisResult`는 adapter에서만 소비하게 만든다.
- **무엇**:
  - 필수 필드: `iris_center_norm`, `iris_radius_norm`, `aperture_mask(비대칭 ellipse 또는 top/bottom fallback)`, `visibility`, `timestamp_ms`
  - 선택 필드: `pupil_center_norm`, `head_pose_yaw_roll` 또는 `reflection_dir`, `avg_iris_luma`, `eye_depth_mm`
  - 금지: raw `face_mesh` 의존, detector-specific landmark 인덱스 의존
- **근거**: 아키텍처 원칙 + 현재 코드 상태. 지금 `IrisResult`는 detector 메타데이터와 렌더 힌트가 섞여 있고, 반대로 렌더러는 필요한 광학 정보(`pupil_center`, `avg_iris_luma`)는 못 받으면서 `uAvgIrisLum = 0.35` 같은 하드코딩으로 버티고 있다.
- **비용**: 구현 중, 런타임 저. adapter 레이어와 opt-in struct만 추가하면 된다.
- **리스크**: C API/Flutter/Web 바인딩에 필드 확장 전파가 필요하다.
- **기각한 대안**: `raw face_mesh 그대로 렌더러 입력`, `head_pose/eye_depth를 필수화`, `검출 실패 폴백을 렌더러가 소유`.
- **기각**:
  - `raw face_mesh 그대로 렌더러 입력`: detector 교체 자유를 즉시 잃는다.
  - `head_pose/eye_depth를 필수화`: 경쟁사 수준의 cue는 좋지만, 없는 detector도 많다. opt-in이어야 맞다.
  - `검출 실패 폴백을 렌더러가 소유`: temporal ownership이 분산된다. 이미 W1이 있는 구조와 충돌한다.
- **결정 트리**: optional 필드가 없으면 기능을 꺼야지, 기본값으로 물리량을 지어내면 안 된다. `pupil_center` 없으면 pupil parallax off, `avg_iris_luma` 없으면 cheap ROI 측정값을 쓰고, 그것도 없을 때만 보수적 fallback을 쓴다.

### 축 F — 성능
- **추천**: `single-pass 유지`를 절대 조건으로 두고, 허용 예산은 `env lookup 1개 + optional detail fetch 4개`까지로 제한한다.
- **무엇**: reflection과 detail reinjection을 모두 기존 렌즈 pass 안에 욱여넣고, 추가 FBO pass는 금지한다. env map은 no-mip 또는 `textureLod(..., 0.0)`로 읽고, detail은 `finalAlpha > threshold` 영역에서만 계산한다.
- **근거**: 현재 테스트도 평균 렌더링 10ms 이하를 30fps 기준으로 잡고 있다. 지금 렌더러는 이미 풀스크린 pass이기 때문에 pass 수가 늘어나는 순간 체감 비용이 커진다. Adreno에서는 mipmap + dynamic branch 이슈도 이미 한 번 맞았다.
- **비용**: 구현 저~중, 런타임 중. 추가 pass 0개를 유지하는 대신 셰이더 복잡도만 조금 늘어난다.
- **리스크**: Mali에서 divergent branch가 길어지면 ALU보다 branch cost가 더 커질 수 있다.
- **기각한 대안**: `reflection/detail 별도 pass`, `runtime mipmap/prefilter chain`, `카메라 기반 SSR`, `full HDR pipeline`.
- **기각**:
  - `reflection/detail 별도 pass`: eye effect 품질보다 FBO 전환 비용을 먼저 사게 된다.
  - `runtime mipmap/prefilter chain`: 눈 영역 이펙트에 비해 비용 대비 체감이 작다.
  - `카메라 기반 SSR`: 품질도 낮고 구현 복잡도만 높다.
  - `full HDR pipeline`: 현재 출력이 8-bit camera composite인 구조에서 과하다.
- **Adreno 주의점**: mipmapped texture를 divergent branch 안에서 샘플하지 않는다. env map은 no-mip 또는 `textureLod` 고정 LOD가 안전하다.
- **Mali 주의점**: branchless 수식이 유리하다. highlight/detail gate는 if 남발보다 `mix`, `step`, uniform 기반 분기가 낫다.

### 축 G — 와일드카드
- **추천**: 유일하게 채택할 가치가 있는 와일드카드는 `eye-only second-stage refiner`다.
- **무엇**: 소형 eye crop 모델 하나를 추가해 `pupil_center`, `occlusion`, optional `gaze yaw/pitch`만 더 뽑는다. full face geometry는 하지 않는다.
- **근거**: 경쟁사 관찰. Perfect의 `second_step_eye_model`(228KB)과 피팅몬스터의 분리된 iris pipeline이 보여주는 것은 "정밀 eye packet"의 가치지, full face mesh 자체의 가치가 아니다. 현재 어색함 중 `동공 중심 불일치`, `눈꺼풀 경계`, `측면 시차`는 이 작은 refiner가 직접 건드린다.
- **비용**: 구현 중, 런타임 중. 소형 모델 1개(대략 100~250KB급) + eye crop 2회 추론이 추가된다.
- **리스크**: 다양한 눈화장/속눈썹/컬러렌즈 착용 사용자에서 일반화 검증이 필요하다.
- **기각한 대안**: `3D Face Geometry 재도입`, `neural harmonization`, `절차적 iris 생성`.
- **기각**:
  - `3D Face Geometry 재도입`: 얻는 것보다 coupling이 크다. 현재 아키텍처 원칙과 정면 충돌한다.
  - `neural harmonization`: 데이터셋, latency, 디버깅 비용이 지금 단계에서 전혀 안 맞다.
  - `절차적 iris 생성`: SKU 제어력이 떨어진다. 컬러렌즈 비즈니스는 procedural beauty demo가 아니라 상품 재현성이 핵심이다.

## Claude가 W3-04까지 해놓은 것 중 틀렸다고 보는 것
- `cpp/src/gpu/shader_sources.cpp:1010-1031`의 분석적 노멀 + fixed light는 방향이 틀렸다. 분석적 노멀 자체는 lookup normal로는 쓸 수 있지만, 여기에 `고정 diffuse/spec`를 얹는 순간 각막 반사가 아니라 eye-local decal 조명이 된다.
- 같은 코드에서 `lightMask = 1.0 - smoothstep(0.0, 1.0, sqrt(r2))`는 하이라이트를 중심부에 강제한다. 실제 반사는 limbal 근처 상부로도 쉽게 간다. 지금 수식은 "가운데가 반짝이는 스티커"를 만든다.
- `blended * mix(1.0, lighting, finalAlpha)`는 렌즈 opacity가 높을수록 광학 반사가 세진다는 뜻인데, 이건 물리적으로 틀렸다. 반사는 pigment alpha가 아니라 cornea/view/light 관계로 정해져야 한다.
- `cpp/src/gpu/shader_sources.cpp:865-874`의 `realSpec = smoothstep(0.7, 0.95, lum)`는 spec 보호가 아니라 밝은 픽셀 보호다. 밝은 홍채, daylight glare, pale lens 패턴이 전부 spec로 오인될 수 있다.
- `cpp/src/gpu/gpu_lens_renderer.cpp:811-813`에서 `uAvgIrisLum = 0.35`를 고정값으로 때려 넣는 것은 논리적으로 특히 나쁘다. 사용자/조명/노출이 바뀌는데 "한국인 평균 근사"로 블렌드를 정규화하면, 개인화가 아니라 priors를 덮어씌우는 셈이다.
- `cpp/src/gpu/shader_sources.cpp:999-1005`처럼 림발을 전역 비활성으로 둔 것은 잘못된 default다. "일부 텍스처에 이미 있다"는 이유로 SDK 차원의 경계 cue를 버리면 품질 책임을 asset authoring에 떠넘긴다.
- `cpp/src/gpu/shader_sources.cpp:903-910`의 `calcScleraFactor`는 밝고 저채도인 렌즈와 spec highlight를 sclera로 오인할 가능성이 크다. 특히 그레이/블루 계열에서 역효과가 날 여지가 높다.
- `cpp/include/iris_sdk/gpu/gpu_lens_renderer.h:262`의 5프레임 eyelid/ellipse hold는 W1이 이미 있는 구조에서 renderer ownership이 과하다. temporal ownership을 분산시키는 쪽으로 가면 안 된다.

## 내가 가장 확신 없는 부분
- `LDR/RGBM env map`만으로 충분한지, 아니면 실제로는 HDR asset의 headroom이 필요한지는 실기기 A/B 없이 단정할 수 없다. 다만 지금 병목이 dynamic range보다 reflection motion cue라는 점은 확신한다.
- `detail reinjection`을 single-pass 4~5 tap 수준으로 넣었을 때 저조도 노이즈가 얼마나 튀는지는 아직 실측이 필요하다.

## 다음 단계 제안 (2~3개)
- `TintLinearV2 / Multiply / ScreenLinear / ColorReplaceLinear` 4종만 구현해서, `렌즈 타입 × 사용자 홍채 × 조명` 매트릭스 20~30클립으로 블라인드 비교를 돌린다.
- 현재 W3-04 하이라이트를 완전히 끄고, `env reflection` 프로토타입 하나만 올린 뒤 Adreno와 Mali에서 추가 pass 없이 성능을 잰다.
- detector adapter에서 `EyeRenderPacket` 초안을 만들고, `pupil_center 없음/있음`, `head_pose 없음/있음` 두 경로만 먼저 연결해 셰이더 분기 없이 품질 차이를 확인한다.
