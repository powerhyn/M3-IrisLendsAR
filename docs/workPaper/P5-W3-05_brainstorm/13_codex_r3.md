# Codex R3 — Claude 종합 검토

## 0. 전체 평가 (한 줄)
Claude의 `99_claude_synthesis.md`는 **일부만 맞다** — 큰 방향은 내 R2를 대체로 반영했지만, `4종 확정`, `Normal 제거`, `avg_iris_luma 1샘플 측정`, `R3 불필요` 판단은 내 R2를 과하게 밀어붙였거나 왜곡했다.

## 1. 내 R2 입장 vs Claude 종합의 불일치

- **정확히 반영된 것**: `realSpec` 폐기, 반사 계층 분리, `EyeRenderPacket` 구조화, `gaze_vector` 드롭, `uAvgIrisLum=0.35` 폐기, 원본 휘도 기반 detail reinjection, W2-W3 경계 정리는 내 R2와 대체로 일치한다.
- **정확히 반영된 것**: `TintLinearV2 / Multiply / ScreenLinear` 3종을 우선 확정한 판단은 내 R2와 일치한다.
- **왜곡되어 반영된 것**: §1 C1은 `블렌드 모드 4종 세트`를 즉시 확정처럼 썼다. 내 R2는 `ColorReplaceLinear`를 `Normal`과 1:1 대결 후 결정하자는 입장이었다. 따라서 C1은 `3종 확정 + 4번째 슬롯 벤치`로 써야 한다.
- **왜곡되어 반영된 것**: §1 D6은 `Normal` 제거를 즉시 제거 대상으로 넣었다. 이건 §2 B1의 `Normal vs ColorReplaceLinear` 벤치와 정면 충돌한다. 벤치 전에는 `Normal`을 제거하면 안 된다.
- **왜곡되어 반영된 것**: §1 C9의 `textureLod(camera, iris_center, 3.0)` 1회 샘플은 내 R2 수식이 아니다. 내 R2는 `avg = sum(dot(rgb,w)*mask)/sum(mask)`, `mask = inner iris ∩ eyelidMask`였다. 중심 1샘플은 pupil/dark center를 iris 평균으로 오인할 수 있다.
- **왜곡되어 반영된 것**: §1 C7의 `α_close=0.15`, `α_open=0.08`은 60~120ms라는 시간 목표와 수학적으로 잘 맞지 않는다. 30fps에서 EMA α=0.15는 2~3프레임 안에 충분히 내려가지 않는다.
- **누락된 것**: §1 C10은 내 R2의 `spec/reflection 제외` 조건을 빠뜨렸다. detail reinjection은 반사 위에 다시 곱해지면 안 된다.
- **누락된 것**: §1 C11은 `occlusion`을 W2 산출물로 언급하지만 §1.3 `EyeRenderPacket` 스키마에는 occlusion이 없다. `visibility/ellipse`로 흡수한다는 뜻이면 그렇게 명시해야 한다.
- **내 입장과 다르게 "합의"로 분류된 것**: `ColorReplaceLinear`의 최종 채택, `Normal` 제거, `avg_iris_luma` 1샘플 측정, detail low-light threshold `0.15`는 합의가 아니다.

## 2. Claude가 "완전 수렴"이라 분류한 7쟁점 재검증

### I3 — LTL realSpec 처리

- **동의/부동의**: 동의. `realSpec`는 삭제가 맞다.
- **세부 수식·값 검증**: `LTL 내부 realSpec 제거`는 맞다. 다만 §1 C5의 외부 반사 계층은 소스가 B2에서 결정되므로, 지금 확정할 수 있는 것은 `계층 구조`와 `realSpec 삭제`이지 특정 reflection source가 아니다.

### I5 — 블링크 ramp

- **동의/부동의**: 원칙은 동의. 즉시 off는 틀렸고, down/up 비대칭 ramp가 맞다.
- **세부 수식·값 검증**: 시간 범위 `down 50~80ms`, `up 100~120ms`는 내 R2와 일치한다. 하지만 `α_close=0.15`, `α_open=0.08`은 이 시간 범위를 보장하지 않는다. synthesis가 계수를 넣으려면 ms 목표와 일관된 값이어야 한다.

### I6 — 입력 계약

- **동의/부동의**: 동의. `EyeRenderPacket` 채택과 raw detector 의존 금지는 맞다.
- **세부 수식·값 검증**: 스키마는 대체로 맞다. 단 C11의 `occlusion` 언급과 실제 스키마가 불일치한다. `occlusion = visibility + aperture mask로 표현`인지, 별도 optional인지 정리해야 한다.

### I7 — avg_iris_luma 하드코드

- **동의/부동의**: 하드코드 제거는 동의. 하지만 synthesis의 측정 수식에는 부동의.
- **세부 수식·값 검증**: 내 R2는 ROI 평균이었다. `textureLod` 중심 1샘플은 pupil 중심을 읽을 수 있어 LTL/CRL 정규화를 망칠 수 있다. C9는 `ROI 평균 또는 그 근사`로 수정해야 한다.

### I9 — 홍채 디테일 재주입

- **동의/부동의**: 원본 휘도 재주입 원칙은 동의.
- **세부 수식·값 검증**: `detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15)`와 `inner r<0.65`는 내 R2와 일치한다. `avg_iris_luma < 0.15` 임계값은 합의된 값이 아니다. 또한 `spec/reflection 제외` 조건이 빠졌다.

### I10 — W2-W3 경계

- **동의/부동의**: 동의. refiner는 W2, W3는 packet 소비자다.
- **세부 수식·값 검증**: W3가 `pupil_center` 같은 소비 필드를 정의하는 것은 맞다. 모델 구현을 W3에 넣으면 안 된다.

### I2-부분 — 3종 확정분

- **동의/부동의**: `TintLinearV2 / Multiply / ScreenLinear` 3종 확정에는 동의.
- **세부 수식·값 검증**: `ColorReplaceLinear`는 아직 확정분이 아니다. Claude 종합의 §5 표는 `블렌드 4종`을 합의로 썼는데, 내 R2 기준으로는 과장이다.

## 3. Claude가 "벤치 이관"이라 분류한 4쟁점 재검증

### I1 — 환경 반사

- **벤치 이관 판단**: 맞다. 말로 더 좁힐 수 없다.
- **벤치 매트릭스 검증**: `OFF / env-map-only / periphery-camera`, `4환경 × 2동작`은 적절하다. 다만 `face_region_mask`는 R2에서 합의된 구현 요소가 아니고 정의도 없다. 벤치 프로토타입 설명에서 빼거나 별도 미확정 구현으로 표기해야 한다.
- **벤치 없이 확정 가능 여부**: 불가. 나는 env-map-only를 여전히 선호하지만, periphery가 실제로 이기면 받아들일 수 있다.

### I2-잔여 — 4번째 슬롯

- **벤치 이관 판단**: 맞다.
- **벤치 매트릭스 검증**: `Normal / ColorReplaceLinear` 24클립은 방향이 맞다. 단 내 R2의 SKU에는 `화이트/그래픽 렌즈`가 포함됐는데 §2 B1은 `불투명 서클`로 축소했다. ColorReplaceLinear의 존재 이유가 색 정확도 높은 불투명/화이트/그래픽 셀이라 이 셀은 빠지면 안 된다.
- **벤치 없이 확정 가능 여부**: 불가. 내 R2에서 이미 `ColorReplaceLinear`는 Normal과 대결 후 채택이라고 수정했다.

### I4 — 림발 자동감지

- **벤치 이관 판단**: 부분 동의. `림발 기본 ON + 메타데이터 off`는 벤치 없이 확정 가능하다. 자동감지는 fallback 후보로만 벤치하면 된다.
- **벤치 매트릭스 검증**: 10개 SKU 정확도 벤치는 최소 검증으로는 가능하다. 하지만 자동감지는 false positive가 제품 품질을 바로 망치므로 `9/10`으로 일반 채택하는 것은 느슨하다. 이 결과라면 fallback only가 한계다.
- **벤치 없이 확정 가능 여부**: 메타데이터 only는 확정 가능하다. 자동감지는 벤치 없이는 채택하면 안 된다.

### I8 — sclera color-veto

- **벤치 이관 판단**: 맞다. Gemini의 luma-only와 내 color-veto는 말로 더 줄이기 어렵다.
- **벤치 매트릭스 검증**: `그레이/블루 + 다크브라운 × 3조명 × 2방식`은 적절하다.
- **벤치 없이 확정 가능 여부**: 원칙은 `geometry-first`로 이미 닫혔다. 남은 것은 `채도 포함 veto vs luma-only veto`라 벤치 이관이 맞다.

## 4. `99_claude_synthesis.md` §1 즉시 확정 17개 검증

### 제거 6개

- **D1 동의**: fixed light/분석 조명 블록은 제거 대상이다.
- **D2 동의**: `LIMBAL_ENABLED=false` 하드코드 제거는 맞다.
- **D3 동의**: LTL 내부 `realSpec` 제거는 맞다.
- **D4 동의**: `uAvgIrisLum=0.35` 하드코드 제거는 맞다.
- **D5 조건부 동의**: `3D Light` 토글은 fixed-light 토글이라 제거가 맞다. 환경 반사 토글을 별도로 만들지는 이 문서에서 새로 결정하지 않는다.
- **D6 부동의**: `Overlay`, non-linear `LuminanceTint`, `SoftLight` 제거는 맞다. `Normal` 제거는 B1 벤치 전에는 틀렸다.

### 추가 11개

- **C1 수정 필요**: 즉시 확정은 `TintLinearV2 / Multiply / ScreenLinear` 3종이다. `ColorReplaceLinear`는 후보 슬롯이다.
- **C2 동의**: `TintLinearV2`에서 realSpec 제거는 맞다.
- **C3 동의**: `ScreenLinear` 추가는 맞다.
- **C4 조건부 동의**: 수식은 내 R1 제안과 일치하지만 채택은 B1 결과 후다.
- **C5 조건부 동의**: 반사 계층 분리는 맞다. 반사 소스는 B2 결과 후다.
- **C6 부분 동의**: 림발 기본 ON + `has_baked_limbal` 메타는 맞다. 다만 `LensConfig`에 bool을 추가하면 공개 API 변경 가능성이 있으므로 "내부 SKU 메타데이터"인지 "공개 config"인지 분리해야 한다.
- **C7 수정 필요**: 시간 범위는 맞다. EMA 계수는 검증되지 않았고 목표 시간과 불일치한다.
- **C8 동의**: `EyeRenderPacket` 도입은 맞다.
- **C9 부동의**: 1샘플 `textureLod`는 내 R2와 다르다. masked ROI 평균이어야 한다.
- **C10 수정 필요**: 원본 detail reinjection은 맞다. `0.15` 임계값은 미확정이고, `spec/reflection 제외`가 빠졌다.
- **C11 동의**: W2-W3 경계 정리는 맞다. 단 `occlusion`이 스키마에 없다는 문서 불일치를 고쳐야 한다.

### EyeRenderPacket 스키마

- **대체로 동의**: 필수 필드와 optional 필드 방향은 맞다.
- **수정 필요**: `occlusion` 언급과 스키마가 불일치한다.
- **수정 필요**: `pupil_center_norm` 설명을 `parallax용`으로만 적은 것은 좁다. 내 R2의 핵심은 pupil cutout/동공 정렬도 포함한다.
- **수정 필요**: C9의 예외 문구는 "self-measure 가능"까지만 맞다. self-measure 방식은 1샘플이 아니라 ROI 평균이어야 한다.

## 5. `99_claude_synthesis.md` §3 구현 순서(S1~S5) 검증

- **브랜치 전략 평가**: 현재 로컬은 `feature/P5-W3-05`이고, `d09bf72`가 `70633ac` 위에 있다. W3-04 직전 기준점에서 새 브랜치를 딴 전략 자체는 적절하다. 고정 조명 실험을 별도 브랜치에서 버리고 새 방향을 적용하는 판단은 맞다.
- **S1 부동의**: `D1~D6 일괄 제거`는 너무 거칠다. 특히 `Normal`은 B1 벤치 대상이므로 S1에서 제거하면 벤치 자체가 무의미해진다.
- **S1 수정 필요**: 고정 조명, realSpec, 비활성 림발 하드코드는 제거해도 된다. 벤치 후보인 Normal과 ColorReplaceLinear 관련 분기는 벤치 전까지 남겨야 한다.
- **S2 부분 동의**: `EyeRenderPacket`, `TintLinearV2`, `ScreenLinear`, blink ramp, avg luma, detail reinjection 스캐폴드는 맞다. 단 C4는 벤치 전 최종 ID로 고정하지 말고 후보로 둬야 한다.
- **S2 수정 필요**: C9를 1샘플 측정으로 구현하면 안 된다. 이 단계에서 구현한다면 masked ROI 평균으로 구현해야 한다.
- **S3 동의**: B1/B2/B4/B8 벤치 실행 순서는 맞다.
- **S4 수정 필요**: "C10(sclera)"는 문서상 오기다. C10은 detail reinjection이고, sclera는 B8 결과 반영 항목이다.
- **S5 동의**: HIGH/MID/LOW 실기기 통합 테스트는 필요하다.
- **소요 시간 평가**: 14~20시간은 낙관적이다. 특히 B2 3프로토타입과 EyeRenderPacket 리팩터가 같은 2~3일 안에 끝날지는 코드 영향 범위 확인 전에는 확정하면 안 된다.

## 6. `99_claude_synthesis.md` §4 Phase 6 이월 8개 검증

- **동의**: 3D Face Geometry, HDR IBL, Full PBR, Neural rendering, corneal refraction, 속눈썹 전용 세그멘테이션은 W3-05 밖이 맞다.
- **동의**: Gemini의 pupil cutout 동적 처리 추가 지적은 W3-05 범위 밖이 맞다. 다만 `pupil_center` 입력 자체는 W3의 packet 계약에 남아야 한다.
- **동의**: head-pose 기반 env rotation의 실제 활용은 W2/refiner 출력 이후가 맞다. 단 optional field 준비는 W3에서 하는 것이 맞고, synthesis도 그렇게 적고 있다.
- **추가 반대 없음**: Phase 6 이월 목록 중 W3-05에 반드시 끌어와야 할 항목은 없다.

## 7. 내가 R2에서 놓쳤거나 약하게 표현했던 것

- **C9를 더 강하게 말했어야 한다**: `avg_iris_luma`는 단순한 값 교체가 아니라 블렌드 품질의 기준 신호다. 중심 1샘플은 내 R2의 의도를 망친다.
- **I2의 조건부를 더 강하게 못 박았어야 한다**: 나는 R2에서 `ColorReplaceLinear는 Normal과 1:1 대결 후`라고 썼다. Claude synthesis는 이 조건을 §2에 쓰면서도 §1에서는 4종 확정처럼 적어 자기모순을 만들었다.
- **D6의 위험을 더 명확히 써야 했다**: 벤치 후보를 S1에서 제거하면 벤치가 아니라 결론 선반영이다.
- **I9의 반사 제외 조건을 더 명확히 써야 했다**: detail reinjection이 reflection/spec 위에 곱해지면 반사 cue를 죽일 수 있다.

## 8. R4 필요성 판단

R4는 원칙적으로 필요 없다. 다만 `99_final_decision.md` 작성 전에 아래 4개 수정은 반드시 반영해야 한다.

- **필수 수정 1**: `ColorReplaceLinear`를 즉시 확정 4종으로 쓰지 말고, `3종 확정 + 4번째 슬롯 벤치`로 고친다.
- **필수 수정 2**: `Normal`을 S1/D6 제거 대상에서 빼고 B1 벤치 후보로 유지한다.
- **필수 수정 3**: `avg_iris_luma` 측정을 `textureLod 중심 1샘플`이 아니라 masked ROI 평균으로 고친다.
- **필수 수정 4**: C7 EMA 계수, C10 low-light threshold, occlusion 스키마 불일치, S4의 `C10(sclera)` 오기를 정리한다.

이 수정이 반영되면 R4 없이 벤치 이관으로 충분하다. 반영되지 않으면 R4가 아니라 synthesis 자체를 고쳐야 한다.
