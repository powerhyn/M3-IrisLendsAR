# Phase 7 R1 브레인스토밍 - Gemini Response

**작성일**: 2026-06-04
**모델**: Gemini (gemini-3-flash-preview)
**상태**: Phase 7 설계 제안 완료

---

## 1. 핵심 질문 답변 (Q1~Q4)

### Q1. 0x501 즉시 수정 vs 디바이스 매트릭스 회귀 우선
- **추천안**: **(a) textureLod 즉시 패치 + 회귀 검증 묶기 (단일 W)**
- **근거**: F2를 통해 확인된 GLSL ES 3.0 spec §8.9 위반은 기기 특성을 타는 '버그' 이전에 '표준 미준수' 사항입니다. `textureLod`를 통한 명시적 LOD 지정은 부작용이 매우 적고 Adreno 전 세대에 걸쳐 안정성을 보장하는 검증된 회피책이므로, 패치를 선행하고 P7-W1에서 HIGH tier 기기(S23+)를 포함한 회귀 검증을 동시에 수행하는 것이 효율적입니다.

### Q2. W5 Phase C 알고리즘 선택
- **추천안**: **(c) Multi-scale Laplacian (sclera 영역 attenuation) 우선 적용**
- **근거**: F1에서 제안된 Oklab은 색상 왜곡 방지에 탁월하나 모바일 GPU의 연산 예산(30fps) 관점에서 cube root 연산 등의 비용 검증이 필요합니다. 반면, Laplacian 기반 edge-aware 융합은 Phase 6에서 결정된 'luma-only' 방향성을 계승하면서도 흰자(sclera)의 과도한 휘도를 억제하는 데 더 직관적이고 가벼운 해법을 제공합니다. Oklab은 PoC 단계에서 별도 trade-off 평가 후 도입을 결정해도 늦지 않습니다.

### Q3. W6 avg_iris_luma 측정 패스 설계
- **추천안**: **(a) 별도 small ROI FBO 다운샘플 + (d) 매 N frame(N=5) 측정 및 EMA 평활화**
- **근거**: F3에 따라 `EXTERNAL_OES`에서의 직접 통계 추출은 불가능하므로 2D 변환은 필수입니다. 30fps 환경에서 매 프레임 전체 텍스처를 처리하는 것은 낭비이므로, 64x64 이하의 작은 ROI FBO로 다운샘플링한 뒤 5프레임 주기로 샘플링하여 CPU/GPU 부하를 분산시키고, 급격한 노출 변화를 막기 위해 EMA(Exponential Moving Average)를 적용하는 것이 가장 안정적입니다.

### Q4. P7 W 분할 우선순위 + 의존성
- **제안 분할**:
    - **P7-W1**: 0x501 수정(textureLod) + Adreno 6xx/7xx 회귀 검증 (P0)
    - **P7-W2**: ROI 기반 Luma 측정 패스 구현 (OES→2D FBO) (P1)
    - **P7-W3**: A그룹 Cleanup (UI 동기화, Deprecated API 제거, SKU 정정) (P1)
    - **P7-W4**: 흰자 빛남 수식 개선(Laplacian) + 저조도 gate 최종 연결 (P2)
    - **P7-W5**: MID/LOW tier 기기 확보 및 전체 회귀 검증 (P2)
- **의존성**: `W1 & W3` (독립) → `W2` → `W4` → `W5`
- **소요 추정**: 총 7.5 MD (W1: 1.0, W2: 2.0, W3: 0.5, W4: 2.5, W5: 1.5)

---

## 2. Stage 1 Deep-Research Finding (F1~F4) 비판적 검토

### F1. Oklab + Multi-scale Laplacian
- **검토**: 학술적 우수성은 명확하나, **[반증]** 모바일 GPU에서 `pow(x, 0.333)` 연산의 반복이 프레임 드랍을 유발할 수 있습니다. 특히 Multi-scale Laplacian은 텍스처 샘플링 횟수를 급격히 늘릴 수 있어 Adreno 6xx 대역폭 한계에 부딪힐 가능성이 큽니다. "성능 예산 내 구현 가능성"이 최대 변수입니다.

### F2. 0x501 = GLSL ES 3.0 spec §8.9 위반
- **검토**: **[수용]** Khronos 명문화 및 Mozilla/Unity 등의 사례로 볼 때 Adreno 드라이버의 dynamic branching 최적화 방식과 충돌할 근거가 충분합니다. Adreno 3xx 사례로 한정 짓기보다, 최신 칩셋에서도 발생 가능한 비결정적 오류의 근원으로 보고 즉시 제거해야 합니다.

### F3. EXTERNAL_OES → 2D FBO 변환 강제
- **검토**: **[수용]** 하드웨어 유닛 제약상 mipmap 생성이 불가하므로 통계 추출을 위한 2D 변환은 우회할 수 없는 경로입니다. 다만, 이 과정에서 발생하는 1프레임의 지연(latency)이 자동 노출 보정(W6) 로직의 진동(oscillation)을 유발하지 않는지 튜닝이 필요합니다.

### F4. Phase 8 face slimming substrate
- **검토**: **[인지]** MediaPipe 468 vertex 기반 anchor는 산업 표준이므로 P8 준비를 위해 이 구조를 따르는 것은 타당합니다. 단, P7에서는 이에 대한 인프라 준비보다 현재의 렌더링 퀄리티 결함(흰자 빛남) 해결이 비즈니스 우선순위상 높습니다.

---

## 3. P7 우선순위 및 의존성 요약

1. **W1 (Stability)**: 0x501 수정 및 HIGH tier 안정성 확보 (Critical)
2. **W2 (Infra)**: 실측 Luma 데이터 파이프라인 구축
3. **W3 (Debt)**: Phase 6 잔여 부채 및 Cleanup
4. **W4 (Quality)**: 흰자 빛남 억제 및 저조도 최적화
5. **W5 (Reach)**: 타겟 디바이스 확장 및 최종 승인
