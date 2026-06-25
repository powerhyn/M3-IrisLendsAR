# P8-W4 ② 형태워프 브레인스토밍 R1 — 종합

> 2026-06-25. 참여: **Codex(gpt-5, codex exec) + Claude**. ⚠️ **Gemini 불참**(GEMINI_API_KEY/auth 미설정, exit 41). [[multi-ai-orchestration-bias]] 기준 "다른 모델(Codex) 검토" 충족하나 3자 아님 — 2자 합의로 명시. 필요 시 사용자가 Gemini auth 설정 후 R1-b 추가 가능.

## 결과: 4/5 완전 합의, 1 부분(사용자 판단)

### ✅ 닫힌 쟁점 (Codex=Claude 합의 → §5 확정 이동 제안)

**쟁점1 아키텍처 = (A) fragment-direct RBF**
14 제어점을 uniform으로, 컴포지트 셰이더에서 인버스 워프 직접 평가(skin/radiance composite와 동형). grid_mesh 4결함 수리는 RBF 재구현이라 이득 없음 + 메시/468-478 부채. ⚠️ grid_mesh는 **미사용 dead substrate로 남김**(kickoff 불변식 #2 "삭제 금지" 준수, ②=grid_mesh 암묵가정은 뒤집힘).

**쟁점2 파이프라인 = (b) 뷰티 composite(렌즈 후)에서 눈높이0 워프**
재배열 불필요(저위험) + 눈높이 변위≈0이 렌즈(눈높이) 무영향 보장 + 추가 FBO/패스 회피(기존 풀스크린 composite 합류). full: 카메라→렌즈→[skin→warp]→화면. 핸드오프 "피부→워프" 순서 유지.

**쟁점4 좌표공간 = CPU 픽셀 uniform 산출 + 셰이더 평가**
제어점 [cx,cy,dx,dy]·σ·bounds를 CPU에서 **디스플레이 픽셀공간**(viewportPx) 산출 → uWarp[14] uniform. 셰이더는 vUv×viewportPx로 RBF만 평가(정규화 비등방 회피=audit #1c/#4 교훈). **미러/회전은 단일 변환 헬퍼**에서 처리(skin mask x→1-x 선례, 좌표계약 안정). 제어점은 워프 전 원본 oval(이마확장 미적용) + One-Euro FACE_OVAL.

**쟁점5 테스트 = 핸드오프 5종 + 대칭 + roll 2종**
변위방향/strength≤0/눈높이<5%/3σbbox/퇴화방어 + 좌우 미러 대칭 + roll 등변(=audit #3 face_width 유클리드 수정 게이트). 합성 타원 얼굴 C++ GoogleTest.

### 🔶 부분 합의 (사용자 판단)

**쟁점3 SDK 표면 매핑**
- 합의: slim_face/thin_chin → V라인 워프(검증 알고리즘), **공개 config 필드 + iris_sdk_apply_face_warp 구동**(internal API는 보조 A/B용만).
- Codex 매핑식: `jawStrength = clamp(max(thin_chin, 0.75*slim_face), 0, 1)`(additive 폭주 방지). Claude: slim_face=메인 강도, thin_chin=하단(턱끝) 제어점 가중. → 구현 시 세부 확정(둘 다 slim/chin을 V라인으로 합성하는 방향 동일).
- **갈림**: `enlarge_eyes` 타이밍.
  - **Codex**: 같은 fragment 평가부에 **별도 radial warp 항으로 이번에 포함**.
  - **Claude**: enlarge_eyes는 검증 핸드오프 없음 + audit #4 결함 + 별도 메커니즘 → **별도 슬라이스(P8-W5)로 deferred**, 본 W는 V라인(slim/chin)만.

## 공통 최대 리스크 (Codex=Claude 일치)
**좌표 미러/회전 픽셀공간 정합** — 정규화 랜드마크를 현재 렌더 텍스처의 미러/회전 픽셀공간으로 일관 변환하는 것. + 눈높이 변위≈0의 실제 렌즈 정합(Claude). → **단일 변환 헬퍼 + 눈높이<5% 단위테스트 + 실기기 렌즈정합**이 핵심 게이트. 깨지면 쟁점2를 (a) 렌즈 전 워프로 후퇴.

## 제안 다음 단계
1. 닫힌 쟁점 1/2/4/5 → §5 확정 이동.
2. 쟁점3 `enlarge_eyes` 타이밍 사용자 결정(이번 포함 vs P8-W5 분리). **Claude 권고=분리**(검증된 V라인 먼저).
3. 구현 착수(ar-lens-implement 또는 직접) — 브랜치 결정(feature/p8-warp).
</content>
