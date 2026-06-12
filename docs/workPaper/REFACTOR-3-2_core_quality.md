# REFACTOR-3-2: 코어 품질 리팩토링 (③-2)

| 항목 | 내용 |
|---|---|
| 상태 | ✅ 구현 완료 (2026-06-12) — B2 실기기 확인 항목 통과 후 머지 |
| 브랜치 | `refactor/p32-core-quality` (develop 453ce3d 기준) |
| 원칙 | **동작 불변** — 골든 통과 머지 조건. 출력을 바꾸는 정확성 수리는 ④/⑤로 이월 |
| 범위 | 감사 워크리스트 31건 중 ③-2 해당분 (데모 3건은 정화 W에서 기처리, DetectionSlot·Strategy 팩토리는 ④ 이월) |

## B1 — 스레드/수명 (f721657)

- [blocker] InferenceThread joinable 미회수 terminate, Starting-stop 행(CAS), detectSync 고착/UAF(deep-copy+세대 가드), async notify, log 락 통일, BufferPool scoped_lock, static 플래그 atomic화, 데드 설정 문서화
- **적대 검증이 important 2건 실증** (thread_ 레이스 TSan / caller 측 Stopping 소실 행 3/3 재현) → lifecycle_mutex_ + caller CAS 3곳으로 같은 배치에서 봉합
- 수명 테스트 6케이스 신설 (검증자 지적: fix2/3/4 회귀 테스트는 detector 주입 불가 구조 때문에 한계 — ④ 팩토리 도입 시 보강 권고)

## B2 — GL/GPU (532a152)

- applyTexture ping/pong 반환(이월 패턴)+안정 핸들, TexturePool ensureCapacity(상한 silent 실패 제거)+정확 크기 매칭+만석 LRU evict(적대 검증 지적), cleanup unit0 복귀, gles doneCurrent 가드, GPUProfiler in-flight 추적, uniform 캐시, 데드 코드 제거. 단위 테스트 13(신규 5)
- **보류(동작 불변 우선)**: 컨텍스트 차용 'current' 가드(④ iris_sdk_notify_gl_context_lost와 함께), glClientWaitSync 스톨(실기기 프로파일 필요), ROI 2패스(출력 기여 확인됨 — blit 대체는 실기기 검증 필요)
- ⚠️ GPU 가드 안 변경은 macOS 미컴파일 — **실기기 확인 목록** 아래 참조

## B3 — 계약 가드 (이 커밋)

- IrisResult/IrisLandmark **필드별 offsetof static_assert 22종** — sizeof 단일 가드가 못 잡던 순서 교환·타입 치환 차단. GLES 가드 밖 배치(macOS에서도 검증)
- applyWarp `landmark_count` 파라미터(기본 478) + kMinWarpLandmarkCount(474) 가드 — 468 배열 OOB 읽기(UB)를 명시 거부로 전환, 가드 테스트 신설
- grid_mesh 468 테이블: **478 확장은 의도적으로 보류** — 확장 시 조용히 탈락하던 iris center 등록이 살아나 워프 출력이 변함(동작 변화) → ⑤ geometry 수리에서. 주석으로 명문화

## 게이트 (배치마다 수행)

- 골든 37/37 PASS ×3회 (베이스라인 무수정), ctest 직렬 회귀 0 (실패 5건 pre-existing — 병렬 3건 추가 실패는 stash 대조로 기존 플레이크 확정), 공개 C API 시그니처 불변

## 실기기 확인 목록 (B2 — 머지 전 사용자 확인)

1. beauty ON에서 색/LUT 출력이 이전과 동일한지 (LUT 토글, whitening 포함) — cleanup 변경
2. beauty ON↔OFF 빠른 토글 시 정상 전환 (풀 정리 호이스팅)
3. 로그에 "capacity raised" 미출현 (960x720 기준 — 출현 시 보고)
4. 렌즈+뷰티 동시 동작 평소와 동일, 프레임 드랍 없음

## 이월 (사유와 함께)

- ④: DetectionSlot 재설계(주입 API 공식화), Strategy 팩토리 제거, IrisResult 단일 정의 통합, 에러 코드 정본 단일화, BufferPool 표면 정리, context loss 훅, 모델 에셋 수명주기(추적 외부화로 소멸 — "고치지 말고 삭제")
- ⑤: grid_mesh 478 확장(워프 출력 변화 동반), BlazeFace 정규화/디코딩, EAR 종횡비, left/right 명명, 타원 aspect, One-Euro 단위 등 정확성 수리 전부
- 별도: clientWaitSync/ROI 2패스 (실기기 프로파일 트랙)

## 변경 이력

- 2026-06-12: B1~B3 구현, 적대 검증 2회(important 5건 추가 발견·봉합), 골든 3회 PASS
