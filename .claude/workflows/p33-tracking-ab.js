export const meta = {
  name: 'p33-tracking-ab',
  description: 'IrisLensSDK ③-3 — MediaPipe Tasks 글루 이식 + JNI 주입 표면 + 데모 A/B 인프라 (설계: docs/workPaper/REFACTOR-3-3_plan.md)',
  whenToUse: '리팩토링 ③-3 실행 시. 사전 조건: refactor/p33-tracking-ab 브랜치 체크아웃 (plan §0)',
  phases: [
    { title: 'Build', detail: 'tracker-port(이식) ∥ jni-bridge(주입 표면) — 병렬 2' },
    { title: 'Integrate', detail: 'ab-demo: 토글 + 듀얼 비교 + 메트릭' },
    { title: 'Verify', detail: '게이트 재실행 + 좌표 규약 적대 검토 + 범위·불변 검토 — 병렬 3' },
    { title: 'Revise', detail: 'critical/important 반영 + 게이트 재검증' },
  ],
}

const ROOT = '/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK'
const PLAN = ROOT + '/docs/workPaper/REFACTOR-3-3_plan.md'
const ADR = ROOT + '/docs/decisions/0001-landmark-injection-tracking-replacement.md'
const LENSSIM = '/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator'

const COMMON = `당신은 IrisLensSDK ③-3(추적 교체 A/B) 작업자다. 저장소: ${ROOT}, 브랜치 refactor/p33-tracking-ab.

필독 (작업 전 반드시 정독):
1. ${PLAN} — 본 작업의 설계서. 섹션 번호(§)로 지시를 참조한다.
2. ${ADR} — §5(버전 고정 사유), §6(주입 계약), §7(좌표·시맨틱 계약 4종), §10(후퇴 트리거·A/B 주의), §12(게이트)

공통 규칙:
- ⚠️ 새 빌드 디렉토리 생성 금지 — cpp/cmake-build-debug 재사용 (-DIRIS_SDK_FETCH_TFLITE=OFF)
- plan §5의 주의 6항 엄수. 특히 §5-5: LEGACY(자체 추적) 경로 코드는 한 줄도 변경 금지
- 공개 C API는 추가만 허용(기존 시그니처 불변). 커밋 금지 — 구현+검증까지만
- tasks-vision은 0.10.35 외 금지 (latest.release 금지)
- 모든 서술 한국어, 코드 식별자 원문.`

const BUILD_SCHEMA = {
  type: 'object',
  required: ['summary', 'artifacts', 'decisions', 'verification', 'openIssues'],
  properties: {
    summary: { type: 'string' },
    artifacts: { type: 'array', items: { type: 'string' } },
    decisions: { type: 'string' },
    verification: { type: 'string', description: '직접 실행한 검증 명령+결과' },
    openIssues: { type: 'array', items: { type: 'string' } },
  },
}
const REVIEW_SCHEMA = {
  type: 'object',
  required: ['passed', 'assessment', 'issues'],
  properties: {
    passed: { type: 'boolean' },
    assessment: { type: 'string' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'severity', 'detail', 'suggestion'],
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'important', 'minor'] },
          location: { type: 'string' },
          detail: { type: 'string' },
          suggestion: { type: 'string' },
        },
      },
    },
  },
}
const REVISE_SCHEMA = {
  type: 'object',
  required: ['summary', 'changes', 'rejected', 'verification'],
  properties: {
    summary: { type: 'string' },
    changes: { type: 'array', items: { type: 'string' } },
    rejected: { type: 'array', items: { type: 'string' } },
    verification: { type: 'string' },
  },
}

function trim(s, n) { s = s || ''; return s.length > n ? s.slice(0, n) + '…' : s }

log('③-3 Build: tracker-port ∥ jni-bridge')

const builds = (
  await parallel([
    () =>
      agent(
        COMMON + `
당신은 ③-3 이식 작업자(tracker-port)다. plan §2를 그대로 수행하라:
1. demo-app build.gradle.kts에 tasks-vision:0.10.35 추가
2. plan §2 매핑 테이블의 자산 이식 (원본: ${LENSSIM}/sdk/android/lenssdk/) — 값 무변경, 패키지 치환만. 보류 표시된 파일(OneEuroFilter/CameraController)은 이식하지 마라
3. face_landmarker.task를 demo assets로 복사
4. 단위 테스트 이식 (CoordMapperTest, IrisGeometryTest) + demo-app 테스트 의존성 확인
5. 검증: ./gradlew :demo-app:compileDebugKotlin && :demo-app:testDebugUnitTest — 이식 테스트 통과 확인
주의: 이 단계에서는 기존 데모 코드(GpuRenderActivity 등)를 수정하지 마라 — 통합은 다음 단계(ab-demo)가 한다. 신규 파일 + gradle + assets만.`,
        { label: 'build:tracker-port', phase: 'Build', schema: BUILD_SCHEMA }
      ),
    () =>
      agent(
        COMMON + `
당신은 ③-3 주입 표면 작업자(jni-bridge)다. 모던 C++17. plan §3을 그대로 수행하라:
1. C API 신설: iris_get_injected_result(IrisResult* out) — sdk_api.h 선언 + landmark_injection 경유 구현 (readDerived 위임, 미주입 시 명시 에러). 기존 함수 무변경
2. JNI 3개 + Java 래퍼 (plan §3-2 시그니처) — iris_jni.cpp는 기존 패턴(JniCache, copyResultToJava) 재사용, 478*3 길이 이중 가드
3. C++ E2E 테스트: cpp/tests/test_landmark_injection.cpp에 'C API 경유 왕복' 케이스 추가 (골든 baseline JSON 478점 → iris_set_landmarks → iris_get_injected_result → detector 파생값 ε 비교)
4. 검증: cmake 빌드(신규 경고 0) + test_landmark_injection 통과 + 골든 게이트(plan §6-3) exit 0 + ./gradlew :iris-sdk:compileDebugKotlin(JNI 컴파일 — NDK)
주의: demo-app 파일은 건드리지 마라 (tracker-port와 파일 분리).`,
        { label: 'build:jni-bridge', phase: 'Build', schema: BUILD_SCHEMA, agentType: 'systems-programming:cpp-pro' }
      ),
  ])
).filter(Boolean)

if (builds.length < 2) {
  return { builds: builds, error: 'Build 단계 미완(한도/중단 가능) — 트랜스크립트 수확 후 연속 실행 필요 (safety-workflow-checkpoint)' }
}

log('Integrate: ab-demo')
const buildCtx = builds.map(b => `산출물 ${JSON.stringify(b.artifacts)} / 판단 ${trim(b.decisions, 600)} / 미해결 ${JSON.stringify(b.openIssues)}`).join('\n')

const integrate = await agent(
  COMMON + `
당신은 ③-3 A/B 통합 작업자(ab-demo)다. 선행 작업 완료 상태:
${buildCtx}

plan §4를 그대로 수행하라:
1. §4.1 공급자 토글 — TrackerMode(LEGACY/TASKS) + UI 토글, TASKS 경로는 FaceTracker→TasksToIrisResult.kt(신설, §4.1 변환 계약 전부: sensorToUpright/홍채 5점 순서/radius 픽셀 환산/EAR/confidence 1.0)→updateDetectionSlot. LEGACY 경로 무변경(§5-5)
2. §4.2 듀얼 비교 모드 — Tasks IMAGE 모드 동기 호출, AB_METRIC 구조화 로그 + 누적 요약 + HUD 1줄, rot0 중심
3. §5 주의 1~3 엄수 (스레드 친화성 보존, RGBA 경로 결정+비용 분리 표기, 이중 필터 금지)
4. 검증: ./gradlew :demo-app:assembleDebug 성공 + 토글/측정 모드 코드 경로 정적 점검 + git diff로 LEGACY 경로 무변경 확인
보고의 decisions에 RGBA 스트림 방식 선택(별도 use case vs NV21 변환)과 그 비용 표기 방법을 명시하라.`,
  { label: 'integrate:ab-demo', phase: 'Integrate', schema: BUILD_SCHEMA }
)
if (!integrate) {
  return { builds: builds, integrate: null, error: 'Integrate 미완 — 수확 후 연속 실행 필요' }
}

log('Verify: 게이트 + 좌표 규약 + 범위·불변 (병렬 3)')
const ctx = `\n\n구현 보고 요약:\n${buildCtx}\n통합: 산출물 ${JSON.stringify(integrate.artifacts)} / 판단 ${trim(integrate.decisions, 800)}`

const REVIEWS = [
  {
    key: 'gates',
    lens: `게이트 독립 재실행 (plan §6 전체) — 구현자 주장을 믿지 말고 직접: ① cmake 빌드+ctest 직렬(pre-existing 외 회귀 0) ② 골든 캡처+비교 exit 0 + 베이스라인 git diff 클린 ③ ./gradlew :demo-app:assembleDebug + :demo-app:testDebugUnitTest ④ 16KB 정렬 기록(§6-5) ⑤ tasks-vision 버전이 정확히 0.10.35인지. 실패 항목은 전부 critical.`,
  },
  {
    key: 'coord-contract',
    lens: `좌표 규약 적대 검토 — TasksToIrisResult.kt와 듀얼 비교 변환 체인을 ADR §7(4계약)과 함정 #13(원본 좌표계 반환→sensorToUpright 필수, 미러보다 회전 먼저)/#12/#10/#5(픽셀 환산 치수)에 대조하라. ${LENSSIM}/sdk/android/lenssdk/의 원본 FaceTracker.kt/CoordMapper.kt와 이식본을 diff해 '값 무변경 이식' 위반을 찾아라. 홍채 5점 순서(§7.0 right→top→left→bottom), 듀얼 비교 메트릭의 좌표 공간 일치(양쪽 다 upright 픽셀인지), 이중 필터 여부를 코드로 확인. 좌표 계약 위반은 critical(A/B 무효).`,
  },
  {
    key: 'scope-invariance',
    lens: `범위·불변 검토 — git diff 전수: ① LEGACY 추적·렌더 경로 기존 코드 무변경(plan §5-5 — 위반 critical) ② 코어 cpp 변경이 plan §3 C API 추가뿐인지 ③ 공개 헤더 기존 선언 변경 0 ④ 데모 정화 W 산출물(스냅샷 복사/HUD/debugMode) 보존 ⑤ 이식 보류 파일(OneEuroFilter/CameraController)이 들어오지 않았는지 ⑥ plan §9 이월 항목을 침범하지 않았는지.`,
  },
]

const reviews = (
  await parallel(REVIEWS.map(r => () =>
    agent(COMMON + '\n당신은 적대적 검증자다.\n검증 렌즈: ' + r.lens + ctx + '\n\ncritical = 머지 차단. 직접 확인한 것만 issues에 넣어라.', { label: 'review:' + r.key, phase: 'Verify', schema: REVIEW_SCHEMA })
  ))
).filter(Boolean)

const allIssues = reviews.flatMap(r => r.issues || [])
const actionable = allIssues.filter(i => i.severity !== 'minor')
log('검증 완료: critical/important ' + actionable.length + '건')

let revise = null
if (actionable.length > 0) {
  revise = await agent(
    COMMON + `\n당신은 ③-3 수정 작업자다. 이슈를 반영하라:\n${JSON.stringify(actionable)}\n\n산출물: ${JSON.stringify(builds.flatMap(b => b.artifacts).concat(integrate.artifacts))}\n\ncritical 필수 반영, important는 부당하면 rejected에 사유. 수정 후 plan §6 게이트(빌드/ctest/골든/gradle 테스트) 재검증 필수. LEGACY 무변경 원칙 유지. 커밋 금지.`,
    { label: 'revise:p33', phase: 'Revise', schema: REVISE_SCHEMA, agentType: 'systems-programming:cpp-pro' }
  )
}

return {
  builds: builds,
  integrate: integrate,
  reviews: reviews.map(r => ({ passed: r.passed, assessment: trim(r.assessment, 700), issues: r.issues })),
  actionable: actionable,
  minor: allIssues.filter(i => i.severity === 'minor'),
  revise: revise,
  nextSteps: '커밋(메인 루프) → 실기기 설치 → plan §7 A/B 절차 안내 → 사용자 판정(T1/T2) → 통과 시 ④는 별도 승인',
}
