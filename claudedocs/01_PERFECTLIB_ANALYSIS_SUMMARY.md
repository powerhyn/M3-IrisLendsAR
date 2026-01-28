# PerfectLib vs IrisLensSDK 경쟁사 분석 총평

**작성일**: 2026-01-21
**분석 에이전트**: c4-architecture, system-architect, architect-review (x2)

---

## 1. 경쟁사(PerfectLib) SDK 개요

### 1.1 규모 및 구성

| 모듈 | 크기 | 클래스 | 네이티브 라이브러리 | 역할 |
|------|------|--------|------------------|------|
| Core | 3.3MB | 1,750 | MNN, libperfect | SDK 인프라, AI 추론 |
| Makeup | 6.4MB | 589 | libvenus (5.5MB) | 메이크업 렌더링 |
| FaceTracking | 5.2MB | 1 | libvenus_tracking (3.0MB) | 얼굴 추적, 스무딩 |
| ProductHandler | 1.5MB | 2,170 | 없음 | 제품 관리, VTO |
| HandlerCore | 0.15MB | 194 | 없음 | SQLite 데이터 관리 |
| **합계** | **16.55MB** | **4,704** | **~11MB** | - |

### 1.2 핵심 기술 자산

| 기술 | 설명 | 평가 |
|------|------|------|
| **Venus 엔진** | 독자 GPU 렌더링 엔진 (24+ 필터) | 업계 최고 수준 |
| **MNN Framework** | 모바일 최적화 딥러닝 추론 | TFLite 대비 경량 |
| **FaceAlignMotionSmoother** | 시간적 일관성 보장 | 핵심 차별화 요소 |
| **106+ 랜드마크** | 정밀 얼굴 추적 | 업계 표준 |
| **Liquify Warp** | 18 파라미터 얼굴 재형 | 고급 기능 |

---

## 2. 자사(IrisLensSDK) 현황

### 2.1 현재 구현 상태 (Phase 1)

| 컴포넌트 | 상태 | 구현 수준 |
|----------|------|----------|
| C++ Core Engine | 완료 | MediaPipe 기반 검출 |
| Android JNI | 완료 | 실시간 처리 17-19fps |
| LensRenderer | 기본 | CPU 기반 오버레이 |
| MotionSmoother | 부분 | Android 레이어만 |
| iOS/Flutter/Web | 미구현 | Phase 2/3 예정 |

### 2.2 기술 스택

```
Application Layer
    ↓
Binding Layer (JNI/Obj-C++/FFI/WASM)
    ↓
C API Layer
    ↓
C++ Core Engine (SDKManager, IrisDetector, LensRenderer)
    ↓
Third Party (MediaPipe, OpenCV, TFLite)
```

---

## 3. 종합 비교 평가

### 3.1 아키텍처 품질 점수

| 평가 항목 | PerfectLib | IrisLensSDK | 우위 |
|----------|------------|-------------|------|
| 모듈화 수준 | 6/10 | 9/10 | IrisLensSDK |
| API 설계 | 7/10 | 8/10 | IrisLensSDK |
| 에러 처리 | 6/10 | 8/10 | IrisLensSDK |
| 테스트 용이성 | 4/10 | 9/10 | IrisLensSDK |
| 유지보수성 | 5/10 | 9/10 | IrisLensSDK |
| SOLID 준수 | 6/10 | 9/10 | IrisLensSDK |
| **설계 총점** | **5.7/10** | **8.7/10** | **IrisLensSDK** |

### 3.2 기능/성능 점수

| 평가 항목 | PerfectLib | IrisLensSDK | 우위 |
|----------|------------|-------------|------|
| 기능 범위 | 10/10 | 3/10 | PerfectLib |
| 렌더링 품질 | 10/10 | 5/10 | PerfectLib |
| 실시간 성능 | 10/10 | 6/10 | PerfectLib |
| 모션 스무딩 | 10/10 | 4/10 | PerfectLib |
| 플랫폼 지원 | 6/10 | 4/10 | PerfectLib |
| **기능 총점** | **9.2/10** | **4.4/10** | **PerfectLib** |

### 3.3 확장성 평가

| 시나리오 | PerfectLib | IrisLensSDK |
|----------|-----------|-------------|
| 검출 알고리즘 추가 | 중간 | 매우 우수 (Strategy 패턴) |
| 렌더링 기능 추가 | 우수 (Venus 엔진) | 보통 |
| 플랫폼 확장 | 보통 (Android 특화) | 매우 우수 (C API) |
| 비즈니스 통합 | 매우 우수 | 미흡 |
| 성능 최적화 | 우수 (GPU 파이프라인) | 보통 |

---

## 4. 핵심 발견사항

### 4.1 PerfectLib 강점

1. **완성도 높은 제품**: 4,704 클래스, 16.5MB 규모의 성숙한 SDK
2. **GPU 기반 렌더링**: Venus 엔진의 24+ OpenGL ES 필터
3. **모션 스무딩**: FaceAlignMotionSmoother로 떨림 없는 추적
4. **비즈니스 레이어**: ProductHandler를 통한 SKU/VTO 관리
5. **풍부한 기능**: 메이크업, 주얼리, 네일, 헤어 등 전 영역 지원

### 4.2 PerfectLib 약점

1. **God Module**: ProductHandler가 전체의 44% (2,170 클래스)
2. **높은 결합도**: 모듈 간 의존성이 복잡
3. **플랫폼 종속**: Android 특화 설계
4. **난독화**: 내부 로직 분석/학습 불가

### 4.3 IrisLensSDK 강점

1. **클린 아키텍처**: Strategy, pImpl, Factory 패턴의 적절한 조합
2. **테스트 친화적**: 인터페이스 기반 설계로 모킹 용이
3. **크로스플랫폼**: C API로 모든 플랫폼 바인딩 통일
4. **확장성**: DetectorType enum으로 새 검출기 쉽게 추가
5. **경량화**: 목표 SDK 크기 20MB 이하

### 4.4 IrisLensSDK 약점

1. **성능 부족**: 17-19fps (목표: 30+)
2. **모션 스무딩 부재**: Face Mesh 떨림 발생
3. **CPU 렌더링**: GPU 파이프라인 미구현
4. **제한된 기능**: 홍채 렌즈만 지원

---

## 5. 전략적 권고

### 5.1 단기 전략 (Phase 2)

**PerfectLib의 모든 기능을 따라가지 말 것.**

대신 **홍채 렌즈 피팅**이라는 핵심 기능에서 동등 이상의 품질을 달성하고,
**경량화와 크로스플랫폼 지원**으로 차별화.

| 우선순위 | 작업 | 목표 |
|---------|------|------|
| P0 | 모션 스무딩 | 떨림 90% 감소 |
| P0 | GPU 렌더링 | 30fps+ 달성 |
| P1 | HybridDetector | 검출 안정성 |
| P2 | iOS 지원 | 크로스플랫폼 |

### 5.2 중기 전략 (Phase 3)

| 우선순위 | 작업 | 설명 |
|---------|------|------|
| P0 | Eye-Only 모델 | MediaPipe 한계 극복 |
| P1 | 다중 렌즈 효과 | 다양한 렌즈 지원 |
| P2 | Flutter Plugin | 개발자 접근성 향상 |

### 5.3 차별화 포지셔닝

| 항목 | PerfectLib | IrisLensSDK 목표 |
|------|-----------|-----------------|
| 포지션 | 엔터프라이즈 AR SDK | 경량 홍채 렌즈 SDK |
| 타겟 | 대기업 뷰티 앱 | 중소기업, 인디 개발자 |
| 가격 | 고가 라이선스 | 오픈소스/저가 |
| 강점 | 기능 완성도 | 경량화, 확장성, 크로스플랫폼 |

---

## 6. 결론

### 설계 품질 평가
> **IrisLensSDK는 아키텍처 설계 품질 면에서 PerfectLib보다 우수합니다.**
> - SOLID 원칙 준수도: 9/10 vs 6/10
> - 테스트 용이성: 9/10 vs 4/10

### 기능 성숙도 평가
> **PerfectLib는 기능과 성능 면에서 현저히 앞서 있습니다.**
> - 기능 범위: 10/10 vs 3/10
> - 실시간 성능: 10/10 vs 6/10

### 최종 평가
IrisLensSDK는 견고한 아키텍처 기반 위에 구축되어 있으며, 핵심 기능(모션 스무딩, GPU 렌더링)을 보강하면 특정 사용 사례(홍채 렌즈 피팅)에서 충분한 경쟁력을 확보할 수 있습니다.

**핵심 메시지**: 전면적 경쟁보다 **틈새 시장에서의 우위 확보**가 현실적인 전략입니다.
