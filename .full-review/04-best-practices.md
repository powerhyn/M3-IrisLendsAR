# Phase 4: Best Practices & Standards

## 대상: P4-W4-01b Soft Light 합성 전환

---

## Framework & Language Findings (9건)

### High (3건)

| ID | 항목 | 상세 |
|----|------|------|
| **GLSL-1** | **sRGB 전달함수 정확도** | `pow(x,2.2)`는 sRGB 규격 근사치. 실제 sRGB는 0.04045 이하에서 선형 세그먼트 사용. 어두운 피부톤에서 색상 왜곡. 정확한 `sRGBToLinear()` 또는 하드웨어 sRGB 권장. |
| **GLSL-2** | **하드웨어 sRGB vs 수동 pow()** | GLES 3.0+ `GL_SRGB8_ALPHA8` + `GL_FRAMEBUFFER_SRGB`로 셰이더 내 pow() 2회 완전 제거 가능. 성능(PERF-3)과 정확도 동시 해결. **가장 높은 ROI 단일 변경.** |
| **GPU-3** | **pow() 모바일 비용** | PERF-3 재확인. Mali-G7x에서 12-24 cycles. GLSL-2로 해결 가능. |

### Medium (3건)

| ID | 항목 | 상세 |
|----|------|------|
| **GLSL-4** | **Soft Light 수식 MAD 최적화** | 현재: `(1-2b)*a²+2b*a`. 개선: `a*(a+2b*(1-a))`. 곱셈 4회→3회, MAD 패턴 적합. |
| **CPP-1** | **매직 넘버 상수화** | `0.70f` → `constexpr float kMaxHighFreqAttenuation = 0.70f;` |
| **CPP-2** | **NaN/Inf 파라미터 검증** | `mapSkinQuality` 진입부 `std::isnan`/`std::isinf` 가드 추가 권장. (SEC-2와 동일) |

### Low (3건)

| ID | 항목 | 상세 |
|----|------|------|
| GLSL-3 | precision qualifier 세분화 | 마스크에 `mediump` 가능하나 실질 효과 미미. 현재 유지 합리적. |
| GPU-2 | 마스크 Y좌표 플립 위치 | vertex shader varying 전달이 깔끔하나 성능 차이 없음. |
| GLSL-5 | blend clamp과 gain loss 관계 | 의도적 설계. gain compensation 적용 시 재검토 필요. |

### 긍정적 평가

| 항목 | 판정 |
|------|------|
| Deprecated API | 없음 — `#version 310 es`, `texture()`, `in`/`out` 현행 표준 |
| GLES 3.1 호환성 | 양호 |
| C++17 활용 | 양호 — `std::clamp`, 구조화된 바인딩 적절 사용 |

---

## CI/CD & DevOps Findings

해당 없음 — SDK 프로젝트 미배포 상태, CI/CD 리뷰 제외.
