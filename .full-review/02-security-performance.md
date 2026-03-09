# Phase 2: Security & Performance Review

## 대상: P4-W4-01b Soft Light 합성 전환

---

## Security Findings (5건)

### 해당 없음 (N/A) — 보안 위험 없음
| ID | 항목 | 판정 |
|----|------|------|
| SEC-1 | Shader Injection | 안전 — 셰이더 소스가 R"glsl()" 컴파일타임 고정, 런타임 삽입 경로 없음 |
| SEC-3 | 버퍼 오버플로우 | 안전 — uWeights[29] 고정 크기, radius std::clamp(1,28) |
| SEC-4 | 데이터 노출 | 안전 — 온디바이스 처리만, 네트워크 전송 경로 없음 |
| SEC-6 | Warp divergence (보안) | 안전 — 조건 분기 없음, uniform branch만 존재 |

### Low (1건)
| ID | 위치 | 요약 |
|----|------|------|
| SEC-2 | gpu_beauty_backend.cpp:995 | **NaN/Inf 입력 가드 누락**: `mapSkinQuality`에서 `skin_quality`가 NaN/Inf일 때 `std::clamp` 결과가 미정의 → `blur_radius`에 쓰레기 값 전파 가능. `if (std::isnan(skin_quality) || std::isinf(skin_quality))` 가드 추가 권장. |

### Shader Numeric Safety
| 항목 | 판정 |
|------|------|
| pow(orig, 2.2) | orig ∈ [0,1] 보장 → 결과 [0,1] ✅ |
| pow(max(result,0), 1/2.2) | max로 음수 방어 → GLSL UB 차단 ✅ |
| clamp(0.5+adjusted_high) | 필수 — adjusted_high ∈ [-1,1] 가능 → blend ∈ [-0.5,1.5] 방지 ✅ |
| Soft Light 출력 범위 | blend,base ∈ [0,1] → 결과 항상 [0,1] ✅ |

---

## Performance Findings (7건)

### Critical/High (2건)

| ID | 심각도 | 항목 | 상세 |
|----|--------|------|------|
| **PERF-4** | **HIGH** | **8-bit linear 양자화 밴딩 (P1 이슈)** | `texture_pool.cpp:337`에서 중간 버퍼를 GL_RGBA/GL_UNSIGNED_BYTE로 생성. Linear 색공간에서 8-bit는 어두운 영역(value<0.1)에서 ~25단계만 사용 가능 → ~4-8% 밝기 점프, 포스터라이제이션 발생. |
| **PERF-3** | **MED-HIGH** | **pow() 2회 비용** | sRGB↔Linear 변환에 `pow(x,2.2)`/`pow(x,1/2.2)` 사용. `pow`는 `exp2(y*log2(x))`로 확장되어 vec3 기준 6개 SFU 호출. MID tier GPU(Mali-G7x)에서 12-24 cycles, Composite 패스의 40-60% 차지 가능. |

### Medium (2건)

| ID | 심각도 | 항목 | 상세 |
|----|--------|------|------|
| PERF-2 | MEDIUM | Texture fetch 병목 | 4 texture fetch/fragment, 1080p@30fps 기준 ~1GB/s 텍스처 대역폭. 다만 Gaussian blur 패스가 실제 병목이므로 Composite 패스는 상대적으로 가벼움. |
| PERF-4b | MEDIUM | GL_RGBA16F 전환 비용 | RGBA16F 시 메모리 2x(24.9→49.7MB), 대역폭 2x. 대안: GL_R11F_G11F_B10F는 동일 4바이트/픽셀로 메모리 증가 없이 정밀도 개선 가능. |

### Low (3건)

| ID | 심각도 | 항목 | 상세 |
|----|--------|------|------|
| PERF-1 | LOW | Soft Light ALU 비용 | Additive(1 add) → Soft Light(7-8 ops). GPU별 +1-2 cycles. Texture fetch latency에 숨겨짐. **영향 무시 가능.** |
| PERF-5 | LOW | Register pressure | ~8-10 vec4 레지스터. Mali-G7x 64 vec4, Adreno 128 vec4 대비 점유율 영향 없음. |
| PERF-7 | LOW | clamp() 필요성 | 입력 범위상 필수(adjusted_high ∈ [-1,1] 가능). 제거 불가. |

### 성능 영향 종합

| GPU 아키텍처 | Soft Light ALU 추가 | pow() 비용 | 30fps 유지 |
|-------------|---------------------|-----------|-----------|
| Mali-G710+ (Valhall) | +1-2 cycles | 6-12 cycles | ✅ |
| Mali-G7x (Bifrost) | +1-2 cycles | 12-24 cycles | ✅ (여유 감소) |
| Adreno 7xx | +1 cycle | 6 cycles | ✅ |
| Adreno 6xx | +1-2 cycles | 6-12 cycles | ✅ |
| PowerVR Rogue | +2 cycles | 12-18 cycles | ✅ (여유 감소) |

---

## 권장 수정 우선순위

### 즉시 (P4-W4 내)
1. **PERF-4**: TexturePool에 `internal_format` 파라미터 추가, FreqSep 중간 버퍼를 `GL_R11F_G11F_B10F`로 전환 (메모리 증가 없이 밴딩 해소)
2. **SEC-2**: mapSkinQuality 진입부에 NaN/Inf 가드 1줄 추가

### 단기
3. **PERF-3**: pow(x,2.2)/pow(x,1/2.2)를 다항식 근사로 교체 → MID tier GPU에서 Composite 패스 ~30-50% 절감

### 장기
4. **PERF-4 확장**: DeviceTier 기반 HIGH=RGBA16F / MID=R11F_G11F_B10F 분기
5. **PERF-3 Option B**: 파이프라인 전체 linear 통일, 최종 출력에서만 sRGB 변환

---

## Critical Issues for Phase 3 Context

1. **P2 (gain 손실) 수정 시**: gain 보상 스케일러 도입 후 전 skinQuality 범위 시각적 검증 테스트 필요
2. **P1 (버퍼 포맷) 수정 시**: GL_R11F_G11F_B10F 지원 여부 디바이스별 테스트 필요 (GLES 3.0+ 필수)
3. **pow() 근사 적용 시**: sRGB↔Linear 변환 정확도 검증 테스트 필요 (max error < 0.5%)
4. **문서화**: Soft Light gain 특성, 버퍼 포맷 결정 근거 문서화 필요
