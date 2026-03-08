# Phase 2: Security & Performance Review

## Security Findings

### High (2건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| S1 | std::stoi 예외 미처리 | `gpu_beauty_backend.cpp:1173, 1188` | GPU 렌더러 문자열 파싱 시 int 범위 초과 → std::out_of_range 예외 → SDK 초기화 크래시. `std::strtol` 또는 try-catch로 교체 필요 |
| S2 | detectDeviceTier() static public + GL 의존 | `gpu_beauty_backend.h:249` | static public이라 GL 컨텍스트 없이 외부 호출 가능. glGetString() UB 발생. private 인스턴스 메서드로 변경 필요 |

### Medium (5건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| S3 | One Euro Filter release() 미리셋 | `gpu_beauty_backend.h:446-448` | release() 후 재초기화 시 stale state로 첫 프레임 ROI 왜곡 |
| S4 | device_tier_ release() 미리셋 | `gpu_beauty_backend.h:451` | 기본값 HIGH로 남아 부분 접근 경로에서 오분류 가능 |
| S5 | ROI face_rect 직접 수정 → Scissor 타이밍 불일치 | `gpu_beauty_backend.cpp:1631-1632` | Scissor가 temporal filtering 이전에 설정되어 mask 중심과 scissor 영역 불일치 |
| S6 | Viewport 복원 RAII 미적용 | `gpu_beauty_backend.cpp:1259, 1303` | 현재 에러 경로에선 문제없으나, 향후 early return 추가 시 viewport half-res 잔류 위험 |
| S7 | OneEuroFilter thread safety 주석 부재 | `gpu_beauty_backend.h:446-448` | mutex_ 하에서만 접근 가능함을 명시하는 주석 필요 |

### Low (3건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| S8 | GPU 렌더러 문자열 파싱 휴리스틱 한계 | `gpu_beauty_backend.cpp:1162-1207` | Mali-G78(HIGH급)이 MID로 분류, 미인식 GPU는 LOW |
| S9 | half_w/half_h 홀수 해상도 오프셋 | `gpu_beauty_backend.cpp:1221-1222` | 시각적 영향 미미. 기존 방어 코드 충분 |
| S10 | LOGD 매크로 do-while 미적용 | `gpu_beauty_backend.cpp:23-27` | if-else 내 dangling-else 가능. do-while(0) 래핑 권장 |

### Positive Observations

- 방어적 null 체크 (glGetString 반환값)
- 텍스처 할당 실패 시 정리 및 false 반환
- half_w/half_h 최소값 가드 (< 1 체크)
- blur_radius 하한값 std::max(3, ...) 적용
- Mutex 일관성 (모든 public 메서드)
- Copy/Move 삭제 (Rule of Five)
- Scissor 교집합 기반 ROI 안전 처리

---

## Performance Findings

### High (1건)

| # | 이슈 | 추정 영향 | 설명 |
|---|------|-----------|------|
| P1 | 정적 DeviceTier - 열 스로틀링 미대응 | 과열 시 프레임 드롭 | GPU 클럭 동적 하강 시에도 HIGH tier full-res 유지. 런타임 적응형 tier 전환 필요 (P4-W3-05) |

### Medium (3건)

| # | 이슈 | 추정 영향 | 설명 |
|---|------|-----------|------|
| P2 | 중복 glTexParameteri 4회/프레임 | <0.05ms | TexturePool 기본값과 동일한 GL_LINEAR 재설정. 제거 + 주석 권장 |
| P3 | Half-res 텍스처 풀 모니터링 부재 | 추가 6.22MB | full-res + half-res 혼재 시 풀 크기 증가. PoolStats 로깅 권장 |
| P4 | Viewport 에러 경로 안전성 | 현재 0 (잠재적) | RAII ViewportGuard 고려 |

### Low (4건)

| # | 이슈 | 추정 영향 | 설명 |
|---|------|-----------|------|
| P5 | 중복 glUseProgram 1회/프레임 | <0.01ms | 가독성 위해 유지 합리적 |
| P6 | One Euro Filter 메모리 | 144 bytes | 무시 가능 |
| P7 | detectDeviceTier() 파싱 비용 | 0.01ms (1회) | initialize()에서만 호출, 캐싱됨 |
| P8 | roi_ptr 직접 변이 | 잠재적 리스크 | 로컬 복사본 고려 |

### Frame Budget Analysis

**MID Tier 1080p 기준 (Adreno 6xx)**:
- FreqSep Half-Res: ~8.9ms
- 후속 패스 (Combined Color, Masking, etc.): ~6.0ms
- **총 파이프라인: ~14.9ms** (33ms 버짓의 45%)

**대역폭 절감**: HIGH 대비 **-54.6%** (91.3MB → 41.5MB/프레임)
**GPU 메모리 절감**: HIGH 대비 **-75%** (24.88MB → 6.22MB)

### Positive Design Patterns

- detectDeviceTier() 1회 호출 + 결과 캐싱
- Gaussian weights CPU 사전 계산
- 셰이더 전환 최소화 (5패스 중 1회)
- Bilinear 하드웨어 보간으로 별도 업샘플링 패스 불필요
- TexturePool acquire/release로 GPU 메모리 할당 오버헤드 제거
- Bilateral fallback 안전망
- GL_LINEAR 필터 미복원 이슈: TexturePool 기본값이 GL_LINEAR이므로 **문제 아님** 확인

---

## Critical Issues for Phase 3 Context

1. **테스트**: detectDeviceTier()가 static + GL 의존이라 단위 테스트 불가. private 인스턴스 메서드로 변경 또는 테스트용 오버라이드 필요
2. **테스트**: std::stoi 예외 처리 로직의 에지 케이스 테스트 필요
3. **문서**: DeviceTier 분류 기준, One Euro Filter 파라미터 선택 근거 문서화 필요
4. **테스트**: 열 스로틀링 시나리오 재현 테스트 방법론 검토 필요
