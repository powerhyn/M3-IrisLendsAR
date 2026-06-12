# Golden Baseline — 동작 불변 회귀 기준선 (계획 2-2)

리팩토링 2단계에서 구축한 **골든 베이스라인**이다. 현재 코드의 ① 랜드마크/검출
출력(JSON) ② 렌더 출력(PNG)을 표준 입력 세트에 대해 결정적으로 캡처해 두고,
이후 "동작 불변"을 표방하는 모든 단계(특히 3단계 경계 도입·코드 이동)가
이 기준선과 비교해 회귀가 없음을 증명하는 데 쓴다.

> **골든의 목적은 정답성 증명이 아니라 "현재 동작의 동결"이다.** 아래
> [알려진 현재 동작](#알려진-현재-동작-결함-포함--고정-대상) 절이 보여주듯
> 현재 코드의 결함(회전 좌표 불일치, render 실패 등)이 그대로 기준선에
> 포함되어 있다 — 그 결함을 "고치는" 수정은 골든 FAIL이 **정상**이며,
> 베이스라인 재캡처 + 사용자 승인(의도적 변화 분리 기록, ADR-0001 §11)
> 대상이다.

> 데스크톱(macOS)에서는 GPU(GLES/EGL) 경로를 띄울 수 없으므로 **CPU 경로만**
> 골든 대상이다: `iris_sdk_detect_with_rotation`, `iris_sdk_render_lens`,
> `iris_sdk_apply_beauty_v2_c`(use_gpu=0).

## 디렉토리 구조

```
cpp/tests/golden/
├── README.md
├── inputs/                 # 표준 입력 세트(최대 변 960px)
│   ├── face_closeup.png        # 인물 클로즈업 (960x720, 640x480 업스케일)
│   ├── screenshot_640a.png     # 실기기 스크린샷 640 원본 계열 (960x720)
│   └── screenshot_720a.png     # 실기기 스크린샷 720 원본 계열 (960x540)
└── baseline/               # 캡처된 기준 산출물 (golden_capture_all.sh 결과)
    ├── <stem>__<variant>.result.json   # 결정적 검출 필드 전체
    ├── <stem>__<variant>.render.png    # 렌즈 렌더 결과 (CPU)
    └── <stem>__<variant>.beauty.png    # CPU beauty 결과 (rot0/gamma05 변형만)
```

### 입력 선정 기준 (재발 방지 규칙)

1. **검출 성공**(detected=true)만으로는 불충분하다 — **검출 좌표가 실제 홍채
   위에 있는지 시각 확인**(좌표 마커 합성)을 통과해야 입력으로 채택한다.
2. **얼굴 위에 오버레이(face mesh 선, 랜드마크 마커 등)가 그려진 이미지 금지.**
   화면 모서리의 HUD 텍스트(FPS 등)는 눈 영역과 무관하므로 허용
   (640a/720a/face_closeup 모두 이 부류 — 좌표 정확성 시각 검증 완료).
3. 이력: 초기 입력 `face_highres.png`(demo `sample_01.png`, 960x718)는 face
   mesh 오버레이가 얼굴에 박힌 데모 스크린샷이었고, rot0/rot180 검출 좌표가
   눈이 아닌 **뺨(약 67px 아래)에 고정**되는 병리 검출이 conf=0.879로
   걸러지지 않은 채 기준선에 동결되어 있었다(mirror/gamma05 변형은 정상 —
   오버레이 오염에 의한 변형 간 비일관). 2단계 적대 리뷰에서 적발되어
   `face_closeup.png`(shared/demo `screenshot_20260109_172002` 계열의
   `..._172018.png`, 640x480→960x720 업스케일, 눈 영역 무오염)로 교체하고
   전체 재캡처했다. 교체 후 rot0 좌표가 좌/우 홍채 위에 정확함을 마커
   합성으로 확인했다 (conf=0.955).

### 변형 매트릭스 (입력별)

| variant         | 의미                                  | 산출물             | 적용 입력 |
|-----------------|---------------------------------------|--------------------|-----------|
| `rot0`          | 기본                                  | json+render+beauty | 전체      |
| `rot90`         | 회전 경로 동결                        | json+render¹       | 전체      |
| `rot180`        | 회전 경로 동결                        | json+render        | 전체      |
| `rot270`        | 회전 경로 동결                        | json+render¹       | 전체      |
| `mirror`        | 전면 카메라 좌우반전 **입력** 변형²   | json+render        | 전체      |
| `gamma05`       | 저조도 변형                           | json+render+beauty | 첫 입력만 |
| `mirror_rot270` | 미러×회전 조합 (전면 카메라 실경로)   | json 전용          | 첫 입력만 |
| `lensmirror`    | `IrisLensConfig.is_mirror=1` 렌더 분기| json+render³       | 첫 입력만 |

¹ **`screenshot_720a__rot90/rot270`은 render.png가 없다** — `iris_sdk_render_lens`가
  `IRIS_SDK_RENDER_FAILED`를 반환한다(검출 좌표공간과 회전 버퍼 공간 불일치로
  좌표가 540px 폭을 벗어남 — 아래 '알려진 현재 동작' ③). 이 **부재 자체가 기대
  동작**이며, `golden_capture_all.sh`의 `KNOWN_RENDER_FAILURES` 허용 목록과
  `golden_compare.py`의 파일셋 비교([추가]/[누락] 검출)가 양방향으로 앵커링한다 —
  이 변형에서 render가 생성되기 시작해도 동작 변화로 잡힌다.

² `mirror` 변형은 캡처 도구가 `cv::flip`으로 **픽셀만 반전한 입력**이다.
  SDK에는 mirror 정보가 전달되지 않는다(검출 API에 mirror 파라미터 없음).

³ `lensmirror`의 JSON은 `rot0`과 동일하다(같은 입력·검출, 렌더 설정만 다름 —
  의도된 중복 앵커). render는 '알려진 현재 동작' ④ 참조.

`rot90/rot270`은 이미지를 실제로 90도 회전시킨 뒤 `rotation_degrees`를
`iris_sdk_detect_with_rotation`에 함께 넘긴다.

beauty PNG는 용량 규율(inputs+baseline ≤ 15MB) 때문에 대표 변형(rot0, gamma05)에만
생성한다.

## 회전 경로 커버리지의 정확한 범위

감사 blocker였던 "회전 경로 무테스트"에 대해 본 골든이 커버하는 것은
**JSON(검출 출력)의 동결에 한정**된다:

- **유효**: 회전 4종의 검출 출력 전체(478점·홍채 5점·radius·conf)가 JSON으로
  동결되어, 회전 분기 코드의 어떤 조용한 변화도 JSON diff로 잡힌다.
  rot0≡rot180·rot90≡rot270 비트 일치(아래 ①)로 내부 역회전의 일관성도 동결된다.
- **유효하지 않음**: 회전 변형의 render PNG는 "렌즈가 홍채 위에 올바르게
  그려짐"을 증명하지 않는다 — 아래 ②/③처럼 현재 동작은 오배치/실패이며,
  골든은 그 오배치를 그대로 동결한 회귀 앵커일 뿐이다.

## 알려진 현재 동작 (결함 포함 — 고정 대상)

2단계 검증(전수 JSON 비교 + 좌표 마커 합성 시각 확인, 2026-06-11)으로 확정한
현재 SDK의 동작이다. **이 동작을 변경하는 수정은 골든 FAIL이 정상**이며,
의도적 변화로 분리 기록 + 베이스라인 재캡처 + 사용자 승인 대상이다.
감사 보고서(docs/lenssim-handoff/audit-report.md) blocker 2(회전/letterbox
경로 무테스트) 및 ADR-0001 §7.1(upright 좌표 계약)과 상호 참조.

1. **회전 JSON의 퇴화**: 3입력 모두에서 `rot0`과 `rot180`의 JSON이 메타데이터
   (rotation, input_width/height) 제외 **완전 동일**하고, `rot90`과 `rot270`도
   동일하다. 4개 회전의 실효 구분은 2개 값 집합뿐이다(90/270 합치기, 180
   미보정 계열). 3단계에서 90/270 구분이나 180 매핑을 "고치면" 골든 FAIL이
   나는데, 그것은 회귀가 아니라 의도적 변화다.
2. **회전 변형 render의 렌즈 오배치**: 회전 변형의 검출 좌표(JSON)는 회전된
   프레임에서 홍채 위치와 일치하지 않는다 — face_closeup 마커 합성 확인 결과
   rot0만 정렬 정확, rot90/rot180/rot270은 좌표가 눈 밖(배경/뺨)이다.
   `iris_sdk_render_lens`는 이 좌표에 렌즈를 그리므로 회전 변형 render의
   렌즈는 홍채 위에 있지 않다. 검출 좌표공간과 render에 전달된 회전 버퍼의
   공간이 불일치하는 것으로, **이것이 SDK 설계 제약인지 실제 버그인지 3단계
   착수 전 확인 항목**이다(코어 수정은 2단계 범위 밖이라 동결만 했다).
3. **`screenshot_720a__rot90/rot270` render 실패**: 위 ②의 좌표공간 불일치로
   좌표가 회전 프레임 폭(540px)을 벗어나 `IRIS_SDK_RENDER_FAILED`가 반환되고
   render.png가 생성되지 않는다(detected=true, JSON은 정상 생성). 파일 부재가
   기대 동작이다(변형 매트릭스 ¹).
4. **`IrisLensConfig.is_mirror`는 CPU 렌더 경로에서 no-op**:
   `face_closeup__lensmirror.render.png`는 `face_closeup__rot0.render.png`와
   **byte 동일**하다. 원인은 sdk_api.cpp `convert_to_cpp_lens_config`가
   `is_mirror`(및 C 구조체의 해당 필드)를 C++ LensConfig로 복사하지 않기
   때문(GPU v2 경로만 복사함 — sdk_api_v2.cpp). 이 no-op 자체를 골든이
   앵커링한다 — 변환 누락을 "고치면" lensmirror render가 달라져 골든 FAIL.
5. **회전 변형의 confidence 하락**: rot90/rot270에서 conf가 rot0 대비 크게
   낮다(예: face_closeup 0.955 → 0.662). 현재 동작으로 동결.

## 미커버 경로 (한계)

다음은 본 골든이 **커버하지 않는다**. 3단계 착수 전 보강 항목으로 기록한다
(감사 수정안의 4회전 상호 변환 비교 테스트와 연결):

- **GPU 경로 전체** (GLES/EGL 데스크톱 부재 — 아래 실기기 후속 계획).
- **비동기 경로 `submitFrameWithRotation`** (InferenceThread 경유 — 실제
  앱/JNI의 기본 경로). 골든은 동기 `iris_sdk_detect_with_rotation`만 호출한다.
  후속으로 캡처 도구에 submit 후 polling 옵션 추가를 검토.
- **`detectOnlyWithRotation`의 rotation==0 분기**: sdk_api.cpp는 rotation==0이면
  `detectOnly`로 우회하므로, rot0 골든은 회전 함수가 아니라 일반 detect 경로를
  앵커링한다(회전 함수의 0도 분기는 미통과).
- **letterbox 역변환의 단위 고정**: detect 경유로 간접 동결될 뿐 단위 테스트
  수준의 독립 고정이 없다.
- **`IrisLensConfig.rotation`(라디안) 렌더 분기**: 기본값 0.0만 사용.
  (`is_mirror`는 lensmirror 변형으로 1건 커버 — 단 현재 no-op, 위 ④.)
- **검출 API 수준의 mirror 파라미터**: mirror 변형은 입력 픽셀 반전일 뿐이다
  (변형 매트릭스 ²). mirror×회전 조합은 `mirror_rot270`(JSON)로 1건만 커버.

## 재캡처 방법

```bash
# 1) 도구 빌드 (기존 빌드 디렉토리 재사용 — 새 디렉토리 생성 금지)
cd cpp/cmake-build-debug
cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF
cmake --build . --target golden_capture --parallel

# 2) 매트릭스 일괄 캡처 (저장소 루트에서)
scripts/golden_capture_all.sh
#   인자: scripts/golden_capture_all.sh [OUT_DIR] [BUILD_DIR]
#   기본 OUT_DIR=cpp/tests/golden/baseline, BUILD_DIR=cpp/cmake-build-debug
#   환경변수 TEXTURE/MODELS 로 텍스처·모델 경로 재정의 가능
```

캡처 스크립트는 **침묵 실패를 차단**한다: 캡처마다 기대 산출물(json/render/
beauty)의 존재를 검증하고, 기대 파일 누락 또는 `KNOWN_RENDER_FAILURES` 허용
목록 위반(목록의 stem에서 render가 생성되는 역방향 변화 포함) 시 비0 종료한다.
render/beauty `[Warning]` 발생 캡처는 최종 요약에 목록으로 출력된다.
허용 목록을 바꿀 때는 본 README의 '알려진 현재 동작' 절을 함께 갱신한다.

입력을 바꾸려면 `inputs/`에 최대 변 960px PNG를 추가/교체한 뒤
[입력 선정 기준](#입력-선정-기준-재발-방지-규칙)(좌표 시각 검증 포함)을 통과시키고
재캡처한다. 렌즈 텍스처는 `shared/textures/2. KYOTO BROWN 1.png` 1개로 고정
(환경변수로 변경 가능).

## 비교 방법

```bash
# 후보 산출물을 임시 디렉토리에 캡처한 뒤 baseline과 비교
scripts/golden_capture_all.sh /tmp/golden_candidate
python3 scripts/golden_compare.py \
  --baseline cpp/tests/golden/baseline \
  --candidate /tmp/golden_candidate
# 종료코드 0 = 일치(PASS), 1 = 불일치(FAIL). 표준 라이브러리만 사용(PIL 불요).
```

- **JSON**: 필드를 의미 단위로 비교한다.
  - 정규화 좌표류(x/y/z, face_rect, face_rotation, confidence, visibility,
    eyelid_ratio, iris_quality, avg_iris_luma 등): `|Δ| ≤ eps_norm`
  - 반지름/픽셀류(`left_radius`, `right_radius`): `|Δ| ≤ eps_pix`
  - 정수/문자열/bool 메타(schema, model_version, rotation, mirror,
    input_width/height, *_detected, *_valid, *_count): 정확 일치
- **PNG**: 바이트 동일성. 불일치 시 크기·치수·다른바이트 수·첫 오프셋을 리포트.
- **파일셋**: baseline/candidate 한쪽에만 있는 파일은 [누락]/[추가]로 FAIL —
  720a 회전 render 부재(기대 동작) 같은 "파일이 없어야 하는" 앵커도 이것으로
  지켜진다.

## ε(허용오차) 근거와 적용 범위

- `eps_norm = 1e-4` (정규화 좌표): 정규화 좌표는 0~1 범위이며 1e-4는 960px 폭
  기준 약 0.1px에 해당한다. 덤프는 `%.6f` 고정 포맷이라 결정적 캡처 간 차이는 0이며,
  1e-4는 부동소수점 표현/플랫폼 컴파일러 차이를 흡수하는 안전 마진이다.
  실제 의미있는 좌표 이동(예: 픽셀 단위 어긋남)은 1e-4를 즉시 초과해 잡힌다.
- `eps_pix = 0.5` (반지름): 반지름은 픽셀 단위이며 0.5px(반 픽셀) 미만 차이는
  렌더 결과에 무영향이다. 0.5보다 큰 반지름 변화는 렌즈 크기 회귀로 간주한다.
  (홍채 중심·face_mesh 좌표는 정규화 필드이므로 `eps_norm` 적용 — 픽셀 ε은
  반지름 전용이다.)
- **적용 범위**: 이 ε은 **같은 detector 출력 전제의 동작 불변 검증 전용**이다.
  이종 추적기 A/B 비교(예: 자체 추적기 vs MediaPipe Tasks)의 판정 기준으로
  차용하지 않는다 — ADR-0001 §10 T1/§11 참조.
- **결정성**: 동일 코드/입력으로 2회 캡처 시 JSON·PNG diff는 0이어야 한다
  (2026-06-11 재캡처에서 2회 연속 diff 0 재검증 완료).
  비결정 필드(timestamp_ms, frame_width/height 등 캡처 시점·포인터 의존)는
  덤프에서 의도적으로 제외했다.

## GPU 골든 실기기 후속 계획 (3줄)

1. 데스크톱은 GLES/EGL 부재로 GPU 렌즈/뷰티/face-warp 경로를 캡처할 수 없어 CPU만 고정함.
2. GPU 골든은 Android 실기기에서 동일 입력·텍스처로 `render_lens_texture` /
   `apply_beauty_texture_v2` 출력을 readback해 별도 baseline으로 캡처(후속 작업).
3. GPU baseline은 드라이버/기기 편차가 있으므로 바이트 동일성 대신 SSIM/허용오차
   기반 이미지 비교로 전환하고, 본 CPU 골든과는 분리 관리한다.
