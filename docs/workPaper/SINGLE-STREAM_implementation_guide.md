# 단일 스트림(Preview-only) 카메라 — 최종 구현 지시서

브랜치: `feature/single-stream-camera` / 기준 커밋 `c38f76f`
근거는 전부 실제 코드 Read로 확인했다. 확인 못 한 것은 **실측 필요**로 명시했다.

---

## 좌표계 확정

### 0. 실제 코드에서 유도한 계약 (근거)

| # | 사실 | 근거 |
|---|---|---|
| A | 쿼드 attribute: `pos(-1,-1)↔tex(0,0)`, `pos(1,1)↔tex(1,1)` | `CameraGLRenderer.kt:70-76` |
| B | `uFlipY==1`은 **정점 y만** 반전(`pos.y=-pos.y`). texCoord 불변 → "목적지 행 순서 반전" | `CameraGLRenderer.kt:94-96` |
| C | `uMirror==1`은 **ST 적용 후** `texCoord.x = 1-x` | `CameraGLRenderer.kt:112-117` |
| D | `uRotate`는 정점 전용 (최종 blit만 사용) | `:102-108`, `:632`(링 0), `:919-922` |
| E | 링 패스 uniform = `uMirror=isMirror, uFlipY=1, uRotate=0, uScale=(1,1)` | `:626-632` |
| F | native 렌즈 쿼드는 texcoord 패스스루(정점=텍스처 좌표 항등) | `shader_sources.cpp:224-233`, `gpu_lens_renderer.cpp:551-560` |
| G | native는 랜드마크를 `y←1-y`, mirror면 `x←1-x` 후 링 텍스처 좌표로 사용 | `gpu_lens_renderer.cpp:1111-1118` |
| H | `uFrameAspect = IrisResult.frame_width / frame_height` | `gpu_lens_renderer.cpp:1163` |
| I | 반경은 `frame_height`로 정규화 → 분석 해상도 독립 | `gpu_lens_renderer.cpp:1068-1073` |
| J | MediaPipe 주입 좌표는 **비미러** | `GpuRenderActivity.kt:1367-1370`, `TasksToIrisResult.kt:94` |

### 1. 유도

표기: 목적지 정규화 좌표 `(u,v)` (`u=(x_ndc+1)/2`, `v=(y_ndc+1)/2`), `ST(·)`=`uSTMatrix` 적용, `Mir(a,b)=(1-a,b)`, `m = isMirror?1:0`, `g(x)= m? 1-x : x`.

- A·B·C·E로부터 **링**: `C_ring(s,t) = Mir^m( ST(s, 1-t) )`
- F·G로부터 **native**: 랜드마크 `(x,y)` → 링 텍스처 좌표 `(g(x), 1-y)`
- `glReadPixels`는 window y=0(=NDC y=-1=`v=0`)부터 읽고, MediaPipe는 첫 행을 `y=0`으로 본다 → **분석 정규화 좌표 = 목적지 `(u,v)` 그대로**

따라서 정합 조건은 ST의 형태와 무관하게:

```
C_an(u, v)  ==  C_ring( g(u), 1-v )  ==  Mir^m( ST( g(u), v ) )
```

즉 **분석 이미지 = 링 이미지를 상하 반전(+미러면 좌우 반전)한 것**이다. 소스 쪽(ST/uMirror)을 건드리는 게 아니라 **목적지 쪽에서 native 매핑의 역을 적용**하는 것이 정답이다.

이를 만족하는 uniform 조합:
- `uFlipY=0` → `t = v` ✔
- 목적지 x 반전(`pos.x = -pos.x`)을 `m=1`일 때만 → `s = 1-u = g(u)` ✔
- `uMirror = m` (**링과 동일**), `uSTMatrix = stMatrix` (**링과 동일**)

→ `C_an(u,v) = Mir^m(ST(1-u, v))` = 요구값. **ST에 회전 성분이 있어도 성립한다.**

### 2. 확정표

| 항목 | 링 패스 (`:614-644`, 무변경) | **분석 패스 (신규)** | 근거 |
|---|---|---|---|
| 소스 텍스처 | OES `oesTextureId` | **OES (동일)** | 링을 readback하면 이미 미러·flipY가 걸려 계약이 상대적이 됨 |
| `uSTMatrix` | `stMatrix` | **`stMatrix` (동일)** | 회전·크롭은 ST 단독 담당 |
| `uMirror` | `if(isMirror) 1 else 0` | **`if(isMirror) 1 else 0` (동일)** | §1 유도. **0이 아니다** |
| **`uFlipPosX` (신규)** | **0 (명시)** | **`if(isMirror) 1 else 0`** | 목적지 x 반전 = native `x←1-x`의 역 |
| `uFlipY` | `1` | **`0`** | 목적지 y 반전 = native `y←1-y`의 역 + readPixels top-down |
| `uRotate` | `0` | **`0`** | 회전은 ST 단독 |
| `uScale` | `(1,1)` | **`(1,1)`** | 전체 프레임. displayZoom/crop 금지 |
| viewport | `frameWidth×frameHeight` | **`analysisW×analysisH`** (upright 종횡비) | §3 |
| readback | — | `glPixelStorei(GL_PACK_ALIGNMENT,4)` + `glReadPixels(0,0,aW,aH,GL_RGBA,GL_UNSIGNED_BYTE)` → `rowStride == aW*4` 보장 | FaceTracker 제로카피 분기 진입 |
| `rotationDegrees` (FaceTracker) | (ImageProxy 값) | **`0`** | §4 |

**퇴화 확인**: ST에 회전이 없고 `ST(s,t)=(s,1-t)`인 경우 → 신규식 `Mir(ST(1-u,v)) = Mir((1-u,1-v)) = (u,1-v)`. 원설계식 `ST(u,v)=(u,1-v)`와 동일. 즉 신규 조합은 회전 없는 경우 원설계와 결과가 같고, 회전이 있을 때만 달라진다(원설계는 이 경우 좌우+상하가 동시에 뒤집힌 버퍼를 만든다).

**후면 카메라(`m=0`)**에서는 `uFlipPosX=0`, `uMirror=0`이라 원설계와 비트 동일.

### 3. 분석 FBO 치수 = upright 종횡비 (필수)

링 FBO는 upright 내용을 센서 치수 텍스처에 **비등방 스쿼시**해 담는다(그래서 `renderToScreen:891-897`이 `isRotated`로 W/H를 스왑해 되돌린다). 분석 FBO를 센서 종횡비로 잡으면 MP가 찌그러진 얼굴을 보고 `uFrameAspect`(H)도 거짓이 되어 렌즈가 타원이 된다.

```kotlin
val swap = (frameRotation == 90 || frameRotation == 270)
val upW = if (swap) frameHeight else frameWidth
val upH = if (swap) frameWidth  else frameHeight
var k = 1; while (maxOf(upW, upH) / k > ANALYSIS_MAX_LONG_SIDE) k++   // = 960
val aW = upW / k; val aH = upH / k
val taps = ((k + 1) / 2).coerceIn(1, 4)   // 탭 1개 = HW bilinear 2×2 → 유효 2N×2N 박스
```

| 캡처(센서) | rot | upright | k | 분석 FBO | taps | 버퍼 1장 |
|---|---|---|---|---|---|---|
| 1920x1080 | 270 | 1080x1920 | 2 | 540x960 | 1 | 2.07MB |
| 3840x2160 | 270 | 2160x3840 | 4 | 540x960 | 2 | 2.07MB |
| 4000x3000 | 270 | 3000x4000 | 5 | 600x800 | 3 | 1.92MB |

`ANALYSIS_MAX_LONG_SIDE = 960`은 현행 `selectAnalysisResolution`의 `960x540`(`GpuRenderActivity.kt:1216-1226`)과 픽셀 수가 같다 → Step 3에서 **추론 입력 크기가 변인이 아니게** 된다.

정수 나눗셈 잔차로 `aW/aH`가 `upW/upH`와 최대 0.5% 어긋날 수 있다(`uFrameAspect` 오차). 육안 무영향으로 판단하나 로그로 실제 종횡비 델타를 남긴다.

박스 필터 오프셋 `uSrcTexel = (1/frameWidth, 1/frameHeight)`는 **OES 버퍼 텍셀** 기준이고, `vTexCoord`가 post-ST 공간이라도 k가 두 축 공통 스칼라이므로 회전 유무와 무관하게 N×N 박스가 성립한다.

> 단순 `GL_LINEAR` 1탭 축소는 3840→540(7배)에서 심한 언더샘플링이 된다. 현행 ImageAnalysis는 ISP 스케일러가 이 일을 대신 해준다 → **박스 필터는 선택이 아니라 등가 유지 조건**.

### 4. `rotationDegrees = 0`

ST가 이미 회전을 적용해 분석 버퍼가 upright이므로 0을 넘긴다. 그러면
- `processingOptions(0)` → MP 추가 회전 없음
- `CoordMapper.sensorToUpright` 항등, `upW/upH` 스왑 없음 (`TasksToIrisResult.kt:85-90`, `FaceTracker.kt:517-521`)
- `out.frameWidth/Height = (aW, aH)` = upright 종횡비 ✔

### 5. 이 표는 "실측으로 확정"해야 한다

§1 유도는 코드에서 나온 것이지만, `stMatrix`의 실제 내용(회전 성분 유무)은 **드라이버·기기 의존**이다(`docs/lenssim-handoff/audit-report.md:296/307`이 "TransformationInfo 미사용 → 기기 의존 동작에 위임"을 major로 지적). 그래서:
- **Step 0에서 `stMatrix` 16개 float를 1회 로깅**한다. 교차항(column-major `[1]`, `[4]`)이 0이 아니면 회전 존재 = 위 정정이 필수.
- **Step 2 PNG 덤프 게이트**(얼굴 정립 / 미러 아님 / 상하 정상 / 종횡비 정상)를 반드시 통과시킨다.
- 런타임 A/B 토글을 **2개** 둔다: `setAnalysisFlipY(0/1)` **와** `setAnalysisFlipPosX(0/1)`. 두 축이 회전에 의해 엉키므로 한쪽만으로는 분리 진단이 안 된다.

---

## 단계별 구현

각 단계 = **독립 커밋 + 독립 실기기 판정**. 통과 못 하면 다음으로 안 간다.

---

### Step 0 — 측정만 (코드 ~10줄, 게이트)

**파일**: `GpuRenderActivity.kt:1317-1333`(진단 블록), `CameraGLRenderer.kt:511` 직후

```kotlin
// GpuRenderActivity.kt — 진단 블록 안
sizes?.take(6)?.forEach { s ->
    Log.i(TAG, "  ${s.width}x${s.height} minDur=${map.getOutputMinFrameDuration(SurfaceTexture::class.java, s)/1e6}ms" +
               " stall=${map.getOutputStallDuration(SurfaceTexture::class.java, s)/1e6}ms")
}

// CameraGLRenderer.onDrawFrame — getTransformMatrix 직후, 1회만
if (!stMatrixLogged) { stMatrixLogged = true
    Log.i(TAG, "stMatrix=" + stMatrix.joinToString { "%.3f".format(it) })
}
```

현행 `EXPERIMENT_PREVIEW_ONLY=true` 빌드로 3840x2160 / 4000x3000 바인딩·fps 실측.

**판정**
- **통과(스트림 조합)**: 바인딩 성공 + `minFrameDuration ≤ 33ms`
- fps는 이 단계에서 **판정하지 않는다.** `EXPERIMENT_PREVIEW_ONLY`는 추론이 없어 `slotDetected=false` → 렌즈 패스 미실행(`CameraGLRenderer.kt:542`), 뷰티 기본 OFF, readback 없음 = "링 write + 최종 blit"만 잰다. 실제 Step 4 구성과 다르므로 여기서 fps ≥28을 받아도 30fps 보증이 안 된다.
- **실패**: 4K 포기 → Step 4 목표를 2560x1440으로 하향. 그것도 실패면 트랙 재검토.

**부수 산출물**: `stMatrix` 교차항 → §5 확정표의 최종 근거.

---

### Step 1 — SDK 진입점 추가 (동작 무변경)

**파일**: `android/iris-sdk/src/main/java/com/irislenssdk/tracking/FaceTracker.kt`

`analyze(ImageProxy)`(`:163-256`)를 어댑터로 축소하고 본체를 `analyzeBuffer`로 추출한 뒤, GL readback용 진입점을 **추가만** 한다(바이너리 호환).

```kotlin
/** 분석 executor 스레드 전용. ImageProxy close는 이 함수 책임(기존 계약 유지). */
fun analyze(imageProxy: ImageProxy) {
    if (closed) { imageProxy.close(); return }
    try {
        val p = imageProxy.planes[0]
        analyzeBuffer(p.buffer, imageProxy.width, imageProxy.height, p.rowStride,
            imageProxy.imageInfo.rotationDegrees, imageProxy.imageInfo.timestamp)
    } finally { imageProxy.close() }
}

/**
 * ImageProxy 없는 RGBA_8888 진입점 (단일 스트림 GL readback 경로).
 * @param rgba direct ByteBuffer. **capacity가 정확히 width*4*height** 여야 한다.
 *             호출 반환 전까지 호출자가 유효성을 보장한다(반환 후 재사용 허용).
 *             절대 인덱스 접근만 — position/limit 변경 금지.
 * @param rotationDegrees 이미 upright인 버퍼면 0.
 * 백프레셔는 호출자 책임 (ImageProxy 경로의 KEEP_ONLY_LATEST 없음).
 */
fun analyzeRgba(rgba: ByteBuffer, width: Int, height: Int,
                rotationDegrees: Int, frameTimestampNs: Long) {
    if (closed) return
    if (!rgba.isDirect) { onError("analyzeRgba: direct ByteBuffer 필요"); return }
    if (rgba.capacity() != width * 4 * height) {
        onError("analyzeRgba: capacity ${rgba.capacity()} != ${width*4*height}"); return
    }
    analyzeBuffer(rgba, width, height, width * 4, rotationDegrees, frameTimestampNs)
}

private fun analyzeBuffer(src: ByteBuffer, width: Int, height: Int, rowStride: Int,
                          rotation: Int, frameTimestampNs: Long) {
    /* 기존 :168~252 본체 — imageProxy 참조만 파라미터로 치환 */
}
```

**의도적으로 하지 않는 것**: `obtainCompactBuffer`(`:486-490`) / `obtainStreamBuffer`(`:332-`)의 "capacity ≥ size 재사용"은 건드리지 않는다. MediaPipe `ByteBufferImageBuilder`가 정확 용량을 요구하는지는 **실측 필요**(코드에서 확인 못 함)이며, 기존 경로를 만지면 Step 1의 "무변경" 보증이 깨진다. 신규 경로는 풀이 정확 용량으로 할당하므로 이 위험에 노출되지 않는다.

**KDoc 정정**: 클래스 주석 `:30-33`("KEEP_ONLY_LATEST 백프레셔 전제")과 `onRawResult` 주석 `:65-67`("ImageProxy close 전")을 "ImageProxy 경로 한정 / 버퍼 경로는 호출자 책임"으로.

**검증**: 기존 동작 시각·수치 무변화(렌즈 정합, infer ms, 검출률). JVM 단위테스트는 신규 없음(순수 리팩토링). 회귀 시 커밋 revert.

---

### Step 2 — readback 패스 가동 (소비 없음, 좌표·색공간 검증)

**파일**: `CameraGLRenderer.kt`, `CameraGLView.kt`, `GpuRenderActivity.kt`
**플래그**: `ANALYSIS_READBACK_DEBUG = true` (ImageAnalysis는 그대로 유지 → 추론은 기존 경로)

#### 2-a. VERTEX_SHADER에 uniform 1개 추가 (`CameraGLRenderer.kt:79-121`)

```glsl
uniform int uFlipPosX;   // 목적지 x 반전 — 분석 패스 전용(링/화면은 0)
...
    if (uFlipY == 1)    { pos.y = -pos.y; }
    if (uFlipPosX == 1) { pos.x = -pos.x; }   // uScale/uRotate 적용 전
    pos *= uScale;
```

`#version`은 **첫 바이트 유지**(Mali 셰이더 전멸 회귀 c5d9f37). 링(`:626-632`)·화면(`:885-887`) 패스에서 `uFlipPosX=0`을 **명시 설정**해 기존 동작 비트 동일 보장(VERTEX_SHADER 공유).

#### 2-b. 박스 다운샘플 프래그먼트 셰이더 (companion, `:204` 뒤)

```glsl
private const val ANALYSIS_BOX_FRAGMENT_SHADER = """#version 310 es
    #extension GL_OES_EGL_image_external_essl3 : require
    precision highp float;
    uniform samplerExternalOES uOESTexture;
    uniform vec2 uSrcTexel;   // (1/frameWidth, 1/frameHeight)
    uniform int  uTaps;       // N (1..4)
    in vec2 vTexCoord;
    out vec4 fragColor;
    void main() {
        if (uTaps <= 1) { fragColor = texture(uOESTexture, vTexCoord); return; }
        float n = float(uTaps);
        vec2 base = vTexCoord - (n - 1.0) * uSrcTexel;
        vec4 sum = vec4(0.0);
        for (int j = 0; j < 4; ++j) { if (j >= uTaps) break;
          for (int i = 0; i < 4; ++i) { if (i >= uTaps) break;
            sum += texture(uOESTexture, base + vec2(float(i), float(j)) * 2.0 * uSrcTexel);
        } }
        fragColor = sum / (n * n);
    }
"""
```

#### 2-c. 필드 / 생명주기

```kotlin
// === 단일 스트림 추론 readback ===
interface AnalysisSink {
    /** GL 스레드 호출. 즉시 반환하고 별도 스레드에서 처리 후 반드시 release() 호출. */
    fun onAnalysisFrame(buf: ByteBuffer, width: Int, height: Int,
                        frameTimestampNs: Long, release: () -> Unit)
}
@Volatile private var analysisSink: AnalysisSink? = null
private var analysisProgram = 0; private var analysisFbo = 0; private var analysisTex = 0
private var analysisW = 0; private var analysisH = 0; private var analysisTaps = 1
private var uaSrcTexelLoc = -1; private var uaTapsLoc = -1; private var uaSTLoc = -1
private var uaMirrorLoc = -1; private var uaFlipYLoc = -1; private var uaFlipPosXLoc = -1
private var uaScaleLoc = -1; private var uaRotateLoc = -1; private var uaTexLoc = -1
private var lastReadbackTsNs = 0L
private var analysisFlipY = 0        // A/B (정본 0)
private var analysisFlipPosX = -1    // -1 = isMirror 따름, 0/1 = 강제 (A/B)
private var analysisGeneration = 0
private var analysisDumpOnce = false
var onAnalysisDump: ((ByteBuffer, Int, Int) -> Unit)? = null
private var readbackMsEma = 0f       // glReadPixels 자체 비용 계측 (Step 5 근거)
```

- `onSurfaceCreated`(`:439` 뒤): `analysisProgram = createProgram(VERTEX_SHADER, ANALYSIS_BOX_FRAGMENT_SHADER)` + uniform location 캐시
- `markGlHandlesStale`(`:402-421`): `analysisProgram/analysisFbo/analysisTex = 0`, `analysisGeneration++`, 풀·핸드오프 슬롯 비우기
- `release`(`:1364-1404`): analysis FBO/텍스처/프로그램 delete 추가
- `recreateIntermediateBuffers`(`:1187-1233`) 말미: `recreateAnalysisBuffers()` 호출
- **`setFrameRotation`(`:1115-1119`)에서도** rotation이 바뀌면 `recreateAnalysisBuffers()` 호출 — `frameRotation`이 분석 FBO 치수를 결정하기 때문

```kotlin
private fun recreateAnalysisBuffers() {
    // delete-before-recreate (핸들 0이면 no-op)
    if (frameWidth <= 0 || frameHeight <= 0) return
    val swap = (frameRotation == 90 || frameRotation == 270)
    val upW = if (swap) frameHeight else frameWidth
    val upH = if (swap) frameWidth  else frameHeight
    var k = 1; while (maxOf(upW, upH) / k > ANALYSIS_MAX_LONG_SIDE) k++
    analysisW = upW / k; analysisH = upH / k
    analysisTaps = ((k + 1) / 2).coerceIn(1, 4)
    // glGenTextures / glTexImage2D(RGBA, analysisW, analysisH) / FBO attach / 완결성 체크
    synchronized(handoffLock) {
        analysisGeneration++; analysisPool.clear(); analysisPoolAllocated = 0
        pending?.let { }; pending = null; inFlight = false
    }
    Log.i(TAG, "analysis FBO ${analysisW}x${analysisH} k=$k taps=$analysisTaps " +
               "upright=${upW}x${upH} aspectΔ=${"%.4f".format(analysisW.toFloat()/analysisH - upW.toFloat()/upH)}")
}
```

#### 2-d. `onDrawFrame` 삽입 — `:528` 직후 (링 write 완료, **렌즈 합성 전**)

```kotlin
ringWrite = (ringWrite + 1) % ringSize
// 1.5단계: 추론용 저해상 readback. 렌즈/뷰티 합성 이전 = 카메라 원본만.
maybeCaptureAnalysisFrame(frameTsNs)
```

합성 후를 읽으면 `avg_iris_luma`(기본 ON — `gpu_lens_renderer.h:488 use_measured_luma_=true`, 소비 `gpu_lens_renderer.cpp:891`)가 자기 출력을 먹는 피드백 루프가 된다. 측정 주체는 `TasksToIrisResult.roiLumaLinear`(`:320-`)이며 분석 버퍼 픽셀을 직접 읽는다.

```kotlin
private fun maybeCaptureAnalysisFrame(frameTsNs: Long) {
    val sink = analysisSink ?: return
    if (analysisFbo == 0 || analysisW <= 0) return
    if (frameTsNs == 0L || frameTsNs == lastReadbackTsNs) return   // 동일 프레임 중복 추론 방지
    val (gen, buf) = obtainAnalysisBuffer() ?: return
    lastReadbackTsNs = frameTsNs

    val t0 = System.nanoTime()
    renderAnalysisDownsample()                       // analysisFbo 바인딩 유지된 채 반환
    GLES31.glPixelStorei(GLES31.GL_PACK_ALIGNMENT, 4)
    buf.clear()
    GLES31.glReadPixels(0, 0, analysisW, analysisH,
        GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, buf)
    GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)
    GLES31.glViewport(0, 0, viewWidth, viewHeight)   // state 복원 (검은 화면 회귀 방어)
    buf.rewind()
    readbackMsEma += ((System.nanoTime() - t0) / 1e6f - readbackMsEma) * 0.1f

    if (analysisDumpOnce) { analysisDumpOnce = false
        onAnalysisDump?.invoke(buf.duplicate(), analysisW, analysisH) }   // 소비자가 별도 스레드로

    handoffLatestWins(sink, gen, buf, frameTsNs)
}

private fun renderAnalysisDownsample() {
    GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, analysisFbo)
    GLES31.glViewport(0, 0, analysisW, analysisH)
    GLES31.glUseProgram(analysisProgram)
    GLES31.glUniformMatrix4fv(uaSTLoc, 1, false, stMatrix, 0)
    GLES31.glUniform1i(uaMirrorLoc,   if (isMirror) 1 else 0)   // ★ 링과 동일
    GLES31.glUniform1i(uaFlipPosXLoc, if (analysisFlipPosX >= 0) analysisFlipPosX
                                      else if (isMirror) 1 else 0)   // ★ 신규
    GLES31.glUniform1i(uaFlipYLoc, analysisFlipY)                // ★ 0 = readPixels top-down
    GLES31.glUniform1i(uaRotateLoc, 0)
    GLES31.glUniform2f(uaScaleLoc, 1f, 1f)
    GLES31.glUniform2f(uaSrcTexelLoc, 1f / frameWidth, 1f / frameHeight)
    GLES31.glUniform1i(uaTapsLoc, analysisTaps)
    GLES31.glActiveTexture(GLES31.GL_TEXTURE0)
    GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId)
    GLES31.glUniform1i(uaTexLoc, 0)
    renderFullscreenQuad()
}
```

#### 2-e. 핸드오프 = **latest-wins 단일 슬롯** (FIFO 큐 금지)

```kotlin
private class AnalysisFrame(val gen: Int, val buf: ByteBuffer, val tsNs: Long)
private val handoffLock = Any()
private var pending: AnalysisFrame? = null
private var inFlight = false
private val analysisPool = ArrayDeque<Pair<Int, ByteBuffer>>()
private var analysisPoolAllocated = 0     // ANALYSIS_POOL_MAX = 3

private fun handoffLatestWins(sink: AnalysisSink, gen: Int, buf: ByteBuffer, tsNs: Long) {
    var submit = false
    synchronized(handoffLock) {
        pending?.let { recycleLocked(it.gen, it.buf) }   // ★ 묵은 프레임 즉시 폐기
        pending = AnalysisFrame(gen, buf, tsNs)
        if (!inFlight) { inFlight = true; submit = true }
    }
    if (submit) sink.onAnalysisFrame(buf, analysisW, analysisH, tsNs) { drainOnAnalysisThread(sink) }
}
```

싱크 쪽(Activity)은 executor에 태스크 1개를 넣고, 그 태스크가 **슬롯이 빌 때까지 루프**로 최신 프레임만 처리한다(구현은 2-g). 이것이 `STRATEGY_KEEP_ONLY_LATEST`(`GpuRenderActivity.kt:1294`)의 정확한 등가물이다. 단순 `executor.execute{}` 반복 제출은 `Executors.newSingleThreadExecutor()`(`GpuRenderActivity.kt:204`)의 무제한 FIFO 큐 때문에 **묵은 프레임이 먼저 추론**되어 랜드마크 age가 상시 +1추론주기 늘어난다.

```kotlin
private fun obtainAnalysisBuffer(): Pair<Int, ByteBuffer>? = synchronized(handoffLock) {
    analysisPool.removeFirstOrNull()?.let { return@synchronized it }
    if (analysisPoolAllocated >= ANALYSIS_POOL_MAX) return@synchronized null   // 드롭
    analysisPoolAllocated++
    analysisGeneration to ByteBuffer.allocateDirect(analysisW * analysisH * 4)
        .order(ByteOrder.nativeOrder())
}
private fun recycleLocked(gen: Int, buf: ByteBuffer) {
    if (gen != analysisGeneration) return       // ★ stale — 폐기만. 카운터 감소 금지
    analysisPool.addLast(gen to buf)            //   (recreate가 이미 0으로 리셋했으므로 -- 하면 언더플로)
}
```

`ANALYSIS_POOL_MAX = 3`: in-flight 1 + pending 1 + 작성 중 1. 3×2.07MB = 6.2MB.

#### 2-f. `CameraGLView.kt` 위임 (`:206-219` 패턴)

```kotlin
fun setAnalysisSink(s: CameraGLRenderer.AnalysisSink?) = queueEvent { glRenderer.setAnalysisSink(s) }
fun setAnalysisFlipY(v: Int)    = queueEvent { glRenderer.setAnalysisFlipY(v) }
fun setAnalysisFlipPosX(v: Int) = queueEvent { glRenderer.setAnalysisFlipPosX(v) }
fun requestAnalysisDump()       = queueEvent { glRenderer.requestAnalysisDump() }
```

#### 2-g. `GpuRenderActivity.kt` — Step 2에서는 **덤프만**

```kotlin
private const val ANALYSIS_READBACK_DEBUG = true    // Step 2 한정
private const val ANALYSIS_CONSUME = false          // Step 2에서는 analyzeRgba 미호출

private val analysisSink = object : CameraGLRenderer.AnalysisSink {
    override fun onAnalysisFrame(buf: ByteBuffer, w: Int, h: Int, tsNs: Long, release: () -> Unit) {
        val ok = runCatching {
            analysisExecutor.execute {
                try {
                    if (ANALYSIS_CONSUME) ensureFaceTracker().analyzeRgba(buf, w, h, 0, tsNs)
                } catch (t: Throwable) { Log.w(TAG, "analyzeRgba 실패", t)
                } finally { release() }   // ← drain 루프 진입점
            }
        }.isSuccess
        if (!ok) release()                // RejectedExecutionException → 버퍼 회수
    }
}
```

PNG 덤프는 **GL 스레드가 아닌 별도 스레드**에서 저장한다(수백 ms 스톨 방지): `onAnalysisDump`에서 `Bitmap.copyPixelsFromBuffer` 후 `Thread{ ... filesDir/analysis_dump.png }`.

#### 2-h. 색공간 계측 (Step 2에서만 가능 — 두 경로 병존)

같은 시점에 (a) ImageProxy 버퍼 기반 `roiLumaLinear`와 (b) readback 버퍼 기반 값을 각각 로깅해 비율/오프셋을 실측한다. 현재 픽셀은 CameraX `OUTPUT_IMAGE_FORMAT_RGBA_8888`(libyuv YUV→RGBA), 신규는 `samplerExternalOES` 드라이버 YUV→RGB다. range(limited 16–235 vs full 0–255)·매트릭스가 다르면 luma가 계통적으로 이동하고, 그대로 렌즈 밝기 적응(`use_measured_luma_`)에 들어간다.

**검증 게이트 (필수, 전부 통과해야 Step 3)**
1. PNG 덤프 1장 육안: ① 얼굴 정립 ② **미러 아님** ③ 상하 정상 ④ 종횡비 정상. 어긋나면 `setAnalysisFlipY` / `setAnalysisFlipPosX` A/B로 정본 확정 후 §확정표를 갱신.
2. `checkGlError("onDrawFrame")` 0건, 5분 지속 후 검은 화면 없음 (EXTERNAL_OES + FBO + glReadPixels 조합의 검은 화면 회귀·revert 전례: `cpp/include/iris_sdk/gpu/gpu_lens_renderer.h:438-441`).
3. `avg_iris_luma` 두 경로 차 **±5% 이내**. 초과 시 Step 3 전에 보정 결정.
4. `readbackMsEma` 기록 (Step 5 비용/효과 근거).
5. GL fps 하락폭 기록.

**롤백**: `ANALYSIS_READBACK_DEBUG=false` 한 줄.

---

### Step 3 — 스트림 전환 (해상도는 1920x1080 유지)

**변인은 구조 하나뿐.** 해상도는 절대 올리지 않는다.

**플래그**(`GpuRenderActivity.kt:101-111` 교체)
```kotlin
private const val USE_SINGLE_STREAM = true
private val SINGLE_STREAM_PREVIEW_SIZE = Size(1920, 1080)   // Step 4에서 승격
private const val ANALYSIS_CONSUME = true
// EXPERIMENT_PREVIEW_ONLY / EXPERIMENT_PREVIEW_SIZE 제거
```

**`bindCameraUseCases`(`:1228-1338`)**
```kotlin
val targetResolution = if (USE_SINGLE_STREAM) SINGLE_STREAM_PREVIEW_SIZE else selectOptimalResolution()
...
val camera = if (USE_SINGLE_STREAM)
    cameraProvider.bindToLifecycle(this, cameraSelector, preview)
else
    cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)

cameraGLView.setMirror(lensFacing == CameraSelector.LENS_FACING_FRONT)

// ★ frameRotation 재소싱 — setFrameRotation의 유일 호출자가 processFrameTasks:1355 였다.
//   빠뜨리면 frameRotation=0 고정 → renderToScreen(:891-897) Cover 종횡비 붕괴 + 분석 FBO
//   치수 오산 = 조용한 회귀.
val sensorRot = camera.cameraInfo.sensorRotationDegrees
cameraGLView.setFrameRotation(sensorRot)   // setFrameSize보다 먼저 들어오도록 배치
Log.i(TAG, "frameRotation(sensor)=$sensorRot")

if (USE_SINGLE_STREAM || ANALYSIS_READBACK_DEBUG) cameraGLView.setAnalysisSink(analysisSink)
```

`targetRotation = ROTATION_0` 핀(`:1271-1278`)이 유지되므로 `sensorRotationDegrees`가 기존 `imageInfo.rotationDegrees`와 같은 값이어야 한다 — **Step 3에서 로그로 대조 확인**(Step 2 시점 ImageAnalysis 로그와 비교).

**`selectorFor`(`:1251-1268`) 주석 정정**: 1080p 캡을 푸는 실제 스위치는 `PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE`가 아니라 앱이 설정한 `ResolutionStrategy`다(`UseCase.mergeConfigs`가 `OPTION_MAX_RESOLUTION`을 제거). `setResolutionStrategy`를 빼면 캡이 조용히 부활한다.

**`onTasksRawResult`(`:1397-`)는 무수정.** `srcWidth/srcHeight`가 분석 FBO 치수로 바뀔 뿐이고 반경은 `frame_height` 정규화라 해상도 독립(`gpu_lens_renderer.cpp:1068-1073`).

**`processFrameTasks`(`:1351-1359`)**: `USE_SINGLE_STREAM`이면 미사용. `setFrameRotation` 호출은 위로 이관.

**`onDestroy`(`:1958-1981`) 순서 교정**
```kotlin
// GL 스레드가 sink를 더 이상 못 잡도록 동기 배리어로 확실히 끊고 나서 shutdown.
val latch = java.util.concurrent.CountDownLatch(1)
cameraGLView.queueEvent { /* setAnalysisSink(null) */ ; latch.countDown() }
runCatching { latch.await(200, TimeUnit.MILLISECONDS) }
cameraGLView.release()
runCatching { analysisExecutor.execute { faceTracker?.close(); ... } }
analysisExecutor.shutdown()
```

**검증**
- 렌즈 정합 육안 + 디버그 메시(`btnToggleMesh`) + raw iris 오버레이. **얼굴을 좌우로 이동**시켜 좌우 눈 라벨 정상 확인(미러 오류의 가장 확실한 신호).
- **뷰티 ON 상태 필수 포함** (GL↔추론 결합이 가장 심한 조건).
- `frame-sync: minΔ` 로그(현행 대비 감소해야 정상), `Landmark age at render`, 검출률.
- 신규 계측: `analyzeRgba` 호출률, `analyzeRgba 시작 시점 프레임 age`(KEEP_ONLY_LATEST 대비 회귀 직접 확인), `readbackMsEma`.
- **2번째 기기(S23+ 등) 필수.** `stMatrix` 회전 의존은 기기 의존이라 Tab S10 Ultra 1대로 좌표 계약을 확정하면 안 된다.

**롤백**: `USE_SINGLE_STREAM=false` 한 줄 → ImageAnalysis 재바인딩, 현행과 완전 동일.

---

### Step 3.5 — 렌더 캐이던스를 카메라 캐이던스로 (해상도와 분리, 독립 판정)

현재 `renderMode = RENDERMODE_CONTINUOUSLY`(`CameraGLView.kt:123`) + `setOnFrameAvailableListener{ /* 빈 몸통 */ }`(`CameraGLRenderer.kt:459`)이라 **디스플레이 120Hz로 렌더**한다. `onDrawFrame`의 링 write(`:523-528`)에도 새 프레임 게이트가 없다. 4K에서 이 구조는 링 write만 약 4GB/s이고, 렌즈·뷰티·blit도 전부 4배 픽셀로 120Hz 반복된다. **해상도 상향의 최대 리스크는 readback이 아니라 이쪽이다.**

```kotlin
// CameraGLView.init
renderMode = RENDERMODE_WHEN_DIRTY
glRenderer.onFrameAvailable = { requestRender() }   // CameraGLRenderer.kt:459 빈 몸통 대체
```

**필수 부수 작업**: `WHEN_DIRTY`에서는 `queueEvent`가 재draw를 유발하지 않는다. `CameraGLView`의 모든 설정 setter(`setBeautyConfig`/`setBeautyEnabled`/`setMirror`/`setIrisResult`/`setLensConfig`/… 약 20개)를 `queueEvent{...}` → `queueEvent{...}; requestRender()`로 바꾼다. 헬퍼 하나로 통일:
```kotlin
private inline fun queueAndRender(crossinline b: () -> Unit) { queueEvent { b() }; requestRender() }
```
누락하면 "렌즈를 바꿔도 화면이 안 바뀐다" 류 회귀가 난다(카메라 프레임이 오면 결국 갱신되지만 최대 33ms 지연).

**추가 가드**: `requestRender()`가 카메라 프레임 없이 불릴 수 있으므로 `onDrawFrame`에서 `frameTsNs`가 직전과 같으면 **링 슬롯을 진행시키지 않는다**(중복 ts 슬롯 방지).

**효과**
- 4K 파이프라인 부하 120Hz → 30Hz (1/4)
- 링 4슬롯 커버 시간 33ms → 약 133ms → `FrameRingSelector`(`MAX_SKEW_NS=150ms`) 매칭 정확도 개선, Step 4에서 `ringSize`를 2로 줄여도 66ms 확보

**검증**: 현행 1920x1080에서 **무회귀**여야 한다. fps(30 근처로 내려가는 것이 정상 — 이는 개선이지 퇴화가 아님), 렌즈 반응 지연 육안, `frame-sync minΔ`, 모든 UI 토글이 즉시 반영되는지 전수 확인.
**롤백**: `RENDER_ON_DEMAND=false` 플래그로 `RENDERMODE_CONTINUOUSLY` 복귀.

---

### Step 4 — 해상도 상향 + 메모리/필터 대책

1. `SINGLE_STREAM_PREVIEW_SIZE = Size(3840, 2160)` (Step 0 결과 반영). `Preview.Builder.setMaxResolution(...)` 명시 추가.
2. **`ringSize` 적응화 필수** — 3840x2160 RGBA 1장 = 33.2MB, 4장 = 132.7MB로 프로젝트 메모리 목표 100MB를 단독 초과. `private val ringSize = 4`(`CameraGLRenderer.kt:236`)를 `var`로 바꾸고 `recreateIntermediateBuffers`에서 `ringSize = if (width.toLong()*height > 2_500_000) 2 else 4`. `ringTex/ringFbo/ringTsNs`도 최대 크기로 할당 후 유효 구간만 사용.
   Step 3.5를 먼저 마쳤다면 2슬롯 = 약 66ms 커버라 frame-sync가 유지된다. Step 3.5를 건너뛰면 2슬롯 = 약 17ms → **frame-sync 기능이 사실상 무효화**된다(동작은 하지만 항상 최신 슬롯으로 수렴).
3. **`upscaleMode` 재조정** — §재검토 참조. 기본을 0(bilinear)으로 내리고 육안 A/B.
4. dead code 정리: `lensFboId`/`lensOutputTextureId`는 `createLensFbo` 유일 호출처가 `recreateIntermediateBuffers:1227`의 `if (lensFboId != 0)`이라 부트스트랩 경로가 없다(항상 no-op). 함께 제거.

**검증**
- 선명도 육안 A/B (1920 ↔ 3840 실시간 토글)
- `dumpsys meminfo` Graphics
- **렌즈 ON + 뷰티 ON 5분 지속** fps·발열 로그
- 검출률, `frame-sync minΔ`(0에서 이탈하면 즉시 실패)

**롤백**: `SINGLE_STREAM_PREVIEW_SIZE`를 1920x1080으로.

---

### Step 5 — PBO 비동기 readback (Step 4 fps 미달 시에만)

동기 `glReadPixels`는 파이프라인 flush를 유발한다. Step 2의 `readbackMsEma` 실측값이 예산을 먹고 있을 때만 착수한다.

```
pbo[2]: glBufferData(GL_PIXEL_PACK_BUFFER, size, null, GL_STREAM_READ)
매 프레임:
  (소비) readIdx=(writeIdx+1)%2
         fence[readIdx] 있고 glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0) != TIMEOUT_EXPIRED 이면
           bind → glMapBufferRange(GL_MAP_READ_BIT) → 풀 버퍼로 memcpy(2MB, ~0.3ms)
           → glUnmapBuffer → glDeleteSync → handoffLatestWins(..., pboTsNs[readIdx])
         아직이면 **블로킹 없이 스킵**
  (생산) 다운샘플 draw → bind pbo[writeIdx] → glReadPixels(...,0) → unbind
         → fence[writeIdx]=glFenceSync(...) → pboTsNs[writeIdx]=frameTsNs → writeIdx^=1
```

- ⚠️ **매핑된 ByteBuffer를 분석 스레드로 넘기지 말 것.** unmap은 GL 스레드 책임 → 반드시 풀 버퍼로 memcpy 후 핸드오프.
- ⚠️ **`pboTsNs[readIdx]`(캡처 시점 ts)를 넘길 것.** 현재 프레임 ts를 넘기면 frame-sync에 1프레임 스큐가 조용히 생긴다.
- ⚠️ 컨텍스트 손실 시 fence 무효 → `markGlHandlesStale`에서 fence 배열·`writeIdx`·인플라이트 상태 전부 리셋(안 하면 복귀 후 영구 스톨).
- PBO는 2개지만 **핸드오프 슬롯은 여전히 1개(latest-wins)** 유지.

**롤백**: `ANALYSIS_READBACK_ASYNC=false` → 동기 경로.

---

## 롤백

| 플래그 | 위치 | 기본 | 끄면 |
|---|---|---|---|
| `USE_SINGLE_STREAM` | `GpuRenderActivity` companion | Step3에서 `true` | ImageAnalysis 재바인딩 = **현행과 완전 동일** |
| `ANALYSIS_READBACK_DEBUG` | 동상 | Step2 `true` → 이후 `false` | readback 패스 미실행 |
| `ANALYSIS_CONSUME` | 동상 | Step2 `false` / Step3 `true` | 덤프만, 추론 미소비 |
| `RENDER_ON_DEMAND` | `CameraGLView` | Step3.5에서 `true` | `RENDERMODE_CONTINUOUSLY` 복귀 |
| `ANALYSIS_READBACK_ASYNC` | `GpuRenderActivity` | `false` | 동기 glReadPixels |
| `SINGLE_STREAM_PREVIEW_SIZE` | 동상 | `1920x1080` | 해상도만 되돌림 |
| `setAnalysisFlipY(0/1)` | 런타임 디버그 버튼 | `0` | 상하 부호 A/B |
| `setAnalysisFlipPosX(-1/0/1)` | 런타임 디버그 버튼 | `-1`(=isMirror) | 좌우 부호 A/B |
| `ANALYSIS_MAX_LONG_SIDE` | `CameraGLRenderer` | `960` | 분석 해상도 스윕 |
| `ANALYSIS_POOL_MAX` | 동상 | `3` | 인플라이트 상한 |

**브랜치**: `feature/single-stream-camera`에서 Step0~5 각 1커밋. Step 3 이후는 **revert 없이 플래그 한 줄로 현장(실기기) 롤백**이 가능해야 한다 — 판정 중 앱 재빌드 없이 A/B하려면 디버그 버튼에 걸어두는 편이 낫다.
`develop` 머지는 Step 4 판정 통과 후 일괄. 중간 단계는 머지하지 않는다.

---

## 성능 판정 기준

| 지표 | 측정 | 목표 | **실패 = 즉시 롤백** |
|---|---|---|---|
| 렌더 fps (렌즈 ON + 뷰티 ON) | `updateGpuFps`(`:590-600`), HUD | ≥ 30, 5분 지속 | < 28 또는 현행 대비 −15% 초과 |
| 추론 cadence | `analyzeRgba` 호출률 카운터 | ≥ 24 fps | < 20 fps |
| 추론 지연 | `onInferenceStats` infer ms | 현행 ±20% | +30% 초과 |
| **랜드마크 age (신규)** | `analyzeRgba` 시작 시점 `now - frameTsNs` | 현행 KEEP_ONLY_LATEST 대비 +1추론주기 이내 | +33ms 초과 |
| 검출률 | `detected` 비율 60초 집계 | 현행 −2%p 이내 | −5%p 초과 |
| frame-sync | `frame-sync: minΔ` | ≤ 33ms (Δ→0 기대) | > 66ms 또는 클럭 도메인 강등 발생 |
| 렌더 시점 age | `Landmark age at render`(`:568-575`) | 현행 이하 | +33ms 초과 |
| **avg_iris_luma (신규)** | 좌/우 60초 평균 | 현행 ±5% | ±10% 초과 |
| **렌즈 색조 (신규)** | `lum:meas` ↔ `lum:fb` 토글 육안 | 체감 변화 없음 | 색조 이동 체감 |
| `readbackMs` | GL 스레드 EMA | ≤ 2ms | > 5ms → Step 5 착수 |
| 그래픽 메모리 | `dumpsys meminfo` Graphics | ≤ 120MB | > 160MB |
| GL 오류 | `checkGlError`(`:1354-1359`) | 0건 | 1건이라도 |
| 안정성 | 5분 지속 + 백그라운드 복귀 3회 | 무결 | 검은 화면 / 프리즈 / 렌즈 소실 |
| **선명도(최종 목적)** | 1920 ↔ 3840 실시간 토글 육안 | 머리카락·눈썹 격자 감소 체감 | 체감 차 없음 → Step 4 되돌리고 원인 재탐색 |

**우선순위 규칙(사전 확정)**: 렌더 fps와 추론 cadence가 **동시에** 미달이면 **추론 cadence를 우선**하고 해상도를 낮춘다. 렌즈 정합이 선명도보다 상위 목표다.

---

## 검증에서 걸러진 것

| # | 원설계 주장 | 판정 | 대체안 |
|---|---|---|---|
| 1 | 분석 패스 `uMirror=0`, `uFlipY=0`이면 정합 | **반증**. ST에 90/270 회전 성분이 있으면 두 플립이 상쇄되어 분석 버퍼가 링 이미지 그 자체(미러+상하반전)가 된다. 데모는 전면 고정(`GpuRenderActivity.kt:203`, `:1315`)이라 항상 이 분기. 기존 uniform 4종 조합만으로는 어떤 값으로도 올바른 버퍼를 못 만든다 | **`uMirror`·`uSTMatrix`는 링과 동일**, `uFlipY=0`, **신규 `uFlipPosX`로 목적지 x 반전**. ST 형태와 무관하게 성립. 회전 없는 ST에서는 원설계와 동일 결과로 퇴화 |
| 2 | §1.1을 "확정표", §1.2를 "독립 재유도해 확인"으로 제시 | **과장**. `stMatrix`의 실제 내용은 기기 의존이며 코드로 확인 불가 | Step 0에서 `stMatrix` 16 float 로깅 → 교차항 확인. Step 2 PNG 덤프 게이트 + **A/B 토글 2개**(상하·좌우)로 실기기 확정 |
| 3 | `ANALYSIS_POOL_MAX=2` 인플라이트 상한 = `KEEP_ONLY_LATEST` 동등물 | **반증**. `newSingleThreadExecutor`(`:204`)는 무제한 FIFO → 대기 중인 **묵은** 프레임이 먼저 추론된다. 코드베이스가 명시적으로 제거했던 지연(`FaceTracker.kt:30-33`, `:233-234`)이 되살아난다 | **latest-wins 단일 슬롯 + drain 루프**(§Step 2-e). 새 프레임이 오면 대기 중인 옛 버퍼를 즉시 풀로 반납. 판정 지표에 "analyzeRgba 시작 시점 프레임 age" 추가 |
| 4 | `recycleAnalysisBuffer` stale 분기에서 `analysisPoolAllocated--` | **반증**. `recreateAnalysisBuffers`가 이미 0으로 리셋하므로 언더플로 → FBO 리사이즈마다 실효 상한이 2→3→4로 늘어 백프레셔가 조용히 풀린다 | stale 분기에서 카운터를 건드리지 않는다 |
| 5 | Step 0 게이트(`EXPERIMENT_PREVIEW_ONLY`에서 fps≥28)로 30fps 판정 | **반증**. 추론 없음 → `slotDetected=false`로 렌즈 패스 미실행(`:542`), 뷰티 기본 OFF, readback 없음. "링 write + blit"만 잼 | Step 0은 **바인딩 성공 + minFrameDuration**까지만 인정. fps는 Step 2/3에서 다시 받고, Step 4 승격 조건에 "렌즈 ON + 뷰티 ON 5분 지속"을 명시 |
| 6 | 4K 리스크의 주범은 동기 readback | **반증**. `RENDERMODE_CONTINUOUSLY`(`CameraGLView.kt:123`) + `onFrameAvailable` 빈 몸통(`CameraGLRenderer.kt:459`) + 링 write에 프레임 게이트 없음(`:523-528`) → 파이프라인 전체가 **디스플레이 120Hz**로 돈다. 4K에서 링 write만 약 4GB/s | **Step 3.5 신설**: `RENDERMODE_WHEN_DIRTY` + `onFrameAvailable{requestRender()}`. 부하 1/4. 해상도와 분리해 독립 커밋·독립 판정 |
| 7 | Step 4에서 `ringSize` 4→2로 줄여도 "동작한다" | **불충분**. 120Hz 기준 4슬롯=33ms, 2슬롯=17ms → 추론 왕복이 17ms를 넘으면 `fsync:on`(기본 ON, `:238`) 기능이 **조용히 무효화** | Step 3.5를 **선행 조건**으로. WHEN_DIRTY면 4슬롯=133ms, 2슬롯=66ms로 여유 확보. `frame-sync minΔ`가 0에서 이탈하면 실패 판정 |
| 8 | `avg_iris_luma`의 함정 근거가 `FaceTracker.kt:191 sampleIrisLuma` | **근거 오류**. `sampleIrisLuma`는 `onSnapshot`으로만 흐르고 데모는 `onSnapshot = { }`(`:1373`)이라 dead. 결론(합성 전 read)은 맞음 | 실제 소비 경로는 `TasksToIrisResult.convertWithLuma → roiLumaLinear`(`:217-232`, `:320-`) → `use_measured_luma_=true`(`gpu_lens_renderer.h:488`) → `gpu_lens_renderer.cpp:891`. **색공간 변경**(libyuv → OES 드라이버)이 원설계에 누락 → Step 2 계측 + 판정표 2행 추가 |
| 9 | MediaPipe가 `capacity == w*4*h` 정확 일치를 요구 → 기존 `obtainCompactBuffer`가 매 프레임 예외 | **미확인**(코드로 확인 불가, **실측 필요**). 사실이면 `obtainStreamBuffer`(`:332-`)도 같은 문제 | 기존 경로는 **건드리지 않는다**(Step 1 무변경 보증). 신규 경로는 풀이 정확 용량으로 할당해 구조적으로 회피 + `analyzeRgba`에서 명시 검증 |
| 10 | `activeRingSize` 필드 | **존재하지 않음**. `private val ringSize = 4` 상수(`:236`) | `val`→`var` + `recreateIntermediateBuffers`에서 적응 |
| 11 | FMLens는 GL readback으로 고해상 소스를 쓴다 | **반증**. Camera1 `setPreviewCallbackWithBuffer`(NV21)로 검출하고 기본 프리뷰 목표가 `Point(1920,1080)`(`docs/ar-report/fmlens-decompiled/full/sources/com/intervision/fmlens/camera/CameraProxy.java:208`, `:67-74`) | "FM이 더 큰 소스라 선명하다"는 인과는 성립하지 않는다. §재검토 참조 |
| 12 | `highRes` 플래그가 1080p 캡을 푼다 | **부정확**. 실제 스위치는 앱이 설정한 `ResolutionStrategy`(`UseCase.mergeConfigs`가 `OPTION_MAX_RESOLUTION` 제거) | 주석 정정 + Step 4에서 `setMaxResolution` 명시 |

**추가로 확인해 통과한 것들**(반증 아님, 설계대로 진행):
- 데드락 없음 — GL 스레드가 분석 스레드를 기다리는 경로 없음. 역방향(`setIrisResult` → `queueEvent`)도 비블로킹.
- FaceTracker "생성 스레드에서만 detect" 규약 — 싱크가 같은 `analysisExecutor`로 마샬링하므로 생성·detect·close가 동일 단일 스레드(`:204`, `:1362`, `:1969`).
- 버퍼 수명 — `detectForVideo`가 동기이고 `release()`가 `finally`에 있어 MediaPipe가 읽는 동안 재사용 없음. generation 태그로 리사이즈 시 stale 혼입 차단.
- 박스 필터 탭 오프셋 — `vTexCoord`가 post-ST 공간이어도 `uSrcTexel`이 OES 버퍼 텍셀이고 k가 축 공통 스칼라라 회전 유무와 무관하게 성립.
- `glReadPixels` 원점(row 0 = window y=0 = dest v=0)과 `GL_PACK_ALIGNMENT 4` + RGBA8 → `rowStride = W*4` 보장.
- LEGACY 경로 부재(W4-D 제거), `MediaPipeBenchmarkActivity.kt:195`는 자체 ImageAnalysis 별도 바인딩이라 무영향, 벤치 `MODE_STREAM` 팔은 `obtainStreamBuffer`로 자체 복사 후 넘기므로 GL 풀 버퍼 수명과 무관.
- `selectAnalysisResolution`(`:1216`)은 dead가 될 뿐 — Step 4에서 제거.

---

## 재검토 필요 (4K로 바뀌면 전제가 뒤집히는 기존 설정)

| 설정 | 현재 | 4K에서 무엇이 바뀌나 | 조치 |
|---|---|---|---|
| `upscaleMode = 1` (Catmull-Rom bicubic) `CameraGLRenderer.kt:296`, 셰이더 `:147-204` | 1920x1080 → 창(약 2960 폭) **확대** 구간 전제 | 3840 소스면 최종 blit이 **축소**로 뒤집힌다. bicubic은 확대 재구성 필터라 축소에서 안티에일리어싱을 못 하고, mode 2(언샤프)는 에일리어싱을 증폭한다 | 기본을 **0(bilinear)** 으로 내리고 `btnZoom` 순환으로 육안 A/B. 근본 해법은 링 텍스처 mipmap(`LINEAR_MIPMAP_LINEAR` + `glGenerateMipmap`)이나 4K 매 프레임 생성 비용 **실측 필요** |
| `sharpenAmount = 0.35f` `:297` | 확대로 잃은 고주파 복원 | 축소 구간에서는 링잉만 남는다 | mode 0에서는 미사용. mode 2 자체를 4K에서 봉인 |
| `USE_16_9_CAPTURE = true` `GpuRenderActivity.kt:101` | 4:3 상한이 1440x1080이라 16:9가 유리했던 전제 | Preview 단독이면 4:3 **4000x3000**이 열린다. 4:3이 세로 화각 33% 넓고 픽셀도 더 많다 | Step 6 후보. 단 4000x3000 RGBA = 48MB/슬롯 → `ringSize` 2여도 96MB. 메모리 vs FOV 트레이드오프 **실측 필요** |
| `displayZoom = 1.0f` `:292` | 확대하면 흐려지는 전제 (`docs/workPaper/FOV_gap_diagnosis.md`) | 4K에서 blit이 축소 구간(약 0.74~0.77배)이므로 **1/S(약 1.3배)까지 올려도 화질 손실 0** → "선명도 vs FOV" 트레이드오프 자체가 소멸 | 이게 재설계의 **두 번째 실익**. Step 4 성과 지표에 FOV 격차 해소를 함께 올릴 것 |
| `selectOptimalResolution()` `:1186-1206`의 상한 주석 | "1920x1080이 CameraX 실질 상한" | 단일 스트림이 되면 사실이 아니게 됨 | Step 3에서 주석 갱신, Step 4에서 함수 자체 정리 |
| `selectAnalysisResolution()` `:1216-1226` | ImageAnalysis 전용 | dead | Step 4에서 제거 |
| `ringSize = 4` 상수 `:236` | 1080p 기준 8.3MB×4 = 33MB | 3840x2160이면 33.2MB×4 = **132.7MB** (프로젝트 목표 100MB 단독 초과) | Step 4에서 적응화(필수). Step 3.5 선행 조건 |
| `lensFboId` / `lensOutputTextureId` `:256-257`, `createLensFbo():724-771` | — | 유일 호출처가 `if (lensFboId != 0)`(`:1227`)이라 부트스트랩 경로 없음 = 항상 no-op | Step 4에서 제거 (오해 방지) |
| 뷰티 skin smoothing | 기본 OFF (`:210-213`) | 4K에서 블러 반경이 픽셀 기준이면 상대 강도가 1/2로 약해진다 (**실측 필요**) | Step 4 판정에 뷰티 ON 상태 포함. 강도 재튜닝 필요 시 별건 |
| **"FMLens 대비 선명도 격차"의 원인 가설** | "FM이 더 큰 소스를 쓴다" | **반증됨** — FM도 1920x1080 기본이고 렌더 소스마저 NV21→RGBA 변환 텍스처(우리 OES 직결보다 낮은 화질 경로) | Step 3 완료 시점에 **업스케일 필터·뷰티 블러 OFF·AE 차이**만으로 격차가 얼마나 좁혀지는지 A/B로 분리. 그래야 Step 4의 비용 대비 효과 판단이 정확해진다 |
| `frameSyncEnabled = true` `:238` | 링 4슬롯 = 33ms 히스토리 전제 | Step 3.5 후 133ms, Step 4에서 슬롯 2로 줄이면 66ms | Step 4 판정에 `minΔ` 실패 조건 명시 |

---

### 관련 파일 (절대경로)

- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt`
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLView.kt`
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/src/main/java/com/irislenssdk/tracking/FaceTracker.kt`
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/src/main/java/com/irislenssdk/tracking/TasksToIrisResult.kt`
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_lens_renderer.cpp` (좌표 계약 정본 `:1111-1118`, `:1163`)
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/shader_sources.cpp` (`:224-233` 렌즈 정점 셰이더)
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/include/iris_sdk/gpu/gpu_lens_renderer.h` (`:438-441` readback 회귀 전례, `:488` `use_measured_luma_`)
- `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/docs/ar-report/fmlens-decompiled/full/sources/com/intervision/fmlens/camera/CameraProxy.java` (`:67-74`, `:208`)