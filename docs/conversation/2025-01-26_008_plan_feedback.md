# 008. 뷰티 필터 프레임 직접 렌더링 계획에 대한 피드백

**작성일**: 2025-01-26
**작성자**: Claude (AI Assistant)
**대상 문서**: `docs/workPaper/008_direct_frame_rendering.md`

---

## 1. 종합 검토 의견

현재의 Stage 1.5(CPU 기반 처리) 환경에서 뷰티 필터 효과를 사용자에게 시각화하기 위한 **현실적이고 타당한 접근 방식**입니다. `PreviewView`를 숨기고 수정된 NV21 버퍼를 비트맵으로 변환하여 `OverlayView`의 배경으로 직접 그리는 전략은 타당하지만, 실시간성(30fps)을 유지하기 위해 성능 및 메모리 관리 측면에서 몇 가지 심각한 병목이 예상됩니다.

## 2. 세부 피드백

### ✅ 긍정적인 부분
1. **문제 정의의 정확성**: `PreviewView`가 카메라 스트림을 직접 점유하여 필터가 적용되지 않는 문제를 정확히 식별함.
2. **좌표계 일치성**: `OverlayView`에서 `fillCenter` 로직을 재구현하여 기존 렌즈 오버레이 좌표와의 일치성을 유지하려 함.
3. **UI 전환 전략**: 필터 활성화 시점에만 `PreviewView`를 감추고 커스텀 렌더링으로 전환하는 로직이 깔끔함.

### ⚠️ 위험 요소 및 개선 제안

#### ① NV21 → Bitmap 변환 성능 (가장 심각한 병목)
- **이슈**: 제안된 `YuvImage` → `JPEG` → `BitmapFactory` 방식은 CPU 부하가 매우 큽니다. 매 프레임 JPEG 압축/해제를 수행하면 FPS가 급격히 하락(10fps 이하 예상)할 수 있습니다.
- **제안**: 
    - 가능하면 네이티브 레이어(OpenCV)에서 NV21 → RGBA 변환을 수행하고 JNI를 통해 직접 Bitmap에 쓰는 방식이 가장 빠릅니다.
    - Java/Kotlin 레이어에서 처리해야 한다면 `RenderScript`의 `ScriptIntrinsicYuvToRGB`를 사용하거나, OpenCV의 `Imgproc.cvtColor`를 활용하십시오.

#### ② 메모리 할당 및 GC 부하
- **이슈**: `nv21ToBitmap` 내에서 매번 `Bitmap.createBitmap`을 호출하고, 회전을 위해 추가 비트맵을 생성하는 것은 심각한 메모리 파편화와 GC(Garbage Collection) 프리징을 유발합니다.
- **제안**:
    - **비트맵 재사용**: `Bitmap.Config.ARGB_8888` 비트맵을 하나만 미리 생성하여 매 프레임 `copyPixelsFromBuffer` 등으로 덮어쓰기 하십시오.
    - **회전 처리 최적화**: CPU에서 비트맵 픽셀을 회전시키지 말고, `OverlayView.onDraw`에서 `canvas.rotate()`를 사용하여 렌더링 시점에만 회전시켜 메모리 복사 비용을 절약하십시오.

#### ③ 스레드 동기화 및 생명주기
- **이슈**: `FrameAnalyzer`(Analysis 스레드)에서 비트맵을 생성하고 `onResult`(Main 스레드)로 넘기는 과정에서, `OverlayView`가 그리기도 전에 다음 프레임이 데이터를 덮어쓰거나 `recycle()`될 위험이 있습니다.
- **제안**: 
    - `AnalysisResult`에 담기는 비트맵의 소유권을 명확히 하고, Double Buffering 기법을 사용하여 읽기/쓰기 충돌을 방지하십시오.

#### ④ 해상도 다운스케일
- **이슈**: 1080p 전체 해상도를 CPU에서 매 프레임 비트맵으로 변환하고 그리는 것은 무리가 있습니다.
- **제안**: 필터용 비트맵은 화면 해상도에 맞춰 다운스케일(예: 1/2 또는 1/4)하여 처리 성능을 확보하십시오. 뷰티 필터(Soft Focus 등) 특성상 약간의 저해상도는 오히려 효과적으로 보일 수 있습니다.

## 3. 결론

제시된 계획은 논리적으로 완벽하지만, **Phase 5(성능 최적화)의 내용들이 '선택'이 아닌 '필수'**로 포함되어야 실시간 동작이 가능할 것으로 보입니다. 특히 비트맵 재사용과 JPEG 압축 우회는 첫 구현 단계부터 고려하는 것을 권장합니다.

## 4. 향후 조치 제안
1. `nv21ToBitmap` 함수를 `YuvImage` 대신 더 효율적인 방식으로 수정.
2. `OverlayView`에서 비트맵 회전 로직을 `Canvas.rotate()` 방식으로 변경.
3. 비트맵 캐싱 및 재사용 로직을 `FrameAnalyzer`에 즉시 적용.

---
*본 피드백은 IrisLensSDK의 실시간성 목표(30fps)를 달성하기 위한 기술적 검토 결과입니다.*

---

## ✂️ 피드백에 대한 답변

**작성일**: 2025-01-26
**작성자**: Claude (AI Assistant)

---

피드백 내용에 전적으로 동의합니다. 지적해 주신 성능 병목 요소들은 실시간 렌더링에서 치명적인 문제가 될 수 있습니다. 아래와 같이 계획을 수정하겠습니다.

### 1. NV21 → Bitmap 변환 방식 개선 (필수 적용)

**기존 (문제점)**:
```kotlin
YuvImage → JPEG 압축 → BitmapFactory 디코딩
// 예상 처리 시간: 50-100ms (10fps 이하)
```

**개선안 A: JNI + OpenCV (권장)**
```cpp
// C++ 레이어에서 NV21 → RGBA 변환 후 바로 Java Bitmap에 쓰기
extern "C" JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeNv21ToRgba(
    JNIEnv* env, jclass, jbyteArray nv21, jint width, jint height, jobject bitmap) {

    // NV21 → BGR → RGBA 변환 (OpenCV)
    cv::Mat yuv(height + height/2, width, CV_8UC1, nv21Data);
    cv::Mat rgba(height, width, CV_8UC4);
    cv::cvtColor(yuv, rgba, cv::COLOR_YUV2RGBA_NV21);

    // Java Bitmap에 직접 쓰기
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    memcpy(pixels, rgba.data, width * height * 4);
    AndroidBitmap_unlockPixels(env, bitmap);
}
```
- 예상 처리 시간: **5-10ms** (3-5배 이상 개선)
- SDK에 이미 OpenCV가 포함되어 있으므로 추가 의존성 없음

**개선안 B: RenderScript (대안)**
```kotlin
// RenderScript YuvToRGB Intrinsic 사용
val rs = RenderScript.create(context)
val yuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
```
- Deprecated되었지만 Android 12까지는 동작
- 예상 처리 시간: **8-15ms**

→ **결정**: 개선안 A (JNI + OpenCV)를 기본으로 채택

### 2. 메모리 관리 전략 (필수 적용)

**비트맵 재사용 (Double Buffering)**:
```kotlin
class FrameAnalyzer(...) {
    // 두 개의 비트맵을 번갈아 사용 (읽기/쓰기 충돌 방지)
    private val bitmapPool = arrayOf<Bitmap?>(null, null)
    private var currentBitmapIndex = 0

    private fun getNextBitmap(width: Int, height: Int): Bitmap {
        val nextIndex = (currentBitmapIndex + 1) % 2
        var bitmap = bitmapPool[nextIndex]

        if (bitmap == null || bitmap.width != width || bitmap.height != height) {
            bitmap?.recycle()
            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmapPool[nextIndex] = bitmap
        }

        currentBitmapIndex = nextIndex
        return bitmap
    }
}
```

**회전 처리 최적화**:
```kotlin
// OverlayView.drawFilteredFrame()
private fun drawFilteredFrame(canvas: Canvas, frame: Bitmap, rotationDegrees: Int) {
    canvas.save()

    // 중심 기준 회전 (비트맵 복사 대신 캔버스 변환)
    canvas.translate(width / 2f, height / 2f)
    canvas.rotate(rotationDegrees.toFloat())
    canvas.translate(-frame.width / 2f, -frame.height / 2f)

    // 스케일 및 미러링
    if (isMirror) {
        canvas.scale(-1f, 1f, frame.width / 2f, frame.height / 2f)
    }

    canvas.drawBitmap(frame, 0f, 0f, null)
    canvas.restore()
}
```

### 3. 해상도 다운스케일 (필수 적용)

뷰티 필터의 특성상 약간의 저해상도는 오히려 소프트한 효과를 강화할 수 있습니다.

```kotlin
companion object {
    // 뷰티 필터용 다운스케일 (1/2 해상도)
    private const val BEAUTY_DOWNSCALE_FACTOR = 2
}

// C++ 레이어에서 다운스케일된 상태로 변환
// 1920x1080 → 960x540 = 메모리 75% 절감, 처리 시간 ~75% 절감
```

### 4. 수정된 구현 순서

| Phase | 작업 내용 | 비고 |
|-------|-----------|------|
| 1 | JNI `nativeNv21ToRgba()` 함수 구현 | **필수** (성능 핵심) |
| 2 | Double Buffering 비트맵 풀 구현 | **필수** (메모리 안정성) |
| 3 | `OverlayView`에 Canvas 회전 기반 렌더링 | **필수** (메모리 절감) |
| 4 | 다운스케일 옵션 적용 | **필수** (성능 확보) |
| 5 | MainActivity UI 전환 로직 | 기존 계획 유지 |
| 6 | 테스트 및 튜닝 | FPS, 메모리 프로파일링 |

### 5. 예상 성능 목표

| 항목 | 기존 계획 | 수정 후 목표 |
|------|-----------|--------------|
| NV21 → Bitmap 변환 | 50-100ms | **5-10ms** |
| 메모리 할당 | 매 프레임 | **고정 버퍼** |
| 회전 처리 | CPU 픽셀 복사 | **Canvas 변환** |
| 해상도 | 1080p | **540p** (뷰티용) |
| 예상 FPS | 10-15fps | **25-30fps** |

### 6. 결론

피드백에서 지적한 대로, 성능 최적화 요소들을 **Phase 1부터 필수로** 포함하여 구현하겠습니다. 특히:

1. ✅ JNI + OpenCV 기반 YUV 변환 (JPEG 압축 우회)
2. ✅ Double Buffering 비트맵 재사용
3. ✅ Canvas.rotate() 기반 회전 처리
4. ✅ 1/2 해상도 다운스케일 기본 적용

계획서(`008_direct_frame_rendering.md`)를 이 내용으로 업데이트하겠습니다.
