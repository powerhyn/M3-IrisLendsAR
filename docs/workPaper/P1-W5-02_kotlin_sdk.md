# P1-W5-02: Kotlin SDK 클래스

**태스크 ID**: P1-W5-02
**상태**: ✅ 완료
**시작일**: 2026-01-12
**완료일**: 2026-01-12

---

## 1. 계획

### 목표
JNI 네이티브 메서드를 래핑하는 Kotlin API 클래스 구현. 안드로이드 앱에서 사용하기 쉬운 고수준 API 제공

### 산출물
| 파일 | 설명 |
|------|------|
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisLensSDKKt.kt` | 메인 SDK 싱글톤 클래스 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisResultKt.kt` | 검출 결과 데이터 클래스 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/LensConfigKt.kt` | 렌즈 설정 데이터 클래스 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/ProcessResultKt.kt` | 처리 결과 데이터 클래스 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisException.kt` | sealed class 예외 계층 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/FrameFormat.kt` | 프레임 포맷 열거형 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/BlendMode.kt` | 블렌드 모드 열거형 |
| `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/Extensions.kt` | 유틸리티 확장 함수 |

### 검증 기준
- [x] 네이티브 라이브러리 로드 성공 (`System.loadLibrary`)
- [x] IrisLensSDK.init() 정상 동작
- [x] IrisLensSDK.detect() 결과 반환
- [x] IrisResult 데이터 클래스 필드 검증
- [x] 예외 처리 동작 확인

### 선행 조건
- P1-W5-01 JNI 래퍼 구현 완료 ✅

---

## 2. 분석

### 2.1 설계 결정

| 결정 | 이유 |
|------|------|
| Java 클래스 유지 + Kotlin 래퍼 | JNI 호환성 유지, Kotlin 관용적 API 제공 |
| `kotlin/` 서브패키지 | Java API와 Kotlin API 명확히 분리 |
| Result 패턴 사용 | Kotlin 관용적 에러 처리, 명시적 실패 처리 유도 |
| sealed class 예외 계층 | 타입 안전한 에러 분기 (when 표현식) |
| data class 불변성 | 스레드 안전성 보장 |
| 싱글톤 패턴 | C API가 싱글 인스턴스, Kotlin에서도 동일하게 |
| NV21 기본 포맷 | Android 카메라 기본, 변환 오버헤드 최소화 |

### 2.2 패키지 구조

```
android/iris-sdk/src/main/java/com/irislenssdk/
├── IrisLensSDK.java      # Java 메인 SDK (JNI 네이티브 선언)
├── IrisResult.java       # Java 검출 결과 (JNI 필드 접근용)
├── LensConfig.java       # Java 렌즈 설정 (JNI 필드 접근용)
└── kotlin/
    ├── IrisLensSDKKt.kt      # Kotlin 메인 SDK (Java 래핑)
    ├── IrisResultKt.kt       # Kotlin 검출 결과 (불변 data class)
    ├── LensConfigKt.kt       # Kotlin 렌즈 설정 (불변 data class)
    ├── ProcessResultKt.kt    # Kotlin 처리 결과
    ├── IrisException.kt      # sealed class 예외 계층
    ├── FrameFormat.kt        # 프레임 포맷 enum
    ├── BlendMode.kt          # 블렌드 모드 enum
    └── Extensions.kt         # 유틸리티 확장 함수
```

---

## 3. 실행 내역

### 3.1 주요 구현

#### IrisLensSDKKt.kt
```kotlin
class IrisLensSDKKt private constructor() {
    companion object {
        @Volatile private var instance: IrisLensSDKKt? = null

        @JvmStatic
        fun getInstance(): IrisLensSDKKt {
            return instance ?: synchronized(this) {
                instance ?: IrisLensSDKKt().also { instance = it }
            }
        }
    }

    fun init(context: Context): Result<Unit>
    fun detect(frameData: ByteArray, width: Int, height: Int,
               format: FrameFormat = FrameFormat.NV21): Result<IrisResultKt>
    fun process(frameData: ByteArray, width: Int, height: Int,
                format: FrameFormat = FrameFormat.NV21,
                config: LensConfigKt = LensConfigKt.Default): Result<ProcessResultKt>
    fun destroy()
}
```

#### IrisException.kt (sealed class)
```kotlin
sealed class IrisException(message: String, val errorCode: Int) : Exception(message) {
    class NotInitialized(msg: String) : IrisException(msg, NOT_INITIALIZED)
    class NoFaceDetected(msg: String) : IrisException(msg, NO_FACE)
    class InvalidParameter(msg: String) : IrisException(msg, INVALID_PARAM)
    // ...
}
```

#### Extensions.kt
```kotlin
// Camera2/CameraX 이미지 변환
fun Image.toNv21ByteArray(reuseBuffer: ByteArray? = null): ByteArray

// Bitmap 변환 및 검출
fun Bitmap.toRgbaByteArray(): ByteArray
fun Bitmap.detectIris(sdk: IrisLensSDKKt): Result<IrisResultKt>

// 추적 안정성 분석
fun IrisResultKt.distanceTo(other: IrisResultKt): Pair<Float, Float>
fun IrisResultKt.isStableTo(previous: IrisResultKt, threshold: Float): Boolean

// 애니메이션 보간
fun LensConfigKt.lerp(target: LensConfigKt, fraction: Float): LensConfigKt
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| System.loadLibrary 성공 | ✅ | Java SDK에서 로드 |
| IrisLensSDKKt.init() 동작 | ✅ | Result 패턴 적용 |
| IrisLensSDKKt.detect() 결과 | ✅ | IrisResultKt 변환 |
| IrisResult 필드 검증 | ✅ | data class 불변성 |
| 예외 처리 동작 | ✅ | sealed class 계층 |

### 코드 리뷰 결과

| 카테고리 | 점수 | 비고 |
|----------|------|------|
| 보안 | 9/10 | 버퍼 크기 검증, 입력 검증 |
| 성능 | 8/10 | 버퍼 재사용 옵션 추가 |
| 아키텍처 | 9/10 | Java/Kotlin 분리 설계 |
| 유지보수성 | 9/10 | 명확한 패키지 구조 |
| **종합** | **8.75/10** | **프로덕션 품질** |

---

## 5. 이슈 및 학습

### 이슈
| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| 1 | toNv21ByteArray 매 호출 할당 | ✅ 해결 | 버퍼 재사용 파라미터 추가 |
| 2 | toRgbaByteArray config 미검증 | ✅ 해결 | require(config == ARGB_8888) 추가 |
| 3 | Result.map 중복 정의 | ✅ 해결 | stdlib 사용으로 삭제 |

### 결정 사항
| 결정 | 이유 |
|------|------|
| Java 클래스 유지 | JNI 필드 접근은 Java가 안정적 |
| Kotlin 래퍼 패턴 | 불변성, Result 패턴, 확장 함수 활용 |
| sealed class 예외 | when 표현식으로 exhaustive 검사 가능 |
| 버퍼 재사용 옵션 | 30fps 카메라 처리 시 GC 부담 감소 |

### 학습 내용
- Kotlin data class와 Java mutable class 상호 변환
- sealed class로 타입 안전 예외 계층 설계
- Image.planes의 rowStride/pixelStride 처리
- Kotlin stdlib의 Result 확장 함수

---

## 6. 생성된 파일

### Kotlin 파일
| 파일 | 설명 | 라인 수 |
|------|------|---------|
| `IrisLensSDKKt.kt` | 메인 SDK 싱글톤 | ~420 |
| `IrisResultKt.kt` | 검출 결과 + IrisLandmark + FaceRotation | ~175 |
| `LensConfigKt.kt` | 렌즈 설정 data class | ~140 |
| `ProcessResultKt.kt` | 처리 결과 data class | ~75 |
| `IrisException.kt` | sealed class 예외 계층 | ~180 |
| `FrameFormat.kt` | 프레임 포맷 enum | ~65 |
| `BlendMode.kt` | 블렌드 모드 enum | ~60 |
| `Extensions.kt` | 유틸리티 확장 함수 | ~210 |

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 태스크 문서 생성 |
| 2026-01-12 | Kotlin SDK 클래스 구현 완료 |
| 2026-01-12 | 코드 리뷰 수행 (8.75/10) |
| 2026-01-12 | 버퍼 재사용 및 입력 검증 개선 |
