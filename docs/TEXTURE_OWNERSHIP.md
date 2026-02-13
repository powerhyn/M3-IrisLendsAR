# Texture Ownership Convention

## 개요

IrisLensSDK의 GPU 파이프라인에서 OpenGL 텍스처의 소유권과 생명주기를 정의합니다.
이 규약을 준수하지 않으면 이중 해제(double-free), 댕글링 텍스처, 검은 화면 등의 문제가 발생합니다.

## 소유권 규칙

### 1. TexturePool 관리 텍스처

TexturePool이 생성한 텍스처는 **반드시 TexturePool을 통해 해제**해야 합니다.

```
할당: TexturePool::acquireTexture()
해제: TexturePool::releaseTexture(TextureInfo*) 또는 releaseTextureById(GLuint)
소멸: TexturePool 소멸자에서 일괄 glDeleteTextures
```

**금지 사항**:
- Pool 관리 텍스처에 대해 직접 `glDeleteTextures()` 호출 금지
- Pool 외부에서 `TextureInfo*` 포인터 캐싱 금지 (매 프레임 `acquireTexture()` 사용)

### 2. 외부(앱) 소유 텍스처

앱이 생성한 텍스처(카메라 텍스처, LUT 텍스처 등)는 **앱이 직접 해제**합니다.

```
할당: 앱에서 glGenTextures()
SDK 전달: applyBeautyFilterTextureV2(inputTexture, ...) — 읽기 전용
해제: 앱에서 glDeleteTextures()
```

**규칙**:
- SDK는 입력 텍스처를 수정하지 않음 (read-only)
- SDK는 입력 텍스처를 해제하지 않음

### 3. SDK 출력 텍스처

`applyBeautyFilterTextureV2()` 등이 반환하는 텍스처는 **TexturePool 소유**입니다.

```
반환: applyBeautyFilterTextureV2() → outputTexture
사용: 앱에서 렌더링에 사용 (read-only)
해제: 다음 프레임 호출 시 TexturePool이 자동 재활용
```

**금지 사항**:
- 출력 텍스처에 대해 `IrisLensSDK.releaseTexture()` 호출 금지 → 이중 해제 발생
- 렌더러 종료(`release()`)에서도 호출 금지 → `releaseGpuBeauty()`가 일괄 정리
- 출력 텍스처를 다음 프레임 이후까지 캐싱 금지 (재활용됨)

**Passthrough 주의**:
- 필터가 모두 비활성이면 입력 텍스처를 그대로 반환 (Pool 텍스처가 아님)
- `sdk_api_v2`는 passthrough 시 입력 텍스처를 관리 목록에 등록하지 않음

## GPUBeautyBackend 해제 순서

```cpp
// 올바른 순서 (sdk_api_v2.cpp)
g_gpu_beauty->release();        // 1. GL 리소스 해제 (셰이더, FBO, Pool 내 텍스처)
g_managed_textures.clear();     // 2. 추적 컨테이너 정리
g_gpu_beauty.reset();           // 3. 객체 소멸
```

**잘못된 순서**:
- `reset()` → `release()`: 이미 소멸된 객체에 접근 → UAF
- `clear()` → `release()`: 추적 정보 없이 해제 → 누수 가능

## releaseTextureById 안전 메커니즘

```cpp
bool TexturePool::releaseTextureById(GLuint texture_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Pool 내부 검색 → 찾으면 안전하게 해제
    // 찾지 못하면 false 반환 (외부 텍스처이므로 호출자가 처리)
}
```

`GPUBeautyBackend::releaseTexture()`는 이 메서드를 먼저 시도하고,
Pool에 없는 텍스처만 `glDeleteTextures()`로 직접 해제합니다.

## Detection Slot과 텍스처 독립성

Detection Slot(더블 버퍼)은 텍스처 소유권과 독립적입니다:
- Detection Slot: IrisResult 데이터 전달 (Analyzer → GL 스레드)
- TexturePool: GPU 텍스처 관리 (GL 스레드 내부)

두 시스템은 서로의 리소스에 접근하지 않습니다.

## 체크리스트

| 상황 | 올바른 처리 |
|------|------------|
| 뷰티 필터 출력 텍스처 해제 | 하지 않음 (Pool이 자동 관리, 종료 시에도 호출 금지) |
| 카메라 OES 텍스처 해제 | 앱이 `glDeleteTextures()` |
| LUT 3D 텍스처 해제 | 앱이 `glDeleteTextures()` |
| SDK 종료 시 텍스처 | `releaseGpuBeauty()` 호출 (내부에서 일괄 정리) |
| 텍스처 ID 유효성 확인 | `IrisLensSDK.isTextureManaged(id)` |
