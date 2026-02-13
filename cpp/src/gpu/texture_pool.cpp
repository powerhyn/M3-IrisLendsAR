/**
 * @file texture_pool.cpp
 * @brief TexturePool 구현
 */

#include "iris_sdk/gpu/texture_pool.h"

#include <algorithm>

#if IRIS_SDK_GPU_AVAILABLE
#include <android/log.h>
#define LOG_TAG "TexturePool"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) printf("[TexturePool INFO] " __VA_ARGS__); printf("\n")
#define LOGW(...) printf("[TexturePool WARN] " __VA_ARGS__); printf("\n")
#define LOGE(...) printf("[TexturePool ERROR] " __VA_ARGS__); printf("\n")
#endif

namespace iris_sdk {

TexturePool::TexturePool() = default;

TexturePool::~TexturePool() {
    release();
}

bool TexturePool::initialize(int max_textures, int max_width, int max_height) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        LOGW("TexturePool already initialized");
        return true;
    }

    // 파라미터 검증
    max_textures_ = std::clamp(max_textures, 1, 16);
    max_width_ = std::max(1, max_width);
    max_height_ = std::max(1, max_height);

    // 텍스처는 요청 시 생성 (lazy allocation)
    textures_.reserve(static_cast<size_t>(max_textures_));

    initialized_ = true;
    LOGI("TexturePool initialized: max=%d, max_size=%dx%d",
         max_textures_, max_width_, max_height_);

    return true;
}

void TexturePool::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

#if IRIS_SDK_GPU_AVAILABLE
    for (const auto& info : textures_) {
        if (info->fbo_id != 0) {
            glDeleteFramebuffers(1, &info->fbo_id);
        }
        if (info->texture_id != 0) {
            glDeleteTextures(1, &info->texture_id);
        }
    }
#endif

    size_t count = textures_.size();
    textures_.clear();
    last_used_time_.clear();

    initialized_ = false;
    LOGI("TexturePool released: %zu textures freed", count);
}

TexturePool::TextureInfo* TexturePool::acquireRenderTargetLocked(int width, int height) {
    // mutex_ 이미 획득된 상태에서 호출됨

    if (!initialized_) {
        LOGE("TexturePool not initialized");
        return nullptr;
    }

    // 크기 제한 확인
    if (width > max_width_ || height > max_height_) {
        LOGE("Requested size %dx%d exceeds max %dx%d",
             width, height, max_width_, max_height_);
        return nullptr;
    }

    // 기존 텍스처에서 검색
    TextureInfo* info = findAvailable(width, height);

    // 없으면 새로 생성
    if (!info) {
        if (static_cast<int>(textures_.size()) >= max_textures_) {
            LOGE("TexturePool full: %zu/%d", textures_.size(), max_textures_);
            return nullptr;
        }
        info = createTexture(width, height);
    }

    if (info) {
        info->in_use = true;
        last_used_time_[info->texture_id] = currentTimeMs();
    }

    return info;
}

void TexturePool::releaseTextureLocked(TextureInfo* info) {
    // mutex_ 이미 획득된 상태에서 호출됨
    if (!info) return;
    info->in_use = false;
    last_used_time_[info->texture_id] = currentTimeMs();
}

TexturePool::TextureInfo* TexturePool::acquireRenderTarget(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    return acquireRenderTargetLocked(width, height);
}

void TexturePool::releaseTexture(TextureInfo* info) {
    if (!info) return;
    std::lock_guard<std::mutex> lock(mutex_);
    releaseTextureLocked(info);
}

bool TexturePool::releaseTextureById(GLuint texture_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& info : textures_) {
        if (info->texture_id == texture_id) {
            if (!info->in_use) return false;  // Already released
            releaseTextureLocked(info.get());
            return true;
        }
    }
    return false;  // Not managed by pool
}

bool TexturePool::acquirePingPongPair(int width, int height,
                                       TextureInfo*& ping, TextureInfo*& pong) {
    std::lock_guard<std::mutex> lock(mutex_);

    ping = acquireRenderTargetLocked(width, height);
    if (!ping) {
        return false;
    }

    pong = acquireRenderTargetLocked(width, height);
    if (!pong) {
        releaseTextureLocked(ping);
        ping = nullptr;
        return false;
    }

    return true;
}

void TexturePool::releaseAllTextures() {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = currentTimeMs();
    for (auto& info : textures_) {
        info->in_use = false;
        last_used_time_[info->texture_id] = now;
    }

    LOGI("All %zu textures released to pool", textures_.size());
}

int TexturePool::trim(int64_t max_idle_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = currentTimeMs();
    int released_count = 0;

    auto it = textures_.begin();
    while (it != textures_.end()) {
        TextureInfo* info = it->get();

        // 사용 중인 텍스처는 건너뜀
        if (info->in_use) {
            ++it;
            continue;
        }

        // 유휴 시간 확인
        auto time_it = last_used_time_.find(info->texture_id);
        if (time_it != last_used_time_.end()) {
            int64_t idle_time = now - time_it->second;

            if (idle_time > max_idle_ms) {
                // GL 리소스 해제
#if IRIS_SDK_GPU_AVAILABLE
                if (info->fbo_id != 0) {
                    glDeleteFramebuffers(1, &info->fbo_id);
                }
                if (info->texture_id != 0) {
                    glDeleteTextures(1, &info->texture_id);
                }
#endif

                last_used_time_.erase(time_it);
                it = textures_.erase(it);
                released_count++;
                continue;
            }
        }
        ++it;
    }

    if (released_count > 0) {
        LOGI("TexturePool::trim() released %d textures", released_count);
    }
    return released_count;
}

void TexturePool::resizePool(int new_max_textures) {
    std::lock_guard<std::mutex> lock(mutex_);

    new_max_textures = std::clamp(new_max_textures, 1, 16);

    // 현재 텍스처 수가 새 최대값보다 크면 미사용 텍스처 해제
    while (static_cast<int>(textures_.size()) > new_max_textures) {
        // 미사용 텍스처 찾기
        auto it = std::find_if(textures_.begin(), textures_.end(),
            [](const auto& info) { return !info->in_use; });

        if (it == textures_.end()) {
            // 모든 텍스처가 사용 중
            break;
        }

        TextureInfo* info = it->get();
#if IRIS_SDK_GPU_AVAILABLE
        if (info->fbo_id != 0) {
            glDeleteFramebuffers(1, &info->fbo_id);
        }
        if (info->texture_id != 0) {
            glDeleteTextures(1, &info->texture_id);
        }
#endif
        last_used_time_.erase(info->texture_id);
        textures_.erase(it);
    }

    max_textures_ = new_max_textures;
    LOGI("TexturePool resized to %d textures", max_textures_);
}

size_t TexturePool::getUsedMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& info : textures_) {
        total += info->memorySize();
    }
    return total;
}

TexturePool::PoolStats TexturePool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    PoolStats stats = {};
    stats.total_textures = static_cast<int>(textures_.size());

    for (const auto& info : textures_) {
        if (info->in_use) {
            stats.in_use++;
        } else {
            stats.available++;
        }
        stats.total_memory_bytes += info->memorySize();
    }

    return stats;
}

void TexturePool::setMemoryPressureCallback(MemoryPressureCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_pressure_callback_ = std::move(callback);
}

void TexturePool::onMemoryPressure(int level) {
    // Android ComponentCallbacks2 상수:
    // TRIM_MEMORY_RUNNING_LOW = 10
    // TRIM_MEMORY_RUNNING_CRITICAL = 15
    // TRIM_MEMORY_UI_HIDDEN = 20
    // TRIM_MEMORY_BACKGROUND = 40
    // TRIM_MEMORY_MODERATE = 60
    // TRIM_MEMORY_COMPLETE = 80

    if (level >= 60) {
        // MODERATE 이상: 공격적 정리
        trim(0);  // 모든 미사용 텍스처 즉시 해제
        resizePool(4);  // 최소 풀 크기
        LOGW("Memory pressure (level=%d): aggressive cleanup", level);
    } else if (level >= 40) {
        // BACKGROUND: 적당한 정리
        trim(1000);  // 1초 이상 유휴 텍스처 해제
        LOGI("Memory pressure (level=%d): moderate cleanup", level);
    } else if (level >= 10) {
        // RUNNING_LOW: 가벼운 정리
        trim(5000);  // 5초 이상 유휴 텍스처 해제
        LOGI("Memory pressure (level=%d): light cleanup", level);
    }

    // 콜백 호출
    if (memory_pressure_callback_) {
        memory_pressure_callback_(level);
    }
}

TexturePool::TextureInfo* TexturePool::createTexture(int width, int height) {
    auto info = std::make_unique<TextureInfo>();
    info->width = width;
    info->height = height;
    info->format = GL_RGBA;

#if IRIS_SDK_GPU_AVAILABLE
    // 텍스처 생성
    glGenTextures(1, &info->texture_id);
    glBindTexture(GL_TEXTURE_2D, info->texture_id);

    // 텍스처 파라미터 설정
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 텍스처 스토리지 할당
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // FBO 생성 (렌더 타겟용)
    glGenFramebuffers(1, &info->fbo_id);
    glBindFramebuffer(GL_FRAMEBUFFER, info->fbo_id);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, info->texture_id, 0);

    // FBO 완전성 확인
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("Framebuffer incomplete: 0x%x", status);
        glDeleteFramebuffers(1, &info->fbo_id);
        glDeleteTextures(1, &info->texture_id);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        return nullptr;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    LOGI("Created texture %u with FBO %u (%dx%d)",
         info->texture_id, info->fbo_id, width, height);
#else
    // Desktop 스텁
    static GLuint stub_tex_counter = 1;
    static GLuint stub_fbo_counter = 1;
    info->texture_id = stub_tex_counter++;
    info->fbo_id = stub_fbo_counter++;
    LOGI("Created stub texture %u with FBO %u (%dx%d)",
         info->texture_id, info->fbo_id, width, height);
#endif

    TextureInfo* raw_ptr = info.get();
    textures_.push_back(std::move(info));

    return raw_ptr;
}

TexturePool::TextureInfo* TexturePool::findAvailable(int width, int height) {
    // 정확한 크기 매칭 우선
    for (auto& info : textures_) {
        if (!info->in_use &&
            info->width == width &&
            info->height == height) {
            return info.get();
        }
    }

    // 더 큰 텍스처 재사용 (크기가 가까운 것 우선)
    TextureInfo* best_match = nullptr;
    int best_waste = INT32_MAX;

    for (auto& info : textures_) {
        if (!info->in_use &&
            info->width >= width &&
            info->height >= height) {
            int waste = (info->width - width) * (info->height - height);
            if (waste < best_waste) {
                best_waste = waste;
                best_match = info.get();
            }
        }
    }

    return best_match;
}

int64_t TexturePool::currentTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace iris_sdk
