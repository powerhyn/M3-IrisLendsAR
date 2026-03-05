# Change Categorization

## Primary Type: `feat` (New Feature)
## Scope: `beauty`
## Commit Structure: Single commit

### Rationale
- All changes are part of one cohesive feature (P4-W3-02: Frequency Separation GPU pipeline)
- Bug fixes were applied before first commit (code was never in broken state)
- Tests cover the same feature
- Follows project convention: single commit per work paper task (e.g., `feat(render): ... [P4-W2-02]`)

### Change Groups
1. **GPU Shaders** (shader_sources.cpp): FREQ_SEP_GAUSSIAN + COMPOSITE
2. **Pipeline** (gpu_beauty_backend.h/cpp): 5-subpass pipeline, skin mask upload, mapSkinQuality, fallback
3. **API** (beauty_filter.h, sdk_api.h, sdk_api_v2.cpp, JNI, Java): skinQuality field
4. **Safety** (gpu_beauty_backend.cpp): buffer validation, #if guard, scissor management, bool return
5. **Tests** (test_beauty_config_v2.cpp): 15 FreqSepParams + 2 skinQuality validation tests
