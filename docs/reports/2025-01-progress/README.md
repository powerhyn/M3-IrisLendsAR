# IrisLensSDK 2025년 1월 진행 보고서

## 문서 구성

| 파일 | 내용 | Notion 용도 |
|------|------|-------------|
| `01_architecture_comparison.md` | 아키텍처 및 솔루션 비교 | 기술/비즈니스 의사결정 배경 |
| `02_features_achievements.md` | 기능 및 성과 | 데모 및 진행 현황 |

## 이미지 구성

### 스크린샷 (Android 데모 앱)
```
screenshots/
├── 1000011373.jpg  → 렌즈 피팅 1
├── 1000011375.jpg  → 렌즈 피팅 2
└── 1000011377.jpg  → 홍채 추적
```

### 테스트 이미지 (C++ 코어 엔진)
문서에서 `shared/test_data/` 폴더의 이미지를 참조합니다:
- `iris_test_01.png` - 입력 이미지
- `iris_test_01_output.png` - 렌즈 적용 결과

## Notion 업로드

### 방법 1: Markdown Import
1. Notion 페이지에서 `/import` → Markdown 선택
2. `.md` 파일 업로드
3. 이미지는 수동으로 드래그앤드롭

### 방법 2: 복사 붙여넣기
1. `.md` 파일 내용 복사
2. Notion 페이지에 붙여넣기
3. 이미지는 수동으로 추가

### 이미지 경로 참고
Notion 업로드 시 상대 경로가 작동하지 않으므로, 이미지를 직접 드래그앤드롭하세요.
