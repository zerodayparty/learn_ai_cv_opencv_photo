# 🟩 OpenCV 문서 스캐너  

- 한 줄 소개: 휴대폰으로 비스듬히 찍은 종이를 자동으로 찾아 정면에서 본 것처럼 반듯하게 펴 주는 규칙 기반 문서 스캐너  
- 긴 소개: OpenCV와 NumPy만으로 노이즈 제거 → Canny 엣지 → 윤곽선/approxPolyDP → 직접 구현한 꼭짓점 정렬 → 투시 변환 → 적응형 이진화 파이프라인을 구현한다. 트랙바로 파라미터를 실시간 조절하며 단계별 결과를 보고, 20장 테스트셋에 대한 조건별 성공률과 파라미터 스윕으로 "규칙 기반 접근의 한계"를 데이터로 분석한다. 딥러닝 프레임워크와 OCR 라이브러리는 사용하지 않는다.  

<br><br>

## 🟢 README Contents  

- [1. 폴더 구조](#-1-폴더-구조)  
- [2. 설치](#-2-설치)  
- [3. 실행 방법](#-3-실행-방법)  
- [4. 파이프라인](#-4-파이프라인)  
- [5. 요구사항 대응표](#-5-요구사항-대응표)  
- [6. 데이터 준비](#-6-데이터-준비)  

<br><br>

## 🟢 1. 폴더 구조  

```text
learn_ai_cv_opencv_photo/
├── main.py                  # 명령줄 진입점 (scan / batch / sweep / eda / annotate / bench / pdf ...)
├── config.example.json      # 파라미터 설정 예시
├── scanner/
│   ├── config.py            # 파라미터(ScanParams)와 경로 상수
│   ├── pipeline.py          # 전처리 → 엣지 → 윤곽선 → 변환 → 이진화
│   ├── geometry.py          # 꼭짓점 정렬(직접 구현), 종횡비 자동 보정, 투시 변환
│   ├── gui.py               # 트랙바 + 5단계 윈도우
│   ├── dataset.py           # images/metadata.csv 입출력
│   ├── annotate.py          # 정답 꼭짓점 클릭 입력 도구
│   ├── evaluate.py          # 일괄 평가, 조건별 성공률, 파라미터 스윕
│   ├── eda.py               # 히스토그램/밝기 통계/어두운 이미지 엣지 실험
│   ├── convolution.py       # 보너스1: NumPy 가우시안 블러 (for문 vs 벡터화)
│   ├── pdf_export.py        # 보너스3: PDF 내보내기 (Pillow)
│   ├── plotting.py          # matplotlib 한글 폰트 설정
│   ├── synthetic.py         # 동작 확인용 합성 이미지 6장 생성
│   └── photo_factory.py     # 과제용 합성 테스트 사진 20장 + 정답 꼭짓점 생성
├── images/                  # 테스트 이미지 20장 + metadata.csv (generate 로 생성 또는 직접 촬영)
├── reports/
│   ├── analysis_report.md   # 분석 보고서 (템플릿)
│   └── auto/                # 프로그램이 자동 생성하는 표/그래프
├── outputs/                 # 스캔 결과 이미지 (git 제외)
└── docs/requirement.md      # 과제 요구사항
```

<br><br>

## 🟢 2. 설치  

- Python 3.11 이상, [uv](https://docs.astral.sh/uv/) 사용  

```bash
uv sync
```

<br><br>

## 🟢 3. 실행 방법  

| 목적 | 명령어 | 결과 |  
| --- | --- | --- |  
| 경로 입력 후 GUI | `uv run python main.py` | 경로를 물어본 뒤 트랙바 GUI 실행 |  
| 한 장 GUI | `uv run python main.py scan images/a.jpg` | 5개 단계 창 + Trackbar Panel |  
| 한 장 GUI 없이 | `uv run python main.py scan images/a.jpg --no-gui --blur 7 --low 50 --high 150` | `outputs/` 에 단계별 이미지 저장 |  
| 설정 파일 사용 | `uv run python main.py scan images/a.jpg --no-gui --config config.example.json` | JSON 값 적용 |  
| metadata 생성 | `uv run python main.py init` | `images/metadata.csv` 에 파일 이름 등록 |  
| 정답 꼭짓점 입력 | `uv run python main.py annotate` | 클릭 4번 → Enter 저장 |  
| 20장 일괄 평가 | `uv run python main.py batch` | `reports/auto/evaluation_summary.md`, `contact_sheet.jpg` |  
| 파라미터 딜레마 | `uv run python main.py sweep` | `reports/auto/sweep_summary.md`, `sweep_matrix.png` |  
| EDA | `uv run python main.py eda` | `reports/auto/eda_summary.md` + 히스토그램 그림 |  
| 보너스1 | `uv run python main.py bench images/a.jpg --ksize 5` | `reports/auto/bench_convolution.md` |  
| 보너스3 | `uv run python main.py pdf` | `outputs/scanned_all.pdf` |  
| 합성 샘플 | `uv run python main.py sample` | `outputs/synthetic/` (데이터 없을 때 동작 확인용) |  
| 과제용 테스트셋 | `uv run python main.py generate` | `images/` 에 사진 20장 + 정답 꼭짓점이 채워진 `metadata.csv` |  

- GUI 키: `s` = 결과 저장, `p` = PDF 저장, `q` / `ESC` = 종료  
- 트랙바: Blur Kernel, Canny Low, Canny High, Dilate Iter, Thresh Block, Thresh C  

<br><br>

## 🟢 4. 파이프라인  

```text
원본(BGR) → 축소(높이 800) → Grayscale → GaussianBlur → Canny → Dilate
→ findContours(외곽) → 면적 필터(≥10%) → approxPolyDP(4점 + 볼록)
→ order_points(무게중심 각도 정렬, 직접 구현) → 원본 해상도로 좌표 복원
→ getPerspectiveTransform + warpPerspective(종횡비 자동) → adaptiveThreshold → 저장
```

<br><br>

## 🟢 5. 요구사항 대응표  

| 요구사항 | 구현 위치 |  
| --- | --- |  
| 이미지 경로 입력 즉시 동작 | `main.py` cmd_scan |  
| 단계별 5개 윈도우 + 트랙바 | `scanner/gui.py` |  
| Grayscale + 가우시안 블러 + Canny | `scanner/pipeline.py` preprocess, detect_edges |  
| 닫힌 윤곽선 + 면적 필터 + approxPolyDP 4점 + 시각화 | `scanner/pipeline.py` find_document_quad, run_pipeline |  
| 좌표 정렬 직접 구현 (좌상-우상-우하-좌하) | `scanner/geometry.py` order_points |  
| 투시 변환 | `scanner/geometry.py` warp_document |  
| 적응형 이진화 + jpg/png 저장 | `scanner/pipeline.py` postprocess, `save_image` |  
| EDA (히스토그램, 쉬움/어려움 비교, 어두운 이미지 설명) | `scanner/eda.py` |  
| 20장 성공 여부 + 조건별 성공률 표 | `scanner/evaluate.py` evaluate |  
| 파라미터 딜레마 데이터 | `scanner/evaluate.py` sweep |  
| GUI 불가 환경 대체 (imwrite + 명령줄/설정 파일) | `--no-gui`, `--config` |  
| 보너스1 컨볼루션 직접 구현 + 속도 비교 | `scanner/convolution.py` |  
| 보너스2 종횡비 자동 보정 | `scanner/geometry.py` output_size |  
| 보너스3 PDF 내보내기 | `scanner/pdf_export.py` |  
| 웹캠 (선택) | 구현하지 않음 |  

<br><br>

## 🟢 6. 데이터 준비  

- `images/README.md` 참고  
- 사진 20장을 `images/` 에 넣고 → `init` → `metadata.csv` 조건 입력 → `annotate` → `batch` / `sweep` / `eda` 순서로 실행  
- 현재 `images/` 에는 `generate` 로 만든 합성 테스트셋 20장이 들어 있다. 정답 꼭짓점과 촬영 조건이 이미 기록되어 있어 바로 `batch` 를 실행할 수 있다.  
- 실제 사진으로 바꾸려면 합성 사진을 지우고 직접 찍은 사진을 넣은 뒤 `init` 부터 진행한다.  
