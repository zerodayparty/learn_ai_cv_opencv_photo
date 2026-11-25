# 🟩 테스트 이미지 폴더 안내  

<br>

## 🟢 1. 넣어야 할 이미지 (총 20장)  

| category 값 | 조건 | 장수 | 촬영 방법 |  
| --- | --- | --- | --- |  
| `easy` | 단순 배경 + 정면 + 균일 조명 | 5 | 단색 책상 위, 위에서 수직으로, 그림자 없이 |  
| `shadow` | 그림자 있음 | 5 | 손/폰/창틀 그림자가 문서 위에 걸치게 (약한 것 + 강한 것 섞기) |  
| `tilt` | 45도 이상 기울어짐 | 5 | 문서를 45도 이상 돌리거나, 카메라를 비스듬히 눕혀서 |  
| `complex` | 복잡한 배경 | 5 | 키보드, 책, 무늬 천, 다른 종이가 주변에 있는 상태 |  

- 추가 촬영(보너스)은 `crumpled`(구겨짐), `occlusion`(부분 가림), `glare`(빛 반사) 값을 쓰면 표에 자동으로 따로 집계된다.  
- 문서 종류는 영수증, A4, 명함을 섞는다.  
- 파일 형식: `.jpg` / `.png` (휴대폰 기본 해상도 그대로 OK)  
- 파일 이름 예시: `easy_01.jpg`, `shadow_03.jpg` (조건이 보이게 짓는 것을 추천)  

<br><br>

## 🟢 2. metadata.csv 에 적어야 할 내용  

- 이미지를 넣은 뒤 `uv run python main.py init` 을 실행하면 파일 이름이 자동으로 채워진다.  
- 나머지 열은 엑셀/VS Code 로 직접 입력한다.  

| 열 | 필수 | 값 예시 | 설명 |  
| --- | --- | --- | --- |  
| `filename` | 자동 | `easy_01.jpg` | init 이 채움 |  
| `category` | 필수 | `easy` / `shadow` / `tilt` / `complex` | 보고서 1번 표의 행 |  
| `lighting` | 필수 | `uniform` / `dim` / `shadow` / `glare` | 조명별 성공률 표 |  
| `angle` | 필수 | `front` / `tilt30` / `tilt45+` | 각도별 성공률 표 |  
| `background` | 필수 | `simple` / `complex` | 배경별 성공률 표 |  
| `doc_type` | 권장 | `a4` / `receipt` / `card` | 문서 종류 |  
| `true_aspect` | 권장 | `1.414` | 실제 문서의 긴 변 ÷ 짧은 변. A4=1.414, 명함(90×50mm)=1.8, 영수증은 자로 재서 계산 |  
| `notes` | 선택 | `창문 역광` | 촬영 메모 |  
| `tl_x` ~ `bl_y` | 권장 | 픽셀 좌표 | 정답 꼭짓점 8개. `uv run python main.py annotate` 로 클릭 입력 |  
| `manual_result` | 대안 | `success` / `partial` / `fail` | 정답 좌표를 안 넣을 때 사람이 직접 판정 |  

<br><br>

## 🟢 3. 성공 판정 방식  

- 정답 꼭짓점이 있으면 자동 판정: 4꼭짓점 최대 오차가 이미지 대각선의 3% 이하면 성공  
- `true_aspect` 까지 있으면: 꼭짓점은 맞지만 종횡비가 5% 넘게 다르면 부분 성공  
- 정답 좌표가 없으면 `manual_result` 값 사용, 그것도 없으면 `detected`(미검증)로만 표시  
