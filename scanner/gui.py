"""트랙바로 파라미터를 실시간 조절하며 5단계 윈도우를 보여주는 GUI."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

from pathlib import Path  # 경로 처리

import cv2  # OpenCV GUI(imshow, trackbar) 사용
import numpy as np  # 배열 처리

from .config import OUTPUT_DIR, ScanParams  # 저장 폴더와 파라미터
from .pipeline import ScanResult, run_pipeline, save_image  # 파이프라인 실행/저장

PANEL = "Trackbar Panel"  # 트랙바 창 이름
WINDOWS = ["1 Original", "2 Preprocessed", "3 Edges", "4 Contours", "5 Result"]  # 단계별 창 이름 5개
VIEW_H = 420  # 화면에 보여줄 때 각 창의 높이(px)


def _fit(img: np.ndarray, h: int = VIEW_H) -> np.ndarray:  # 화면 표시용으로 높이를 맞춰 줄임
    scale = h / img.shape[0]  # 줄일 배율
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h), interpolation=cv2.INTER_AREA)  # 비율 유지 축소


def _placeholder(text: str) -> np.ndarray:  # 결과가 없을 때 보여줄 검은 화면
    img = np.zeros((VIEW_H, 300, 3), np.uint8)  # 검은 바탕 이미지 생성
    cv2.putText(img, text, (20, VIEW_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 안내 문구 출력
    return img  # 이미지 반환


def _nothing(_: int) -> None:  # 트랙바 콜백용 빈 함수 (값은 루프에서 직접 읽음)
    pass  # 아무것도 하지 않음


def _create_panel(p: ScanParams) -> None:  # 트랙바 패널 생성
    cv2.namedWindow(PANEL, cv2.WINDOW_NORMAL)  # 크기 조절 가능한 창 생성
    cv2.resizeWindow(PANEL, 520, 330)  # 창 크기 지정
    cv2.createTrackbar("Blur Kernel", PANEL, p.blur_kernel, 31, _nothing)  # 블러 커널 크기 (0~31)
    cv2.createTrackbar("Canny Low", PANEL, p.canny_low, 500, _nothing)  # Canny 낮은 임계값 (0~500)
    cv2.createTrackbar("Canny High", PANEL, p.canny_high, 500, _nothing)  # Canny 높은 임계값 (0~500)
    cv2.createTrackbar("Dilate Iter", PANEL, p.dilate_iter, 5, _nothing)  # 엣지 팽창 횟수 (0~5)
    cv2.createTrackbar("Thresh Block", PANEL, p.thresh_block, 101, _nothing)  # 이진화 블록 크기 (3~101)
    cv2.setTrackbarMin("Thresh Block", PANEL, 3)  # 블록 최소값 3
    cv2.createTrackbar("Thresh C", PANEL, p.thresh_c, 50, _nothing)  # 이진화 상수 C (0~50)


def _read_panel(base: ScanParams) -> ScanParams:  # 트랙바 현재 값을 읽어 파라미터로 만듦
    p = ScanParams(**base.to_dict())  # 기본 설정 복사
    p.blur_kernel = cv2.getTrackbarPos("Blur Kernel", PANEL)  # 블러 커널 값 읽기
    p.canny_low = cv2.getTrackbarPos("Canny Low", PANEL)  # Canny low 읽기
    p.canny_high = cv2.getTrackbarPos("Canny High", PANEL)  # Canny high 읽기
    p.dilate_iter = cv2.getTrackbarPos("Dilate Iter", PANEL)  # 팽창 횟수 읽기
    p.thresh_block = cv2.getTrackbarPos("Thresh Block", PANEL)  # 이진화 블록 읽기
    p.thresh_c = cv2.getTrackbarPos("Thresh C", PANEL)  # 이진화 C 읽기
    return p.normalized()  # 짝수 커널 등을 보정해 반환


def _show(result: ScanResult, p: ScanParams) -> None:  # 5개 창에 단계별 결과 표시
    cv2.imshow(WINDOWS[0], _fit(result.resized))  # 1) 원본
    cv2.imshow(WINDOWS[1], _fit(result.blurred))  # 2) 그레이 + 블러
    cv2.imshow(WINDOWS[2], _fit(result.edges))  # 3) Canny 엣지
    cv2.imshow(WINDOWS[3], _fit(result.contour_vis))  # 4) 윤곽선 + 검출 사각형
    final = result.scanned if result.found else None  # 최종 스캔본 (없으면 None)
    cv2.imshow(WINDOWS[4], _fit(final) if final is not None else _placeholder("Document not found"))  # 5) 결과
    status = f"blur={p.blur_kernel} low={p.canny_low} high={p.canny_high} found={result.found}"  # 상태 문자열
    cv2.setWindowTitle(PANEL, f"{PANEL} | {status}")  # 트랙바 창 제목에 현재 상태 표시


def _arrange() -> None:  # 창들을 겹치지 않게 배치
    for i, name in enumerate(WINDOWS):  # 각 창에 대해
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)  # 이미지 크기에 맞는 창 생성
        cv2.moveWindow(name, 20 + (i % 3) * 360, 20 + (i // 3) * (VIEW_H + 60))  # 3열 격자로 배치
    cv2.moveWindow(PANEL, 20 + 2 * 360, 20 + VIEW_H + 60)  # 트랙바 패널은 오른쪽 아래


def run_gui(image: np.ndarray, base: ScanParams, out_name: str, out_dir: Path = OUTPUT_DIR) -> ScanResult | None:  # GUI 메인 루프
    """키 안내: s=결과 저장, p=PDF 저장, q 또는 ESC=종료."""  # 함수 설명
    _create_panel(base)  # 트랙바 패널 생성
    _arrange()  # 창 배치
    last_params = None  # 직전 파라미터 (값이 바뀔 때만 다시 계산하기 위함)
    result = None  # 최근 결과
    print("[GUI] s=저장, p=PDF 저장, q/ESC=종료")  # 사용법 출력
    while True:  # 사용자가 종료할 때까지 반복
        p = _read_panel(base)  # 트랙바 값 읽기
        if p != last_params:  # 값이 바뀌었을 때만
            result = run_pipeline(image, p)  # 파이프라인 다시 실행
            _show(result, p)  # 화면 갱신
            last_params = p  # 현재 값을 기억
        key = cv2.waitKey(30) & 0xFF  # 30ms 동안 키 입력 대기
        if key in (ord("q"), 27):  # q 또는 ESC 키면
            break  # 루프 종료
        if key == ord("s") and result is not None:  # s 키면
            if result.found:  # 문서를 찾았을 때만
                path = save_image(out_dir / f"scanned_{out_name}.png", result.scanned)  # 이진화 스캔본 저장
                save_image(out_dir / f"warped_{out_name}.jpg", result.warped)  # 컬러 변환본 저장
                print(f"[저장] {path.name} (params={p.to_dict()})")  # 저장 안내
            else:  # 문서를 못 찾았으면
                print("[저장 실패] 문서를 찾지 못했습니다. 파라미터를 조절하세요.")  # 실패 안내
        if key == ord("p") and result is not None and result.found:  # p 키면
            from .pdf_export import export_pdf  # 필요할 때만 PDF 모듈 불러오기
            pdf = export_pdf([result.scanned], out_dir / f"scanned_{out_name}.pdf")  # PDF로 저장
            print(f"[PDF 저장] {pdf.name}")  # 저장 안내
        if cv2.getWindowProperty(PANEL, cv2.WND_PROP_VISIBLE) < 1:  # 트랙바 창을 닫았으면
            break  # 종료
    cv2.destroyAllWindows()  # 모든 창 닫기
    cv2.waitKey(1)  # macOS에서 창이 실제로 닫히도록 이벤트 한 번 처리
    return result  # 마지막 결과 반환
