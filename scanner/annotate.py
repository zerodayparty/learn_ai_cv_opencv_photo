"""정답(Ground Truth) 꼭짓점 4개를 마우스로 클릭해 metadata.csv에 기록하는 도구."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

from pathlib import Path  # 경로 처리

import cv2  # 창 표시와 마우스 이벤트
import numpy as np  # 좌표 계산

from .config import IMAGES_DIR  # 기본 이미지 폴더
from .dataset import CORNER_COLS, load_records, read_metadata, write_metadata  # 메타데이터 입출력
from .geometry import order_points  # 클릭 순서와 상관없이 정렬
from .pipeline import load_image  # 이미지 읽기

WIN = "Annotate (click 4 corners | Enter=save r=reset n=skip q=quit)"  # 창 이름 겸 사용법
VIEW_H = 900  # 화면 표시 높이


def annotate(folder: Path = IMAGES_DIR, only_missing: bool = True) -> None:  # 모든 이미지에 정답 좌표 입력
    csv_path = folder / "metadata.csv"  # 메타데이터 CSV 경로
    records = load_records(folder, csv_path)  # 이미지 + CSV 레코드 목록
    rows = read_metadata(csv_path)  # 수정할 CSV 행들
    todo = [r for r in records if not (only_missing and r.gt_corners is not None)]  # 이미 입력된 이미지는 건너뜀
    print(f"[annotate] 대상 {len(todo)}장 / 전체 {len(records)}장")  # 진행 정보
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)  # 창 생성
    for rec in todo:  # 대상 이미지마다
        img = load_image(rec.path)  # 원본 읽기
        scale = VIEW_H / img.shape[0]  # 화면 축소 배율
        view = cv2.resize(img, (int(img.shape[1] * scale), VIEW_H))  # 화면용 축소본
        clicks: list[tuple[float, float]] = []  # 클릭한 원본 좌표 목록

        def on_mouse(event, x, y, flags, param):  # 마우스 이벤트 처리 함수
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:  # 왼쪽 클릭 & 4개 미만이면
                clicks.append((x / scale, y / scale))  # 원본 해상도 좌표로 바꿔 저장

        cv2.setMouseCallback(WIN, on_mouse)  # 창에 마우스 함수 연결
        while True:  # 사용자가 결정할 때까지
            canvas = view.copy()  # 매번 깨끗한 화면에서 그림
            for x, y in clicks:  # 클릭한 점마다
                cv2.circle(canvas, (int(x * scale), int(y * scale)), 6, (0, 0, 255), -1)  # 빨간 점 표시
            if len(clicks) == 4:  # 4개를 다 찍었으면
                poly = (order_points(np.array(clicks)) * scale).astype(np.int32)  # 정렬된 사각형 좌표
                cv2.polylines(canvas, [poly], True, (0, 255, 0), 2)  # 초록 사각형으로 확인
            cv2.putText(canvas, f"{rec.name} ({len(clicks)}/4)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # 파일명/진행
            cv2.imshow(WIN, canvas)  # 화면 표시
            key = cv2.waitKey(20) & 0xFF  # 키 입력 대기
            if key == ord("r"):  # r: 다시 찍기
                clicks.clear()  # 클릭 초기화
            elif key == ord("n"):  # n: 이 이미지 건너뛰기
                break  # 다음 이미지로
            elif key in (13, 10) and len(clicks) == 4:  # Enter: 저장
                ordered = order_points(np.array(clicks))  # 좌상-우상-우하-좌하 정렬
                for col, val in zip(CORNER_COLS, ordered.reshape(-1)):  # 8개 열에
                    rows[rec.name][col] = f"{val:.1f}"  # 소수점 한 자리로 기록
                write_metadata(rows, csv_path)  # 즉시 CSV 저장 (중간에 꺼져도 안전)
                print(f"[저장] {rec.name}")  # 저장 안내
                break  # 다음 이미지로
            elif key in (ord("q"), 27):  # q/ESC: 전체 종료
                cv2.destroyAllWindows()  # 창 닫기
                return  # 함수 종료
    cv2.destroyAllWindows()  # 모든 작업 후 창 닫기
