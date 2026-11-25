"""동작 확인용 합성 테스트 이미지 생성기 (실제 촬영 데이터를 대신하지 않음, 파이프라인 점검용)."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

from pathlib import Path  # 경로 처리

import cv2  # 그리기/투시 변환
import numpy as np  # 난수와 배열

from .dataset import COLUMNS, CORNER_COLS, write_metadata  # 메타데이터 저장
from .geometry import order_points  # 정답 좌표 정렬
from .pipeline import save_image  # 이미지 저장


def _document(w: int = 840, h: int = 1188) -> np.ndarray:  # A4 비율(1:1.414)의 가짜 문서 생성
    doc = np.full((h, w, 3), 245, np.uint8)  # 거의 흰 종이
    cv2.putText(doc, "SAMPLE DOCUMENT", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3)  # 제목 글씨
    for i in range(22):  # 본문 줄 22개
        y = 200 + i * 42  # 줄 위치
        cv2.putText(doc, "lorem ipsum dolor sit amet 0123456789", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)  # 본문 글씨
    return doc  # 문서 반환


def _background(h: int, w: int, rng: np.random.Generator, complex_bg: bool) -> np.ndarray:  # 배경 생성
    bg = np.zeros((h, w, 3), np.uint8)  # 빈 배경
    bg[:] = (60, 110, 160)  # 나무 책상 느낌 갈색 (BGR)
    noise = rng.normal(0, 8, (h, w, 1))  # 약한 질감 노이즈
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)  # 노이즈 추가
    if complex_bg:  # 복잡한 배경이면
        for _ in range(40):  # 무작위 도형 40개
            c = tuple(int(v) for v in rng.integers(0, 255, 3))  # 무작위 색
            p1 = tuple(int(v) for v in rng.integers(0, [w, h]))  # 무작위 시작점
            p2 = tuple(int(v) for v in rng.integers(0, [w, h]))  # 무작위 끝점
            cv2.line(bg, p1, p2, c, int(rng.integers(2, 12)))  # 선 그리기 (엣지 방해물)
    return bg  # 배경 반환


def make_samples(folder: Path, seed: int = 0) -> list[Path]:  # 합성 이미지 세트 생성
    rng = np.random.default_rng(seed)  # 재현 가능한 난수 생성기
    folder.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    doc = _document()  # 문서 원본
    dh, dw = doc.shape[:2]  # 문서 크기
    src = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])  # 문서 원래 꼭짓점
    cases = [  # (이름, 분류, 배경 복잡, 그림자, 회전 각도, 원근 강도)
        ("easy_01", "easy", False, False, 3, 0.02), ("easy_02", "easy", False, False, -5, 0.04),  # 쉬운 케이스
        ("tilt_01", "tilt", False, False, 45, 0.08), ("tilt_02", "tilt", False, False, -50, 0.12),  # 기울어짐
        ("shadow_01", "shadow", False, True, 10, 0.05), ("complex_01", "complex", True, False, 15, 0.06),  # 그림자/복잡
    ]  # 목록 끝
    rows, paths = {}, []  # 메타데이터와 경로 목록
    for name, cat, cbg, shadow, angle, persp in cases:  # 케이스마다
        H, W = 1600, 1200  # 사진 크기 (세로 사진)
        img = _background(H, W, rng, cbg)  # 배경
        scale = 0.55  # 문서가 사진에서 차지할 크기
        cx, cy = W / 2, H / 2  # 사진 중심
        base = (src - [dw / 2, dh / 2]) * scale * (H / dh) * 0.9  # 중심 기준으로 크기 조절된 꼭짓점
        t = np.deg2rad(angle)  # 회전 각도(라디안)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])  # 2x2 회전 행렬
        dst = base @ rot.T  # 행렬 곱으로 회전
        dst += rng.uniform(-persp, persp, dst.shape) * W  # 꼭짓점을 조금씩 흔들어 원근 효과
        dst = (dst + [cx, cy]).astype(np.float32)  # 사진 중심으로 이동
        M = cv2.getPerspectiveTransform(src, dst)  # 문서 → 사진 투시 행렬
        warped = cv2.warpPerspective(doc, M, (W, H))  # 문서를 사진 위치로 변형
        mask = cv2.warpPerspective(np.full((dh, dw), 255, np.uint8), M, (W, H))  # 문서 영역 마스크
        img[mask > 0] = warped[mask > 0]  # 배경 위에 문서 합성
        if shadow:  # 그림자 케이스면
            band = np.ones((H, W), np.float32)  # 밝기 배율 지도
            for k in range(0, W + H, 260):  # 대각선 줄무늬 그림자
                pts = np.array([[k, 0], [k + 110, 0], [k + 110 - H, H], [k - H, H]], np.int32)  # 줄무늬 다각형
                cv2.fillPoly(band, [pts], 0.35)  # 그 영역 밝기를 35%로
            img = np.clip(img * band[..., None], 0, 255).astype(np.uint8)  # 밝기 적용
        img = cv2.GaussianBlur(img, (3, 3), 0)  # 카메라처럼 살짝 흐리게
        path = save_image(folder / f"{name}.jpg", img)  # 저장
        paths.append(path)  # 경로 기록
        row = {c: "" for c in COLUMNS} | {"filename": path.name, "category": cat, "doc_type": "a4", "true_aspect": "1.414",  # 메타데이터 기본
               "lighting": "shadow" if shadow else "uniform", "angle": "tilt45+" if abs(angle) >= 45 else "front",  # 조명/각도
               "background": "complex" if cbg else "simple", "notes": "synthetic"}  # 배경/메모
        for col, val in zip(CORNER_COLS, order_points(dst).reshape(-1)):  # 정답 꼭짓점 8개
            row[col] = f"{val:.1f}"  # 기록
        rows[path.name] = row  # 사전에 추가
    write_metadata(rows, folder / "metadata.csv")  # CSV 저장
    return paths  # 경로 목록 반환
