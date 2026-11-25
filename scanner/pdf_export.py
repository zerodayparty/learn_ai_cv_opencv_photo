"""보너스 3: 처리된 이미지를 PDF로 저장 (Pillow 사용, 보너스 한정 허용)."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

from pathlib import Path  # 경로 처리

import cv2  # BGR → RGB 색상 변환
import numpy as np  # 배열 타입
from PIL import Image  # Pillow 이미지 객체 (PDF 저장 기능 제공)


def _to_pil(img: np.ndarray) -> Image.Image:  # OpenCV 배열을 Pillow 이미지로 변환
    if img.ndim == 2:  # 흑백(2차원) 이미지면
        return Image.fromarray(img).convert("L")  # 8비트 흑백 모드로 변환
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV는 BGR 순서라 RGB로 바꿔서 변환


def export_pdf(images: list[np.ndarray], path: str | Path, dpi: int = 200) -> Path:  # 여러 이미지를 한 PDF로 저장
    if not images:  # 이미지가 하나도 없으면
        raise ValueError("PDF로 만들 이미지가 없습니다.")  # 오류 발생
    path = Path(path)  # 경로 객체로 변환
    path.parent.mkdir(parents=True, exist_ok=True)  # 저장 폴더 생성
    pages = [_to_pil(img) for img in images]  # 모든 이미지를 Pillow 형식으로 변환
    pages[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])  # 첫 장 + 나머지를 한 PDF로 저장
    return path  # 저장 경로 반환
