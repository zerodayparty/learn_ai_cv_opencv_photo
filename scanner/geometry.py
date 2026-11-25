"""기하학 계산: 꼭짓점 정렬(직접 구현), 결과 크기(종횡비 자동 보정), 투시 변환."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

import cv2  # OpenCV: getPerspectiveTransform, warpPerspective 사용
import numpy as np  # NumPy: 좌표 계산


def order_points(pts: np.ndarray) -> np.ndarray:  # 4개 점을 좌상-우상-우하-좌하 순서로 정렬
    """라이브러리 없이 '무게중심 기준 각도'로 4점을 시계 방향 정렬한다.

    x+y 합/차 방식은 문서가 정확히 45도 돌아가면 두 점의 값이 같아져 실패할 수 있어서,
    무게중심에서 각 점을 바라보는 각도로 먼저 순서를 정하고, 시작점만 '좌상단'으로 맞춘다.
    """  # 함수 상세 설명
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)  # 어떤 모양이 와도 (4, 2) 배열로 맞춤
    if pts.shape[0] != 4:  # 점이 4개가 아니면
        raise ValueError(f"꼭짓점은 4개여야 합니다. 현재 {pts.shape[0]}개")  # 오류를 알림
    center = pts.mean(axis=0)  # 4점의 평균 = 무게중심 좌표
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])  # 중심에서 각 점을 향한 각도(라디안)
    clockwise = pts[np.argsort(angles)]  # 이미지 좌표(y가 아래로 증가)에서 각도 오름차순 = 화면상 시계 방향
    start = int(np.argmin(clockwise.sum(axis=1)))  # x+y가 가장 작은 점 = 좌상단에 가장 가까운 점
    ordered = np.roll(clockwise, -start, axis=0)  # 좌상단 점이 맨 앞에 오도록 회전
    return ordered.astype(np.float32)  # [좌상, 우상, 우하, 좌하] 반환


def distance(p1: np.ndarray, p2: np.ndarray) -> float:  # 두 점 사이의 거리
    return float(np.sqrt(((p1 - p2) ** 2).sum()))  # 피타고라스 정리 √(dx² + dy²)


def output_size(ordered: np.ndarray) -> tuple[int, int]:  # 결과 이미지 가로/세로 크기 계산 (보너스 2)
    """원본 좌표 간 거리로 결과의 가로·세로를 정해 찌그러짐(종횡비 왜곡)을 막는다."""  # 함수 설명
    tl, tr, br, bl = ordered  # 4점을 이름 붙여 꺼냄
    width = max(distance(tl, tr), distance(bl, br))  # 위쪽 변과 아래쪽 변 중 긴 쪽을 가로로
    height = max(distance(tl, bl), distance(tr, br))  # 왼쪽 변과 오른쪽 변 중 긴 쪽을 세로로
    return max(1, int(round(width))), max(1, int(round(height)))  # 최소 1px 정수로 반환


def apply_orientation(ordered: np.ndarray, orientation: str = "portrait") -> np.ndarray:  # 결과 방향(세로/가로) 맞추기
    """정렬된 4점의 시작점을 한 칸 돌려 결과를 세로(portrait) 또는 가로(landscape)로 만든다."""  # 함수 설명
    if orientation not in ("portrait", "landscape"):  # auto 등 다른 값이면
        return ordered  # 검출된 방향 그대로
    w, h = output_size(ordered)  # 현재 방향의 가로/세로
    if (orientation == "portrait" and w <= h) or (orientation == "landscape" and w >= h):  # 이미 원하는 방향이면
        return ordered  # 그대로 반환
    options = [np.roll(ordered, 1, axis=0), np.roll(ordered, -1, axis=0)]  # 시작점을 앞/뒤로 한 칸 돌린 두 후보 (가로·세로가 바뀜)
    return min(options, key=lambda o: float(o[0].sum()))  # 새 좌상단이 화면 왼쪽 위에 더 가까운 후보 선택


def warp_document(image: np.ndarray, corners: np.ndarray, orientation: str = "auto") -> tuple[np.ndarray, np.ndarray]:  # 투시 변환 실행
    ordered = apply_orientation(order_points(corners), orientation)  # 직접 구현한 정렬 + 방향 맞추기
    w, h = output_size(ordered)  # 결과 크기를 원본 비율에 맞게 계산
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)  # 펴진 직사각형의 목표 꼭짓점
    matrix = cv2.getPerspectiveTransform(ordered, dst)  # 4쌍의 점으로 3x3 투시 변환 행렬(호모그래피) 계산
    warped = cv2.warpPerspective(image, matrix, (w, h))  # 행렬을 모든 픽셀에 곱해 문서를 반듯하게 폄
    return warped, matrix  # 결과 이미지와 사용한 행렬 반환


def quad_aspect(corners: np.ndarray) -> float:  # 사각형의 종횡비 (긴 변 ÷ 짧은 변, 방향과 무관)
    w, h = output_size(order_points(corners))  # 결과 크기 계산
    return max(w, h) / min(w, h)  # 항상 1 이상 (A4 ≈ 1.414)


def corner_error(detected: np.ndarray, truth: np.ndarray) -> float:  # 검출 꼭짓점과 정답 꼭짓점의 최대 오차(px)
    d = order_points(detected)  # 검출 좌표 정렬
    t = order_points(truth)  # 정답 좌표 정렬
    return float(np.max(np.sqrt(((d - t) ** 2).sum(axis=1))))  # 4쌍 중 가장 크게 어긋난 거리
