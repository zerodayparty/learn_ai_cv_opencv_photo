"""문서 스캔 파이프라인: 원본 → 전처리 → 엣지 → 윤곽선 → 변환 → 이진화."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

from dataclasses import dataclass, field  # 결과를 담는 클래스 생성용
from pathlib import Path  # 파일 경로 처리

import cv2  # OpenCV 영상 처리 함수
import numpy as np  # NumPy 배열 계산

from .config import ScanParams  # 파라미터 설정 클래스
from .geometry import order_points, warp_document  # 꼭짓점 정렬과 투시 변환


@dataclass  # 결과 묶음 클래스 자동 생성
class ScanResult:  # 파이프라인 한 번 실행한 결과
    original: np.ndarray  # 원본 이미지 (전체 해상도)
    resized: np.ndarray  # 처리용으로 줄인 이미지
    scale: float  # 원본 ÷ 축소본 배율 (좌표 복원용)
    gray: np.ndarray  # 그레이스케일 이미지
    blurred: np.ndarray  # 가우시안 블러 적용 이미지
    edges: np.ndarray  # Canny 엣지 이미지
    contour_vis: np.ndarray  # 윤곽선/사각형을 그린 시각화 이미지
    corners: np.ndarray | None = None  # 원본 해상도 기준 4꼭짓점 (못 찾으면 None)
    warped: np.ndarray | None = None  # 투시 변환된 컬러 이미지
    scanned: np.ndarray | None = None  # 적응형 이진화까지 끝난 최종 스캔본
    info: dict = field(default_factory=dict)  # 디버깅용 부가 정보 (윤곽선 개수 등)

    @property  # 함수를 속성처럼 쓰게 함
    def found(self) -> bool:  # 문서를 찾았는지 여부
        return self.corners is not None  # 꼭짓점이 있으면 성공


def load_image(path: str | Path) -> np.ndarray:  # 이미지 파일 읽기 (한글 경로 지원)
    data = np.fromfile(str(path), dtype=np.uint8)  # 파일을 바이트 배열로 읽음 (cv2.imread는 한글 경로에 약함)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 바이트를 (H, W, 3) BGR 이미지로 해독
    if image is None:  # 해독에 실패하면
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")  # 오류 발생
    return image  # 이미지 배열 반환


def save_image(path: str | Path, image: np.ndarray) -> Path:  # 이미지 파일 저장 (한글 경로 지원)
    path = Path(path)  # 문자열이면 Path로 변환
    path.parent.mkdir(parents=True, exist_ok=True)  # 저장 폴더가 없으면 만듦
    ok, buf = cv2.imencode(path.suffix or ".png", image)  # 확장자에 맞게 이미지를 바이트로 인코딩
    if not ok:  # 인코딩 실패 시
        raise IOError(f"이미지를 저장할 수 없습니다: {path}")  # 오류 발생
    buf.tofile(str(path))  # 바이트를 파일로 기록
    return path  # 저장 경로 반환


def resize_for_processing(image: np.ndarray, target_h: int) -> tuple[np.ndarray, float]:  # 처리용 축소
    h = image.shape[0]  # 원본 높이
    if h <= target_h:  # 이미 충분히 작으면
        return image.copy(), 1.0  # 그대로 복사해서 사용
    scale = h / target_h  # 축소 배율 계산
    new_w = int(round(image.shape[1] / scale))  # 비율을 유지한 새 가로 길이
    small = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)  # 축소에 좋은 보간법으로 줄임
    return small, scale  # 축소본과 배율 반환


def preprocess(image: np.ndarray, blur_kernel: int) -> tuple[np.ndarray, np.ndarray]:  # 전처리 단계
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 컬러(3채널)를 밝기(1채널)로 변환
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1  # 커널은 반드시 홀수
    blurred = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray.copy()  # 가우시안 필터와 컨볼루션해 노이즈 제거
    return gray, blurred  # 그레이와 블러 결과 반환


def detect_edges(blurred: np.ndarray, low: int, high: int, dilate_iter: int) -> np.ndarray:  # 엣지 검출 단계
    edges = cv2.Canny(blurred, low, high)  # 밝기 변화(기울기)가 큰 곳을 흰 선으로 표시
    if dilate_iter > 0:  # 팽창을 쓰도록 설정되어 있으면
        kernel = np.ones((3, 3), np.uint8)  # 3x3 크기의 사각형 구조 요소
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)  # 선을 두껍게 해 끊긴 틈을 메움
    return edges  # 엣지 이미지 반환


def find_document_quad(edges: np.ndarray, params: ScanParams) -> tuple[np.ndarray | None, list, dict]:  # 사각형 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 바깥쪽 닫힌 윤곽선만 찾음
    img_area = edges.shape[0] * edges.shape[1]  # 이미지 전체 면적
    min_area = params.min_area_ratio * img_area  # 이 면적보다 작으면 노이즈로 판단
    big = [c for c in contours if cv2.contourArea(c) >= min_area]  # 면적 필터링으로 작은 노이즈 제거
    big.sort(key=cv2.contourArea, reverse=True)  # 큰 윤곽선부터 검사 (문서는 보통 가장 큼)
    info = {"contours_total": len(contours), "contours_big": len(big)}  # 디버깅 정보 기록
    for c in big:  # 큰 윤곽선마다
        peri = cv2.arcLength(c, True)  # 윤곽선 둘레 길이
        for eps in params.approx_eps_ratios:  # 오차 허용치를 조금씩 늘려 가며
            approx = cv2.approxPolyDP(c, eps * peri, True)  # 윤곽선을 꼭짓점이 적은 다각형으로 단순화
            if len(approx) == 4 and cv2.isContourConvex(approx):  # 꼭짓점 4개 + 볼록하면 문서 후보
                info["approx_eps"] = eps  # 어떤 오차에서 찾았는지 기록
                info["quad_area_ratio"] = round(cv2.contourArea(approx) / img_area, 4)  # 문서가 차지하는 면적 비율
                return approx.reshape(4, 2).astype(np.float32), big, info  # 찾은 4점 반환
    return None, big, info  # 못 찾으면 None 반환


def postprocess(warped: np.ndarray, block: int, c: int) -> np.ndarray:  # 후처리 (적응형 이진화)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # 펴진 문서를 그레이스케일로
    block = block if block % 2 == 1 else block + 1  # 블록 크기는 홀수
    return cv2.adaptiveThreshold(  # 주변 밝기 평균을 기준으로 픽셀마다 다른 임계값을 써서 흑/백으로 나눔
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c  # 가우시안 가중 평균 - C 를 임계값으로 사용
    )  # 이진화 결과 반환


def run_pipeline(image: np.ndarray, params: ScanParams) -> ScanResult:  # 전체 파이프라인 실행
    p = params.normalized()  # 파라미터 값 검증
    small, scale = resize_for_processing(image, p.process_height)  # 속도를 위해 축소
    gray, blurred = preprocess(small, p.blur_kernel)  # 1) 전처리
    edges = detect_edges(blurred, p.canny_low, p.canny_high, p.dilate_iter)  # 2) 엣지 검출
    quad, candidates, info = find_document_quad(edges, p)  # 3) 윤곽선에서 사각형 찾기
    vis = small.copy()  # 시각화용 복사본
    cv2.drawContours(vis, candidates, -1, (0, 255, 0), 1)  # 면적 필터를 통과한 후보 윤곽선은 초록색
    result = ScanResult(original=image, resized=small, scale=scale, gray=gray,  # 결과 객체 생성 (앞부분)
                        blurred=blurred, edges=edges, contour_vis=vis, info=info)  # 결과 객체 생성 (뒷부분)
    if quad is None:  # 문서를 못 찾았으면
        cv2.putText(vis, "NOT FOUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 실패 표시
        return result  # 여기서 종료
    ordered_small = order_points(quad)  # 축소본 기준 꼭짓점 정렬
    cv2.polylines(vis, [ordered_small.astype(np.int32)], True, (0, 0, 255), 3)  # 검출 사각형을 빨간색으로 그림
    for name, (x, y) in zip(["TL", "TR", "BR", "BL"], ordered_small.astype(int)):  # 각 꼭짓점에
        cv2.circle(vis, (int(x), int(y)), 7, (255, 0, 0), -1)  # 파란 점 표시
        cv2.putText(vis, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # 이름 표시
    result.corners = ordered_small * scale  # 원본 해상도 좌표로 복원
    result.warped, _ = warp_document(image, result.corners, p.orientation)  # 4) 원본 해상도에서 투시 변환 (화질 유지)
    result.scanned = postprocess(result.warped, p.thresh_block, p.thresh_c)  # 5) 적응형 이진화
    return result  # 최종 결과 반환
