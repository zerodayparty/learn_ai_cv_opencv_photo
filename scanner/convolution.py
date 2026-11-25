"""보너스 1: OpenCV 없이 NumPy만으로 가우시안 블러(컨볼루션) 구현 + 속도 비교."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

import time  # 실행 시간 측정
from pathlib import Path  # 경로 처리

import numpy as np  # NumPy만 사용해 컨볼루션 구현


OPENCV_SMALL_KERNELS = {  # OpenCV가 sigma<=0일 때 크기 1/3/5/7에서 쓰는 고정 1차원 커널 표
    1: [1.0],  # 크기 1: 블러 없음
    3: [0.25, 0.5, 0.25],  # 크기 3: [1,2,1]/4
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],  # 크기 5: [1,4,6,4,1]/16
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],  # 크기 7
}  # 표 끝


def gaussian_kernel(size: int, sigma: float | None = None) -> np.ndarray:  # 2차원 가우시안 커널 생성
    if size % 2 == 0:  # 커널 크기는 홀수여야 중심이 있음
        raise ValueError("커널 크기는 홀수여야 합니다.")  # 오류 발생
    if sigma is None and size in OPENCV_SMALL_KERNELS:  # 시그마 자동 + 작은 커널이면
        g = np.array(OPENCV_SMALL_KERNELS[size], dtype=np.float64)  # OpenCV와 똑같은 고정 표 사용 (결과 비교를 공정하게)
        return np.outer(g, g)  # 외적으로 2차원 커널 생성
    if sigma is None:  # 시그마를 안 주면
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8  # OpenCV가 쓰는 것과 같은 자동 시그마 공식
    ax = np.arange(size) - size // 2  # 중심이 0인 좌표: 예) size=5 → [-2,-1,0,1,2]
    g1 = np.exp(-(ax ** 2) / (2 * sigma ** 2))  # 1차원 가우시안 종 모양 값
    k = np.outer(g1, g1)  # 외적으로 2차원 커널 생성 (가우시안은 x, y로 분리 가능)
    return (k / k.sum()).astype(np.float64)  # 합이 1이 되게 정규화 (밝기 유지)


def _pad(img: np.ndarray, r: int) -> np.ndarray:  # 가장자리 패딩
    return np.pad(img.astype(np.float64), r, mode="reflect")  # 테두리를 거울처럼 반사해 채움 (OpenCV 기본 BORDER_REFLECT_101과 같음)


def conv_loop(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:  # 이중 for문 버전 (느림, 원리 이해용)
    r = kernel.shape[0] // 2  # 커널 반지름
    padded = _pad(img, r)  # 패딩된 이미지
    h, w = img.shape  # 출력 크기
    out = np.zeros((h, w), dtype=np.float64)  # 결과를 담을 빈 배열
    for y in range(h):  # 모든 행에 대해
        for x in range(w):  # 모든 열에 대해
            acc = 0.0  # 누적 합
            for ky in range(kernel.shape[0]):  # 커널 행
                for kx in range(kernel.shape[1]):  # 커널 열
                    acc += padded[y + ky, x + kx] * kernel[ky, kx]  # 픽셀 × 가중치 를 하나씩 더함 (= 내적)
            out[y, x] = acc  # 결과 기록
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)  # 반올림 후 0~255 정수로


def conv_vectorized(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:  # 벡터화 버전 (빠름)
    r = kernel.shape[0] // 2  # 커널 반지름
    padded = _pad(img, r)  # 패딩된 이미지
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)  # 모든 위치의 k×k 조각을 복사 없이 (H, W, k, k) 뷰로 만듦
    out = np.einsum("ijkl,kl->ij", windows, kernel)  # 각 조각과 커널의 내적을 한 번에 계산
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)  # 반올림 후 0~255 정수로


def conv_separable(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:  # 분리 가능 필터 버전 (가장 빠름)
    r = len(kernel_1d) // 2  # 반지름
    padded = _pad(img, r)  # 패딩
    k = len(kernel_1d)  # 커널 길이
    rows = sum(kernel_1d[i] * padded[:, i:i + padded.shape[1] - k + 1] for i in range(k))  # 가로 방향 1차원 컨볼루션
    out = sum(kernel_1d[i] * rows[i:i + rows.shape[0] - k + 1, :] for i in range(k))  # 세로 방향 1차원 컨볼루션
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)  # 정수 변환


def _timeit(fn, *args, repeat: int = 1) -> tuple[float, np.ndarray]:  # 함수 실행 시간 측정
    best, result = float("inf"), None  # 최단 시간과 결과
    for _ in range(repeat):  # 여러 번 실행해서
        t0 = time.perf_counter()  # 시작 시각
        result = fn(*args)  # 실행
        best = min(best, time.perf_counter() - t0)  # 가장 빠른 시간 기록
    return best, result  # 시간(초)과 결과 반환


def benchmark(gray: np.ndarray, ksize: int, out_md: Path, loop_size: int = 120) -> str:  # 속도/정확도 비교
    import cv2  # 정답 비교용으로만 OpenCV 사용 (구현 자체에는 사용 안 함)
    kernel = gaussian_kernel(ksize)  # 2차원 커널
    g1 = kernel.sum(axis=0)  # 2차원 커널의 열 합 = 1차원 커널
    crop = gray[:loop_size, :loop_size]  # 이중 for문은 너무 느려서 작은 조각으로 비교
    t_loop, r_loop = _timeit(conv_loop, crop, kernel)  # 이중 for문 시간
    t_vec_c, r_vec_c = _timeit(conv_vectorized, crop, kernel, repeat=3)  # 같은 조각에서 벡터화 시간
    t_vec, r_vec = _timeit(conv_vectorized, gray, kernel, repeat=3)  # 전체 이미지 벡터화 시간
    t_sep, r_sep = _timeit(conv_separable, gray, g1, repeat=3)  # 전체 이미지 분리 필터 시간
    t_cv, r_cv = _timeit(cv2.GaussianBlur, gray, (ksize, ksize), 0, repeat=3)  # OpenCV 시간
    diff = lambda a, b: int(np.abs(a.astype(int) - b.astype(int)).max())  # 두 결과의 최대 픽셀 차이
    lines = [  # 마크다운 결과
        "# 🟩 보너스 1: NumPy 컨볼루션 속도 비교 (자동 생성)", "",  # 제목
        f"- 커널: {ksize}×{ksize} 가우시안, 전체 이미지 {gray.shape[1]}×{gray.shape[0]}, 이중 for문 비교 조각 {crop.shape[1]}×{crop.shape[0]}", "",  # 조건
        "| 구현 | 입력 크기 | 시간(ms) | OpenCV 대비 최대 픽셀 차이 |", "| --- | --- | --- | --- |",  # 표 머리
        f"| 이중 for문 (순수 Python) | {crop.shape[1]}×{crop.shape[0]} | {t_loop * 1000:.1f} | {diff(r_loop, cv2.GaussianBlur(crop, (ksize, ksize), 0))} |",  # for문
        f"| 벡터화 (sliding_window + einsum) | {crop.shape[1]}×{crop.shape[0]} | {t_vec_c * 1000:.2f} | {diff(r_vec_c, cv2.GaussianBlur(crop, (ksize, ksize), 0))} |",  # 벡터화(조각)
        f"| 벡터화 (sliding_window + einsum) | {gray.shape[1]}×{gray.shape[0]} | {t_vec * 1000:.1f} | {diff(r_vec, r_cv)} |",  # 벡터화(전체)
        f"| 분리 필터 (1D × 2회) | {gray.shape[1]}×{gray.shape[0]} | {t_sep * 1000:.1f} | {diff(r_sep, r_cv)} |",  # 분리 필터
        f"| cv2.GaussianBlur (참고) | {gray.shape[1]}×{gray.shape[0]} | {t_cv * 1000:.2f} | 0 |", "",  # OpenCV
        f"- 같은 조각 기준 벡터화가 이중 for문보다 약 **{t_loop / max(t_vec_c, 1e-9):.0f}배** 빠름",  # 배속
        "- 이유: for문은 픽셀마다 Python 인터프리터가 곱셈/덧셈을 하나씩 실행, 벡터화는 C로 구현된 NumPy가 한 번에 처리",  # 해설 1
        "- 분리 필터: k×k 곱셈 대신 k + k 곱셈 (가우시안 커널 = 1D 커널의 외적이기 때문)",  # 해설 2
        "- 남는 1~2 픽셀 차이는 반올림 방식 차이 (OpenCV는 8비트 입력에 고정소수점 정수 연산 사용)", "",  # 해설 3
    ]  # 목록 끝
    out_md.parent.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    out_md.write_text("\n".join(lines), encoding="utf-8")  # 저장
    return "\n".join(lines)  # 문자열 반환
