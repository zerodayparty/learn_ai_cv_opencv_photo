"""스캐너 전체에서 공통으로 쓰는 파라미터(설정값) 모음."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트를 문자열로 늦게 평가 (호환성)

import json  # 설정 파일(JSON)을 읽기 위해 사용
from dataclasses import asdict, dataclass, field  # 설정을 깔끔한 클래스로 만들기 위해 사용
from pathlib import Path  # 파일 경로를 다루기 위해 사용

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 최상위 폴더 경로
IMAGES_DIR = PROJECT_ROOT / "images"  # 테스트 이미지 20장을 넣는 폴더
METADATA_CSV = IMAGES_DIR / "metadata.csv"  # 촬영 조건/정답 좌표를 기록하는 CSV 파일
OUTPUT_DIR = PROJECT_ROOT / "outputs"  # 스캔 결과 이미지가 저장되는 폴더 (git 제외)
REPORT_DIR = PROJECT_ROOT / "reports"  # 분석 보고서 폴더
GENERATED_DIR = REPORT_DIR / "auto"  # 프로그램이 자동 생성하는 표/그래프 폴더


@dataclass  # 아래 클래스를 '값 묶음' 클래스로 자동 생성
class ScanParams:  # 문서 스캔 파이프라인의 모든 조절 가능한 값
    blur_kernel: int = 5  # 가우시안 블러 커널 크기 (홀수, 클수록 더 흐려짐)
    canny_low: int = 50  # Canny 엣지의 낮은 임계값 (이보다 약한 경계는 버림)
    canny_high: int = 150  # Canny 엣지의 높은 임계값 (이보다 강하면 확실한 엣지)
    dilate_iter: int = 1  # 끊어진 엣지를 이어 붙이기 위한 팽창 반복 횟수
    min_area_ratio: float = 0.10  # 문서로 인정할 최소 면적 (이미지 전체 면적 대비 비율)
    approx_eps_ratios: list[float] = field(default_factory=lambda: [0.02, 0.03, 0.04, 0.05])  # approxPolyDP 오차 비율 후보들
    process_height: int = 800  # 계산 속도를 위해 줄여서 처리할 이미지 높이(px)
    thresh_block: int = 21  # 적응형 이진화의 주변 영역 크기 (홀수)
    thresh_c: int = 10  # 적응형 이진화에서 평균에서 빼는 상수 (클수록 더 하얗게)
    orientation: str = "portrait"  # 결과 방향: portrait(세로) / landscape(가로) / auto(검출 그대로)

    def normalized(self) -> "ScanParams":  # 잘못된 값을 올바른 값으로 고쳐서 돌려줌
        p = ScanParams(**asdict(self))  # 자기 자신을 복사 (원본은 건드리지 않음)
        p.blur_kernel = max(1, int(p.blur_kernel))  # 커널은 최소 1
        if p.blur_kernel % 2 == 0:  # 짝수 커널은 가우시안 블러에서 쓸 수 없으므로
            p.blur_kernel += 1  # 1을 더해 홀수로 만듦
        p.canny_low = max(0, int(p.canny_low))  # 낮은 임계값은 음수가 될 수 없음
        p.canny_high = max(p.canny_low + 1, int(p.canny_high))  # 높은 임계값은 낮은 값보다 커야 함
        p.thresh_block = max(3, int(p.thresh_block))  # 이진화 블록은 최소 3
        if p.thresh_block % 2 == 0:  # 이진화 블록도 홀수여야 하므로
            p.thresh_block += 1  # 짝수면 1을 더함
        return p  # 고쳐진 설정 반환

    def to_dict(self) -> dict:  # 설정을 사전(dict)으로 변환
        return asdict(self)  # dataclass 기본 기능으로 변환


def load_params(path: str | Path | None = None, **overrides) -> ScanParams:  # 설정 파일 + 명령줄 값으로 설정 생성
    params = ScanParams()  # 기본값으로 시작
    if path:  # 설정 파일 경로가 주어졌다면
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # JSON 파일을 읽어 사전으로 변환
        for key, value in data.items():  # 파일 안의 각 항목에 대해
            if hasattr(params, key):  # 알고 있는 설정 이름일 때만
                setattr(params, key, value)  # 해당 값을 덮어씀
    for key, value in overrides.items():  # 명령줄에서 넘어온 값들에 대해
        if value is not None and hasattr(params, key):  # 값이 있고 알고 있는 이름이면
            setattr(params, key, value)  # 파일 값보다 우선해서 덮어씀
    return params.normalized()  # 값 검증 후 반환
