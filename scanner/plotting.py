"""Matplotlib 공통 설정 (한글 폰트, 화면 없는 환경 지원)."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

import matplotlib  # 그래프 라이브러리

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장하는 백엔드 (서버/Colab에서도 동작)

import matplotlib.pyplot as plt  # noqa: E402  그래프 그리기 도구
from matplotlib import font_manager  # noqa: E402  설치된 폰트 목록 조회

KOREAN_FONTS = ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]  # 한글 폰트 후보


def setup_korean_font() -> str | None:  # 사용 가능한 한글 폰트를 찾아 설정
    installed = {f.name for f in font_manager.fontManager.ttflist}  # 설치된 폰트 이름 집합
    for name in KOREAN_FONTS:  # 후보를 순서대로 확인
        if name in installed:  # 설치되어 있으면
            plt.rcParams["font.family"] = name  # 기본 글꼴로 지정
            plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트에서 마이너스(-) 기호 깨짐 방지
            return name  # 사용한 폰트 이름 반환
    return None  # 한글 폰트가 없으면 None (영문만 정상 표시)


setup_korean_font()  # 모듈을 불러오는 순간 폰트 설정 실행
