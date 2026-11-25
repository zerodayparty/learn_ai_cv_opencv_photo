"""과제용 테스트 사진 20장을 '휴대폰으로 찍은 것처럼' 합성하는 모듈.

실제 촬영을 대신하는 '표준 테스트셋' 역할을 한다.
- 문서: A4 보고서 / A4 신청서 / 영수증 / 명함 (한글 글씨 포함)
- 배경: 나무 책상, 단색 책상, 키보드·책, 체크무늬 천, 다른 종이 더미, 잡동사니
- 환경: 3D 카메라 원근, 손 그림자, 창틀 그림자, 어두운 조명, 빛 반사, 구겨짐, 손 가림, 카메라 노이즈
- 정답 꼭짓점을 계산으로 알고 있으므로 metadata.csv 에 자동 기록 (annotate 불필요)
"""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

from functools import lru_cache  # 폰트를 한 번만 불러오기 위한 캐시
from pathlib import Path  # 경로 처리

import cv2  # 그리기 / 변환
import numpy as np  # 배열 계산
from matplotlib import font_manager  # 설치된 한글 폰트 파일 위치 찾기
from PIL import Image, ImageDraw, ImageFont  # 한글 글씨 그리기 (OpenCV putText 는 한글 불가)

from .dataset import COLUMNS, CORNER_COLS, list_images, write_metadata  # 메타데이터 저장
from .geometry import order_points  # 정답 꼭짓점 정렬

PHOTO_W, PHOTO_H = 1500, 2000  # 결과 사진 크기 (휴대폰 세로 사진 3:4 비율)
KOREAN_FONTS = ["Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]  # 한글 폰트 후보 (운영체제별)
DOC_ASPECT = {"a4": "1.414", "receipt": "2.5", "card": "1.8"}  # 문서 종류별 실제 종횡비 (긴 변 ÷ 짧은 변)


# ───────────────────────── 공통 도구 ─────────────────────────

@lru_cache(maxsize=1)  # 한 번 찾은 결과를 기억
def _font_file() -> str | None:  # 한글 폰트 파일 경로 찾기
    for name in KOREAN_FONTS:  # 후보를 순서대로
        try:  # 없으면 예외가 나므로 보호
            return font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)  # 실제 파일 경로
        except ValueError:  # 이 폰트가 없으면
            continue  # 다음 후보
    return None  # 하나도 없으면 None


@lru_cache(maxsize=64)  # 크기별 폰트 객체 기억
def _font(size: int) -> ImageFont.FreeTypeFont:  # 지정 크기의 폰트
    path = _font_file()  # 폰트 파일
    if path:  # 한글 폰트가 있으면
        return ImageFont.truetype(path, size)  # 그 폰트 사용
    return ImageFont.load_default(size=size)  # 없으면 기본 폰트 (한글은 네모로 보일 수 있음)


def _smooth_noise(h: int, w: int, cell: int, rng: np.random.Generator) -> np.ndarray:  # 부드러운 얼룩 노이즈 (-1~1 근처)
    small = rng.standard_normal((max(2, h // cell), max(2, w // cell))).astype(np.float32)  # 작은 무작위 격자
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)  # 크게 늘려서 부드럽게


def _to_bgr(pil: Image.Image) -> np.ndarray:  # PIL(RGB) → OpenCV(BGR) 배열
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)  # 채널 순서 뒤집기


def _paste(img: np.ndarray, patch: np.ndarray, alpha: np.ndarray, center: tuple[float, float], angle: float,
           shadow: float = 0.35, shadow_off: tuple[int, int] = (12, 18), shadow_blur: int = 31) -> None:  # 물체를 회전시켜 붙이기 (그림자 포함)
    ph, pw = patch.shape[:2]  # 물체 크기
    H, W = img.shape[:2]  # 사진 크기
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)  # 물체 중심 기준 회전 행렬 (2x3)
    M[:, 2] += np.array(center) - np.array([pw / 2, ph / 2])  # 원하는 위치로 이동
    a = cv2.warpAffine(alpha, M, (W, H))  # 투명도 지도 옮기기
    p = cv2.warpAffine(patch.astype(np.float32), M, (W, H))  # 물체 색 옮기기
    if shadow > 0:  # 그림자를 그릴 때
        Ms = M.copy()  # 그림자용 행렬
        Ms[:, 2] += shadow_off  # 오른쪽 아래로 살짝 밀기
        s = cv2.GaussianBlur(cv2.warpAffine(alpha, Ms, (W, H)), (0, 0), shadow_blur / 3)  # 흐린 그림자 모양
        img *= (1 - shadow * s)[..., None]  # 그림자만큼 어둡게
    img[:] = img * (1 - a[..., None]) + p * a[..., None]  # 투명도에 맞춰 섞기


def _rect_alpha(h: int, w: int) -> np.ndarray:  # 꽉 찬 직사각형 투명도 지도
    return np.ones((h, w), np.float32)  # 전부 1 (불투명)


# ───────────────────────── 문서 만들기 ─────────────────────────

def _doc_a4_report(rng: np.random.Generator) -> np.ndarray:  # A4 설비 점검 보고서 (1240x1754 = 150dpi A4)
    W, H = 1240, 1754  # A4 픽셀 크기
    im = Image.new("RGB", (W, H), (248, 248, 245))  # 살짝 따뜻한 흰 종이
    d = ImageDraw.Draw(im)  # 그리기 도구
    ink = (30, 30, 35)  # 글씨 색 (거의 검정)
    d.text((110, 110), "설비 점검 보고서", font=_font(70), fill=ink)  # 제목
    d.line([(110, 210), (W - 110, 210)], fill=ink, width=4)  # 제목 밑줄
    for i, s in enumerate(["점검 일자: 2026-11-20", "점검 라인: LINE-03 프레스", "점검자: 품질팀 김OO"]):  # 기본 정보 3줄
        d.text((110, 245 + i * 50), s, font=_font(32), fill=ink)  # 정보 글씨
    top, rows, cols = 430, 9, [110, 420, 700, 930, W - 110]  # 표 위치 / 행 수 / 열 경계
    for r in range(rows + 1):  # 가로선
        d.line([(cols[0], top + r * 62), (cols[-1], top + r * 62)], fill=ink, width=2)  # 표 가로선
    for x in cols:  # 세로선
        d.line([(x, top), (x, top + rows * 62)], fill=ink, width=2)  # 표 세로선
    head = ["점검 항목", "기준", "측정값", "판정"]  # 표 머리글
    items = ["유압 압력", "모터 온도", "벨트 장력", "소음", "진동", "윤활유", "안전 커버", "비상 정지"]  # 점검 항목
    for c, s in enumerate(head):  # 머리글 쓰기
        d.text((cols[c] + 20, top + 14), s, font=_font(30), fill=ink)  # 머리글 글씨
    for r, s in enumerate(items, start=1):  # 각 행 쓰기
        val = f"{rng.uniform(10, 99):.1f}"  # 무작위 측정값
        ok = "적합" if rng.random() > 0.2 else "부적합"  # 대부분 적합
        for c, t in enumerate([s, "규정 이내", val, ok]):  # 칸마다
            d.text((cols[c] + 20, top + r * 62 + 14), t, font=_font(28), fill=ink)  # 칸 글씨
    body = ["1. 프레스 유압 압력은 기준 범위 안에서 안정적으로 유지되었다.", "2. 모터 표면 온도가 오후 시간대에 소폭 상승하였다.",  # 본문 문장
            "3. 벨트 장력은 다음 정기 점검 때 재조정이 필요하다.", "4. 비상 정지 버튼은 3회 시험 모두 정상 동작하였다.",  # 본문 문장
            "5. 부적합 항목은 담당자에게 전달하고 조치 결과를 기록한다."]  # 본문 문장
    for i, s in enumerate(body):  # 본문 쓰기
        d.text((110, 1050 + i * 58), s, font=_font(30), fill=ink)  # 본문 글씨
    d.rectangle([(W - 420, H - 300), (W - 110, H - 140)], outline=ink, width=3)  # 서명 칸
    d.text((W - 400, H - 285), "확인 서명", font=_font(28), fill=ink)  # 서명 칸 글씨
    return _to_bgr(im)  # BGR 배열로 반환


def _doc_a4_form(rng: np.random.Generator) -> np.ndarray:  # A4 반품 신청서 (빈칸 양식)
    W, H = 1240, 1754  # A4 픽셀 크기
    im = Image.new("RGB", (W, H), (247, 247, 247))  # 흰 종이
    d = ImageDraw.Draw(im)  # 그리기 도구
    ink, blue = (25, 25, 30), (40, 70, 150)  # 검정 글씨 / 파란 볼펜
    d.rectangle([(0, 0), (W, 150)], fill=(225, 232, 245))  # 위쪽 연한 파란 머리띠 (종이 가장자리 대비를 낮춤)
    d.text((110, 45), "반품 신청서", font=_font(66), fill=ink)  # 제목
    labels = ["주문 번호", "주문자 이름", "연락처", "상품명", "반품 사유", "환불 계좌"]  # 양식 항목
    for i, s in enumerate(labels):  # 항목마다
        y = 250 + i * 120  # 행 위치
        d.text((110, y), s, font=_font(34), fill=ink)  # 항목 이름
        d.line([(400, y + 48), (W - 110, y + 48)], fill=ink, width=2)  # 적는 줄
    hand = ["2611-20-4471", "이OO", "010-0000-0000", "무선 키보드", "단순 변심", "(개인정보 생략)"]  # 손글씨처럼 적힌 내용
    for i, s in enumerate(hand):  # 적힌 내용
        d.text((420 + int(rng.integers(0, 20)), 250 + i * 120 - 2), s, font=_font(36), fill=blue)  # 파란 글씨
    d.text((110, 1020), "■ 유의 사항", font=_font(34), fill=ink)  # 유의 사항 제목
    for i in range(7):  # 유의 사항 문장
        d.text((130, 1080 + i * 52), f"{i + 1}) 상품 수령 후 7일 이내에 신청해야 하며 포장이 훼손되면 반품이 제한될 수 있습니다.", font=_font(26), fill=ink)  # 작은 글씨
    for i in range(4):  # 체크박스
        d.rectangle([(130 + i * 260, 1500), (160 + i * 260, 1530)], outline=ink, width=3)  # 네모 칸
        d.text((175 + i * 260, 1496), ["교환", "환불", "수리", "기타"][i], font=_font(30), fill=ink)  # 체크박스 이름
    d.line([(135, 1515), (145, 1527), (165, 1495)], fill=blue, width=5)  # 첫 칸에 체크 표시
    return _to_bgr(im)  # BGR 배열로 반환


def _doc_receipt(rng: np.random.Generator) -> np.ndarray:  # 감열지 영수증 (576x1440 = 종횡비 2.5)
    W, H = 576, 1440  # 영수증 크기
    im = Image.new("RGB", (W, H), (244, 243, 236))  # 살짝 누런 감열지
    d = ImageDraw.Draw(im)  # 그리기 도구
    ink = (55, 55, 60)  # 감열지 글씨는 진한 회색
    d.text((150, 50), "OO 마트 본점", font=_font(44), fill=ink)  # 가게 이름
    d.text((60, 120), "사업자 000-00-00000 | 서울시 OO구", font=_font(22), fill=ink)  # 가게 정보
    d.text((60, 155), "2026-11-21 18:42  POS 03", font=_font(22), fill=ink)  # 날짜
    d.text((40, 195), "-" * 48, font=_font(22), fill=ink)  # 점선
    names = ["우유 1L", "식빵", "계란 10구", "사과 1봉", "생수 2L x6", "라면 5입", "커피 원두", "휴지 30롤", "세제", "바나나"]  # 상품 이름
    total = 0  # 합계
    for i, s in enumerate(names):  # 상품마다
        price = int(rng.integers(12, 160)) * 100  # 무작위 가격
        total += price  # 합계 누적
        d.text((50, 240 + i * 52), s, font=_font(28), fill=ink)  # 상품 이름
        d.text((W - 190, 240 + i * 52), f"{price:>7,}", font=_font(28), fill=ink)  # 가격
    y = 240 + len(names) * 52  # 목록 끝 위치
    d.text((40, y), "-" * 48, font=_font(22), fill=ink)  # 점선
    d.text((50, y + 40), "합계", font=_font(40), fill=ink)  # 합계 글씨
    d.text((W - 250, y + 40), f"{total:,}원", font=_font(40), fill=ink)  # 합계 금액
    d.text((50, y + 110), "카드 승인  ****-****-****-0000", font=_font(24), fill=ink)  # 카드 정보 (가짜)
    for x in range(70, W - 70, 6):  # 바코드 막대
        if rng.random() > 0.45:  # 무작위로 막대 그리기
            d.rectangle([(x, H - 230), (x + int(rng.integers(1, 4)), H - 120)], fill=ink)  # 막대 하나
    d.text((170, H - 100), "교환/환불 7일 이내", font=_font(24), fill=ink)  # 안내 문구
    return _to_bgr(im)  # BGR 배열로 반환


def _doc_card(rng: np.random.Generator) -> np.ndarray:  # 명함 (1050x583 = 90x50mm 비율 1.8)
    W, H = 1050, 583  # 명함 크기
    im = Image.new("RGB", (W, H), (250, 250, 250))  # 흰 명함
    d = ImageDraw.Draw(im)  # 그리기 도구
    d.rectangle([(60, 70), (82, H - 70)], fill=(30, 90, 170))  # 왼쪽 파란 세로 띠 (종이 가장자리에 닿지 않음)
    d.ellipse([(W - 230, 60), (W - 90, 200)], fill=(30, 90, 170))  # 로고 동그라미
    d.text((W - 200, 100), "OO", font=_font(50), fill=(255, 255, 255))  # 로고 글씨
    d.text((120, 120), "홍 길 동", font=_font(72), fill=(20, 20, 25))  # 이름
    d.text((120, 220), "스마트팩토리팀 | 선임 연구원", font=_font(32), fill=(70, 70, 80))  # 소속
    for i, s in enumerate(["T. 02-000-0000", "M. 010-0000-0000", "E. hong@example.com"]):  # 연락처 (가짜)
        d.text((120, 330 + i * 52), s, font=_font(30), fill=(40, 40, 50))  # 연락처 글씨
    return _to_bgr(im)  # BGR 배열로 반환


DOCS = {"a4_report": (_doc_a4_report, "a4"), "a4_form": (_doc_a4_form, "a4"), "receipt": (_doc_receipt, "receipt"), "card": (_doc_card, "card")}  # 문서 이름 → (생성 함수, 종류)


def _crumple(doc: np.ndarray, rng: np.random.Generator, amp: float = 14.0) -> tuple[np.ndarray, np.ndarray]:  # 종이 구기기 (꼭짓점 위치는 유지)
    h, w = doc.shape[:2]  # 문서 크기
    height = _smooth_noise(h, w, 160, rng) * 1.0 + _smooth_noise(h, w, 60, rng) * 0.35  # 울퉁불퉁한 높이 지도
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)  # 픽셀 좌표 격자
    for _ in range(4):  # 접힌 자국(직선 주름) 4개
        a, c = rng.uniform(0, np.pi), rng.uniform(0.3, 0.7)  # 주름 방향 / 위치
        dist = (xx - w * c) * np.cos(a) + (yy - h * c) * np.sin(a)  # 주름 선까지의 거리 (부호 있음)
        height += 0.9 * np.exp(-np.abs(dist) / 25)  # 주름 선에서 뾰족하게 솟음
    gy, gx = np.gradient(height)  # 높이의 기울기 (어느 쪽으로 기울었나)
    gx, gy = gx / (np.abs(gx).max() + 1e-6), gy / (np.abs(gy).max() + 1e-6)  # -1~1 로 맞춤
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)  # 네 꼭짓점
    dmin = np.min([np.hypot(xx - cx, yy - cy) for cx, cy in corners], axis=0)  # 가장 가까운 꼭짓점까지 거리
    weight = np.clip(dmin / (0.12 * min(w, h)), 0, 1)  # 꼭짓점 근처는 변형 0 (정답 꼭짓점 보존)
    map_x = xx + amp * gx * weight  # 가로로 밀기
    map_y = yy + amp * gy * weight  # 세로로 밀기
    bent = cv2.remap(doc, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # 글씨가 휘어 보이게
    alpha = cv2.remap(_rect_alpha(h, w), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)  # 가장자리가 물결치게
    shade = 1.0 + 0.22 * (gx * -0.7 + gy * -0.7)  # 왼쪽 위에서 빛이 올 때 기울기별 밝기
    bent = np.clip(bent * shade[..., None], 0, 255).astype(np.uint8)  # 주름 명암 적용
    return bent, alpha  # 구겨진 문서와 투명도


# ───────────────────────── 배경 만들기 ─────────────────────────

def _bg_wood(h: int, w: int, rng: np.random.Generator, base=(62, 108, 160)) -> np.ndarray:  # 나무 책상
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)  # 좌표 격자
    warp = _smooth_noise(h, w, 220, rng) * 45 + _smooth_noise(h, w, 50, rng) * 6  # 나뭇결 휘어짐
    grain = np.sin((yy + warp) / 8.0) * 0.5 + np.sin((yy * 0.37 + warp * 1.6) / 2.5) * 0.25  # 나뭇결 무늬
    plank = ((yy // 520) % 2) * 0.07  # 판자마다 색 조금 다르게
    gain = 1 + 0.09 * grain - plank + 0.05 * _smooth_noise(h, w, 300, rng)  # 밝기 배율
    img = np.array(base, np.float32)[None, None, :] * gain[..., None]  # 기본 색 × 배율
    img[(yy % 520) < 3] *= 0.5  # 판자 사이 틈
    return img  # float 배경 반환


def _bg_solid(h: int, w: int, rng: np.random.Generator, color) -> np.ndarray:  # 단색 책상/매트
    gain = 1 + 0.04 * _smooth_noise(h, w, 260, rng)  # 아주 약한 얼룩
    img = np.array(color, np.float32)[None, None, :] * gain[..., None]  # 색 × 얼룩
    return img + rng.normal(0, 2.5, (h, w, 1)).astype(np.float32)  # 표면 질감 노이즈


def _bg_plaid(h: int, w: int, rng: np.random.Generator) -> np.ndarray:  # 체크무늬 천
    yy, xx = np.mgrid[0:h, 0:w]  # 좌표 격자
    img = np.empty((h, w, 3), np.float32)  # 빈 배경
    img[:] = (55, 45, 150)  # 빨간 천 (BGR)
    band_x, band_y = (xx // 70) % 3 == 0, (yy // 70) % 3 == 0  # 굵은 줄무늬 위치
    img[band_x] = img[band_x] * 0.55 + np.array((40, 30, 30)) * 0.45  # 세로 줄 어둡게
    img[band_y] = img[band_y] * 0.55 + np.array((40, 30, 30)) * 0.45  # 가로 줄 어둡게
    img[(xx % 210) < 6] = (180, 220, 235)  # 얇은 밝은 세로선
    img[(yy % 210) < 6] = (180, 220, 235)  # 얇은 밝은 가로선
    weave = ((xx + yy) % 4 < 2) * 10.0  # 천 짜임 무늬
    return img + weave[..., None] + rng.normal(0, 4, (h, w, 1))  # 짜임 + 노이즈


def _patch_keyboard() -> tuple[np.ndarray, np.ndarray]:  # 키보드 물체
    W, H = 1350, 470  # 키보드 크기
    p = np.full((H, W, 3), 38, np.uint8)  # 검은 몸체 (putText 는 uint8 만 지원)
    keys = "QWERTYUIOP[]ASDFGHJKL;'ZXCVBNM,./"  # 키 글자
    k = 0  # 글자 순서
    for r in range(5):  # 5줄
        for c in range(14):  # 14칸
            x, y = 25 + c * 93 + (r % 3) * 20, 25 + r * 88  # 키 위치 (줄마다 살짝 어긋남)
            if x + 80 > W - 10:  # 오른쪽 끝을 넘으면
                continue  # 그리지 않음
            cv2.rectangle(p, (x, y), (x + 80, y + 76), (72, 72, 76), -1)  # 키 모양
            cv2.rectangle(p, (x, y), (x + 80, y + 76), (95, 95, 100), 2)  # 키 테두리 (밝은 경계 = 강한 엣지)
            cv2.putText(p, keys[k % len(keys)], (x + 26, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)  # 키 글자
            k += 1  # 다음 글자
    return p, _rect_alpha(H, W)  # 색과 투명도


def _patch_book(rng: np.random.Generator, color) -> tuple[np.ndarray, np.ndarray]:  # 책 표지
    W, H = int(rng.integers(650, 800)), int(rng.integers(900, 1050))  # 책 크기
    p = np.empty((H, W, 3), np.uint8)  # 빈 표지 (putText 는 uint8 만 지원)
    p[:] = color  # 표지 색
    cv2.rectangle(p, (0, 0), (40, H), tuple(int(v * 0.6) for v in color), -1)  # 책등 쪽 어두운 띠
    cv2.rectangle(p, (90, 120), (W - 90, 330), (235, 235, 235), -1)  # 제목 상자
    cv2.putText(p, "COMPUTER VISION", (110, 250), cv2.FONT_HERSHEY_DUPLEX, 1.6, (30, 30, 30), 3)  # 제목
    cv2.circle(p, (W // 2, H * 2 // 3), 150, (230, 200, 60), -1)  # 표지 그림
    return p, _rect_alpha(H, W)  # 색과 투명도


def _patch_paper(rng: np.random.Generator, w: int = 820, h: int = 1160, color=(240, 240, 240)) -> tuple[np.ndarray, np.ndarray]:  # 다른 종이
    p = np.empty((h, w, 3), np.float32)  # 빈 종이
    p[:] = color  # 종이 색
    for i in range(18):  # 글줄 18개
        y = 120 + i * 52  # 줄 위치
        cv2.line(p, (80, y), (int(80 + rng.uniform(0.5, 1.0) * (w - 160)), y), (90, 90, 90), 6)  # 글씨처럼 보이는 회색 막대
    return p, _rect_alpha(h, w)  # 색과 투명도


def _patch_sticky(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:  # 포스트잇
    p = np.empty((320, 320, 3), np.float32)  # 빈 메모지
    p[:] = (110, 235, 250)  # 노란색 (BGR)
    for i in range(4):  # 손글씨 4줄
        pts = np.array([[40 + j * 24, 80 + i * 55 + int(rng.integers(-6, 6))] for j in range(10)], np.int32)  # 꼬불꼬불 점들
        cv2.polylines(p, [pts], False, (120, 60, 30), 4)  # 손글씨 선
    return p, _rect_alpha(320, 320)  # 색과 투명도


def _patch_mug() -> tuple[np.ndarray, np.ndarray]:  # 위에서 본 머그컵
    S = 420  # 크기
    p = np.zeros((S, S, 3), np.float32)  # 빈 그림
    a = np.zeros((S, S), np.float32)  # 투명도
    cv2.circle(a, (180, 210), 170, 1.0, -1)  # 컵 몸통
    cv2.ellipse(a, (360, 210), (55, 75), 0, 0, 360, 1.0, 22)  # 손잡이
    p[:] = (235, 235, 240)  # 흰 컵
    cv2.circle(p, (180, 210), 135, (25, 45, 70), -1)  # 커피 (갈색)
    cv2.circle(p, (150, 180), 30, (60, 90, 120), -1)  # 커피 표면 반사
    return p, cv2.GaussianBlur(a, (5, 5), 0)  # 색과 부드러운 투명도


def _patch_phone() -> tuple[np.ndarray, np.ndarray]:  # 휴대폰
    W, H = 380, 780  # 크기
    p = np.full((H, W, 3), 20, np.float32)  # 검은 몸체
    cv2.rectangle(p, (18, 40), (W - 18, H - 40), (90, 50, 30), -1)  # 화면 (어두운 파랑)
    a = np.zeros((H, W), np.float32)  # 투명도
    cv2.rectangle(a, (20, 0), (W - 20, H), 1.0, -1)  # 둥근 직사각형 근사 (세로)
    cv2.rectangle(a, (0, 20), (W, H - 20), 1.0, -1)  # 둥근 직사각형 근사 (가로)
    return p, a  # 색과 투명도


def _patch_pen() -> tuple[np.ndarray, np.ndarray]:  # 볼펜
    p = np.zeros((40, 700, 3), np.float32)  # 길쭉한 그림
    p[:] = (160, 60, 20)  # 파란 볼펜 (BGR)
    p[:, 600:] = (40, 40, 40)  # 펜 끝 검정
    p[8:14, :] += 60  # 원통 반사광
    return p, _rect_alpha(40, 700)  # 색과 투명도


def _patch_hand(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:  # 손 (손가락이 위쪽을 향함)
    S = int(700 * scale)  # 그림 크기
    m = np.zeros((S, S), np.uint8)  # 손 모양 마스크
    cx, cy = S // 2, int(S * 0.62)  # 손바닥 중심
    cv2.ellipse(m, (cx, cy), (int(S * 0.19), int(S * 0.22)), 0, 0, 360, 255, -1)  # 손바닥
    for i, ang in enumerate([-22, -7, 7, 21]):  # 손가락 4개 각도
        t = np.deg2rad(ang)  # 라디안 변환
        base = (cx + int((i - 1.5) * S * 0.085), cy - int(S * 0.14))  # 손가락 시작점
        length = S * [0.30, 0.34, 0.33, 0.27][i]  # 손가락 길이
        tip = (int(base[0] + np.sin(t) * length), int(base[1] - np.cos(t) * length))  # 손가락 끝
        cv2.line(m, base, tip, 255, int(S * 0.075))  # 굵은 선 = 손가락
    cv2.line(m, (cx - int(S * 0.16), cy), (cx - int(S * 0.36), cy - int(S * 0.16)), 255, int(S * 0.085))  # 엄지
    cv2.rectangle(m, (cx - int(S * 0.15), cy), (cx + int(S * 0.15), S), 255, -1)  # 손목
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)  # 가장자리까지 거리 (입체감용)
    shade = 0.72 + 0.33 * np.clip(dt / (S * 0.05), 0, 1)  # 가장자리는 어둡게
    p = np.array((120, 158, 212), np.float32)[None, None, :] * shade[..., None]  # 피부색 × 명암
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (5, 5), 0)  # 부드러운 투명도
    return p, a  # 색과 투명도


def _make_background(kind: str, rng: np.random.Generator, H: int = PHOTO_H, W: int = PHOTO_W) -> np.ndarray:  # 배경 종류별 생성
    if kind == "wood":  # 밝은 나무
        return _bg_wood(H, W, rng)  # 나무 책상
    if kind == "wood_dark":  # 어두운 나무
        return _bg_wood(H, W, rng, base=(40, 62, 95))  # 진한 나무 책상
    if kind == "gray":  # 회색 책상
        return _bg_solid(H, W, rng, (118, 120, 122))  # 회색
    if kind == "dark":  # 검은 책상
        return _bg_solid(H, W, rng, (42, 40, 40))  # 거의 검정
    if kind == "mat":  # 초록 커팅 매트
        img = _bg_solid(H, W, rng, (70, 110, 40))  # 초록색
        for x in range(0, W, 100):  # 격자 세로선
            cv2.line(img, (x, 0), (x, H), (110, 150, 90), 2)  # 연한 선 (약한 엣지)
        for y in range(0, H, 100):  # 격자 가로선
            cv2.line(img, (0, y), (W, y), (110, 150, 90), 2)  # 연한 선
        return img  # 매트 반환
    if kind == "white_desk":  # 흰 책상 (문서와 대비가 낮음)
        return _bg_solid(H, W, rng, (212, 214, 214))  # 밝은 회백색
    if kind == "plaid":  # 체크무늬 천
        return _bg_plaid(H, W, rng)  # 천
    img = _bg_wood(H, W, rng)  # 나머지는 나무 책상 위에 물건 배치
    if kind == "keyboard_books":  # 키보드 + 책
        _paste(img, *_patch_keyboard(), center=(W * 0.5, H * 0.04), angle=-4)  # 위쪽 키보드 (문서와 떨어짐)
        _paste(img, *_patch_book(rng, (150, 60, 40)), center=(W * -0.08, H * 0.80), angle=12)  # 왼쪽 아래 책 (문서와 떨어짐)
        _paste(img, *_patch_book(rng, (40, 120, 200)), center=(W * 1.10, H * 0.62), angle=-8)  # 오른쪽 책 (문서와 떨어짐)
        _paste(img, *_patch_pen(), center=(W * 0.55, H * 0.93), angle=8)  # 아래 볼펜
    elif kind == "papers":  # 종이 더미
        _paste(img, *_patch_paper(rng), center=(W * 0.30, H * 0.40), angle=-14, shadow=0.15)  # 왼쪽 위 종이
        _paste(img, *_patch_paper(rng, color=(236, 238, 240)), center=(W * 0.72, H * 0.66), angle=9, shadow=0.15)  # 오른쪽 아래 종이
        _paste(img, *_patch_sticky(rng), center=(W * 0.85, H * 0.12), angle=6)  # 포스트잇
        _paste(img, *_patch_sticky(rng), center=(W * 0.10, H * 0.90), angle=-15)  # 포스트잇
    elif kind == "clutter":  # 잡동사니 책상
        _paste(img, *_patch_mug(), center=(W * 0.86, H * 0.18), angle=0)  # 머그컵
        _paste(img, *_patch_phone(), center=(W * 0.12, H * 0.30), angle=-20)  # 휴대폰
        _paste(img, *_patch_pen(), center=(W * 0.45, H * 0.94), angle=-5)  # 볼펜
        _paste(img, *_patch_sticky(rng), center=(W * 0.88, H * 0.85), angle=20)  # 포스트잇
        _paste(img, *_patch_book(rng, (60, 150, 90)), center=(W * 0.05, H * 0.85), angle=5)  # 책
    elif kind == "colorful":  # 알록달록한 책상 (잡지, 책)
        _paste(img, *_patch_book(rng, (200, 80, 160)), center=(W * 0.85, H * 0.25), angle=20)  # 보라 책
        _paste(img, *_patch_book(rng, (40, 200, 230)), center=(W * 0.15, H * 0.80), angle=-10)  # 노란 책
        _paste(img, *_patch_keyboard(), center=(W * 0.55, H * 0.97), angle=3)  # 아래 키보드
        _paste(img, *_patch_mug(), center=(W * 0.12, H * 0.12), angle=0)  # 머그컵
    return img  # 배경 반환


# ───────────────────────── 카메라 원근 ─────────────────────────

def _project(dw: int, dh: int, yaw: float, pitch: float, roll: float, fill: float, offset: tuple[float, float],
             W: int = PHOTO_W, H: int = PHOTO_H) -> np.ndarray:  # 3D 카메라로 문서 네 꼭짓점이 사진 어디에 찍히는지 계산
    P = np.array([[-dw / 2, -dh / 2, 0], [dw / 2, -dh / 2, 0], [dw / 2, dh / 2, 0], [-dw / 2, dh / 2, 0]], np.float64)  # 책상 위 문서 꼭짓점 (3D)
    y, p, r = np.deg2rad([yaw, pitch, roll])  # 각도를 라디안으로
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])  # 책상 위에서 문서 돌리기 (z축 회전)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])  # 카메라 앞뒤로 눕히기 (x축 회전)
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])  # 카메라 좌우로 기울이기 (y축 회전)
    Q = P @ (Ry @ Rx @ Rz).T  # 세 회전을 행렬 곱으로 한 번에 적용
    D = 1.5 * max(dw, dh)  # 카메라와 문서 사이 거리 (가까울수록 원근이 강함)
    uv = D * Q[:, :2] / (D + Q[:, 2:3])  # 핀홀 카메라 투영: 멀리 있는 점일수록 작게
    x, yv = uv[:, 0], uv[:, 1]  # 투영된 좌표
    area = 0.5 * abs(np.dot(x, np.roll(yv, -1)) - np.dot(yv, np.roll(x, -1)))  # 신발끈 공식으로 사각형 넓이
    uv *= np.sqrt(fill * W * H / area)  # 사진의 fill 비율만큼 차지하도록 크기 조절
    span = uv.max(axis=0) - uv.min(axis=0)  # 문서가 차지하는 가로/세로 폭
    shrink = min(1.0, (W * 0.90) / span[0], (H * 0.90) / span[1])  # 사진 밖으로 나가면 줄이기 (여백 5%)
    uv *= shrink  # 크기 보정
    uv -= (uv.max(axis=0) + uv.min(axis=0)) / 2  # 문서 가운데를 원점으로
    room = (np.array([W, H]) * 0.95 - (uv.max(axis=0) - uv.min(axis=0))) / 2  # 남은 여유 공간
    uv += np.array([W / 2, H / 2]) + np.clip(np.array(offset) * np.array([W, H]), -room, room)  # 사진 중심 + 치우침
    return uv.astype(np.float32)  # (4, 2) 사진 좌표 (문서 기준 좌상, 우상, 우하, 좌하)


# ───────────────────────── 조명 / 가림 효과 ─────────────────────────

def _apply_gain(img: np.ndarray, mask: np.ndarray, darkness: float, tint=(1.05, 1.0, 0.95)) -> None:  # 마스크 영역 어둡게 (그림자는 살짝 푸르스름)
    g = 1 - (1 - darkness) * mask  # 밝기 배율 (마스크 1 → darkness)
    for ch, t in enumerate(tint):  # 채널마다
        img[..., ch] *= g * (1 + (t - 1) * mask)  # 배율 적용 (그림자 색감 포함)


def _hand_shadow(img: np.ndarray, center, angle: float, darkness: float, soft: float, scale: float = 1.4) -> None:  # 화면 밖 손이 드리운 그림자
    _, a = _patch_hand(scale)  # 손 모양만 사용
    H, W = img.shape[:2]  # 사진 크기
    M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), angle, 1.0)  # 회전 행렬
    M[:, 2] += np.array(center) - np.array([a.shape[1] / 2, a.shape[0] / 2])  # 위치 이동
    mask = cv2.GaussianBlur(cv2.warpAffine(a, M, (W, H)), (0, 0), soft)  # 흐린 그림자 (soft 클수록 경계가 부드러움)
    _apply_gain(img, mask, darkness)  # 어둡게


def _window_shadow(img: np.ndarray, darkness: float, soft: float) -> None:  # 창틀 그림자 (선명한 대각선 경계)
    H, W = img.shape[:2]  # 사진 크기
    m = np.zeros((H, W), np.float32)  # 그림자 마스크
    cv2.fillPoly(m, [np.array([[0, int(H * 0.25)], [W, int(H * 0.62)], [W, H], [0, H]], np.int32)], 1.0)  # 아래쪽 절반 그림자
    cv2.fillPoly(m, [np.array([[0, int(H * 0.47)], [W, int(H * 0.84)], [W, int(H * 0.92)], [0, int(H * 0.55)]], np.int32)], 0.0)  # 창틀 사이로 들어온 빛 줄기
    _apply_gain(img, cv2.GaussianBlur(m, (0, 0), soft), darkness)  # 어둡게


def _phone_shadow(img: np.ndarray, center, angle: float, darkness: float, soft: float) -> None:  # 촬영하는 휴대폰의 그림자
    H, W = img.shape[:2]  # 사진 크기
    m = np.zeros((H, W), np.float32)  # 그림자 마스크
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (700.0, 1100.0), angle)).astype(np.int32)  # 기울어진 직사각형 꼭짓점
    cv2.fillPoly(m, [box], 1.0)  # 직사각형 채우기
    _apply_gain(img, cv2.GaussianBlur(m, (0, 0), soft), darkness)  # 어둡게


def _side_light(img: np.ndarray, strength: float) -> None:  # 한쪽에서만 비추는 조명 (반대쪽이 어두움)
    H, W = img.shape[:2]  # 사진 크기
    ramp = np.linspace(1.0, 1.0 - strength, W, dtype=np.float32)[None, :] * np.linspace(1.0, 0.85, H, dtype=np.float32)[:, None]  # 왼쪽 위 밝고 오른쪽 아래 어두움
    img *= ramp[..., None]  # 적용


def _glare(img: np.ndarray, center, axes, angle: float, power: float) -> None:  # 형광등 빛 반사 (하얗게 날아감)
    H, W = img.shape[:2]  # 사진 크기
    m = np.zeros((H, W), np.float32)  # 반사 마스크
    cv2.ellipse(m, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle, 0, 360, 1.0, -1)  # 타원 반사
    m = cv2.GaussianBlur(m, (0, 0), 45)  # 가장자리 번짐
    img += (power * 255 * m)[..., None]  # 밝게 더하기 (나중에 255에서 잘림 = 하얗게 날아감)


def _vignette(img: np.ndarray, strength: float) -> None:  # 렌즈 주변부 어두워짐
    H, W = img.shape[:2]  # 사진 크기
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)  # 좌표 격자
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2  # 중심에서 거리² (모서리 ≈ 2)
    img *= (1 - strength * r2 / 2)[..., None]  # 바깥으로 갈수록 어둡게


def _camera(img: np.ndarray, rng: np.random.Generator, blur: float, noise: float) -> np.ndarray:  # 카메라 렌즈 흐림 + 센서 노이즈
    img = cv2.GaussianBlur(img, (0, 0), blur)  # 초점이 살짝 흐림
    img += rng.normal(0, noise, img.shape[:2])[..., None]  # 밝기 노이즈 (모든 채널 공통)
    img += rng.normal(0, noise * 0.4, img.shape)  # 색 노이즈 (채널마다 조금씩)
    return np.clip(img, 0, 255).astype(np.uint8)  # 0~255 로 자르고 정수로


# ───────────────────────── 20장 촬영 계획 ─────────────────────────

CASES = [  # 각 사진의 조건 (yaw=책상 위 문서 회전, pitch/roll=카메라 기울기, fill=사진에서 문서 넓이 비율, offset=중심에서 치우침)
    dict(name="easy_01", category="easy", doc="a4_report", bg="wood", yaw=2, pitch=5, fill=0.50, lighting="uniform", angle="front", background="simple", notes="나무 책상, 위에서 수직 촬영, 균일 조명"),  # 쉬움 1
    dict(name="easy_02", category="easy", doc="receipt", bg="dark", yaw=-4, pitch=4, fill=0.24, lighting="uniform", angle="front", background="simple", notes="검은 책상 위 영수증"),  # 쉬움 2
    dict(name="easy_03", category="easy", doc="card", bg="gray", yaw=5, pitch=3, fill=0.22, lighting="uniform", angle="front", background="simple", notes="회색 책상 위 명함 근접 촬영"),  # 쉬움 3
    dict(name="easy_04", category="easy", doc="a4_form", bg="mat", yaw=-2, pitch=4, fill=0.48, lighting="uniform", angle="front", background="simple", notes="초록 커팅 매트 위 신청서 (매트 격자선 있음)"),  # 쉬움 4
    dict(name="easy_05", category="easy", doc="a4_report", bg="wood_dark", yaw=1, pitch=5, fill=0.55, lighting="uniform", angle="front", background="simple", notes="진한 나무 책상 위 보고서"),  # 쉬움 5
    dict(name="shadow_01", category="shadow", doc="a4_form", bg="wood", yaw=3, pitch=4, fill=0.50, lighting="shadow", angle="front", background="simple",
         effects=[("hand_shadow", dict(center=(1050, 1500), angle=-35, darkness=0.55, soft=28))], notes="손 그림자(부드러운 경계)가 오른쪽 아래 모서리를 덮음"),  # 그림자 1
    dict(name="shadow_02", category="shadow", doc="a4_report", bg="gray", yaw=-3, pitch=6, fill=0.50, lighting="shadow", angle="front", background="simple",
         effects=[("window_shadow", dict(darkness=0.32, soft=2.5))], notes="창틀 그림자(선명한 경계)가 문서를 대각선으로 가로지름"),  # 그림자 2
    dict(name="shadow_03", category="shadow", doc="receipt", bg="dark", yaw=6, pitch=5, fill=0.24, lighting="dim", angle="front", background="simple",
         effects=[("dim", dict(gain=0.16))], noise=7.0, notes="불 꺼진 방에서 촬영한 어두운 사진 (고감도 노이즈)"),  # 그림자 3 (어두움)
    dict(name="shadow_04", category="shadow", doc="a4_report", bg="white_desk", yaw=-5, pitch=5, fill=0.50, lighting="shadow", angle="front", background="simple",
         effects=[("phone_shadow", dict(center=(760, 300), angle=8, darkness=0.6, soft=35))], notes="흰 책상(문서와 대비 낮음) + 촬영 휴대폰 그림자가 위쪽을 덮음"),  # 그림자 4
    dict(name="shadow_05", category="shadow", doc="a4_form", bg="wood", yaw=4, pitch=5, fill=0.48, lighting="shadow", angle="front", background="simple", crumple=16.0,
         effects=[("side_light", dict(strength=0.55))], notes="구겨진 종이 + 옆 조명으로 주름 그림자와 밝기 차이"),  # 그림자 5 (구겨짐)
    dict(name="tilt_01", category="tilt", doc="a4_report", bg="wood", yaw=45, pitch=15, fill=0.42, lighting="uniform", angle="tilt45+", background="simple", notes="문서를 45도 돌려 놓음"),  # 기울기 1
    dict(name="tilt_02", category="tilt", doc="a4_form", bg="gray", yaw=-55, pitch=25, roll=10, fill=0.40, lighting="uniform", angle="tilt45+", background="simple", notes="문서 -55도 + 카메라 25도 기울임"),  # 기울기 2
    dict(name="tilt_03", category="tilt", doc="receipt", bg="dark", yaw=60, pitch=12, fill=0.24, lighting="uniform", angle="tilt45+", background="simple", notes="영수증을 60도 돌려 놓음"),  # 기울기 3
    dict(name="tilt_04", category="tilt", doc="a4_report", bg="wood_dark", yaw=15, pitch=55, fill=0.40, lighting="uniform", angle="tilt45+", background="simple", notes="카메라를 55도 눕혀 촬영 (강한 원근)"),  # 기울기 4
    dict(name="tilt_05", category="tilt", doc="card", bg="gray", yaw=-70, pitch=35, roll=-12, fill=0.22, lighting="uniform", angle="tilt45+", background="simple", notes="명함 -70도 + 카메라 35도 기울임"),  # 기울기 5
    dict(name="complex_01", category="complex", doc="a4_report", bg="keyboard_books", yaw=6, pitch=5, fill=0.36, offset=(0, 0.04), lighting="uniform", angle="front", background="complex", notes="키보드, 책, 볼펜이 주변에 있음 (문서와 닿지 않음)"),  # 복잡 1
    dict(name="complex_02", category="complex", doc="receipt", bg="plaid", yaw=-12, pitch=8, fill=0.24, lighting="uniform", angle="front", background="complex", notes="체크무늬 천 위 영수증"),  # 복잡 2
    dict(name="complex_03", category="complex", doc="a4_form", bg="papers", yaw=4, pitch=8, fill=0.40, offset=(0.02, 0.02), lighting="uniform", angle="front", background="complex",
         notes="다른 흰 종이들과 겹쳐 있음 (흰색-흰색 경계)"),  # 복잡 3
    dict(name="complex_04", category="complex", doc="a4_report", bg="clutter", yaw=-8, pitch=12, fill=0.44, lighting="uniform", angle="front", background="complex",
         occlusion=dict(corner=2, angle=40), notes="잡동사니 책상 + 손이 오른쪽 아래 모서리를 가림 (부분 가림)"),  # 복잡 4 (가림)
    dict(name="complex_05", category="complex", doc="a4_form", bg="colorful", yaw=10, pitch=14, fill=0.42, lighting="glare", angle="front", background="complex",
         effects=[("glare", dict(corner=1, axes=(260, 150), angle=30, power=1.1))], notes="알록달록한 책상 + 형광등 빛 반사가 오른쪽 위 모서리를 덮음"),  # 복잡 5 (반사)
]  # 촬영 계획 끝


def render_case(case: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:  # 한 장 만들기 → (사진, 정답 꼭짓점)
    make_doc, _ = DOCS[case["doc"]]  # 문서 생성 함수
    doc = make_doc(rng)  # 문서 그림
    dh, dw = doc.shape[:2]  # 문서 크기
    if case.get("crumple"):  # 구겨짐이면
        doc, alpha = _crumple(doc, rng, case["crumple"])  # 구기기
    else:  # 평평하면
        alpha = _rect_alpha(dh, dw)  # 꽉 찬 투명도
    img = _make_background(case["bg"], rng)  # 배경
    dst = _project(dw, dh, case["yaw"], case.get("pitch", 0), case.get("roll", 0), case["fill"], case.get("offset", (0, 0)))  # 사진 속 문서 꼭짓점
    src = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])  # 문서 원래 꼭짓점
    M = cv2.getPerspectiveTransform(src, dst)  # 문서 → 사진 투시 행렬 (3x3)
    a = cv2.warpPerspective(alpha, M, (PHOTO_W, PHOTO_H))  # 투명도 옮기기 (가장자리 부드럽게)
    p = cv2.warpPerspective(doc, M, (PHOTO_W, PHOTO_H)).astype(np.float32)  # 문서 옮기기
    shadow = cv2.GaussianBlur(cv2.warpPerspective(alpha, M, (PHOTO_W, PHOTO_H), borderValue=0), (0, 0), 12)  # 종이 아래 그림자 모양
    shadow = np.roll(shadow, (10, 7), axis=(0, 1))  # 오른쪽 아래로 밀기
    img *= (1 - 0.35 * shadow)[..., None]  # 종이 그림자 적용
    img[:] = img * (1 - a[..., None]) + p * a[..., None]  # 배경 위에 문서 합성
    for kind, kw in case.get("effects", []):  # 조명 효과들
        if kind == "hand_shadow":  # 손 그림자
            _hand_shadow(img, **kw)  # 적용
        elif kind == "window_shadow":  # 창틀 그림자
            _window_shadow(img, **kw)  # 적용
        elif kind == "phone_shadow":  # 휴대폰 그림자
            _phone_shadow(img, **kw)  # 적용
        elif kind == "side_light":  # 옆 조명
            _side_light(img, **kw)  # 적용
        elif kind == "dim":  # 어두운 조명
            img *= kw["gain"]  # 전체 밝기를 낮춤
        elif kind == "glare":  # 빛 반사
            _glare(img, dst[kw["corner"]], kw["axes"], kw["angle"], kw["power"])  # 지정한 꼭짓점 위에 반사
    if case.get("occlusion"):  # 손으로 가림
        occ = case["occlusion"]  # 가림 설정
        hand, ha = _patch_hand(1.3)  # 손 그림 (실제 손 크기 비율)
        corner = dst[occ["corner"]]  # 가릴 꼭짓점
        t = np.deg2rad(occ["angle"])  # 손 회전 각도 (라디안)
        finger_dir = np.array([-np.sin(t), -np.cos(t)])  # 회전 후 손가락이 가리키는 방향 (원래는 위쪽)
        center = corner - finger_dir * hand.shape[0] * 0.22  # 손가락 끝이 꼭짓점을 넘어가도록 손 중심 위치 계산
        _paste(img, hand, ha, center=tuple(center), angle=occ["angle"], shadow=0.45, shadow_off=(25, 35), shadow_blur=45)  # 손 붙이기
    _vignette(img, 0.18)  # 렌즈 주변부 어두움
    img *= np.array([0.97, 1.0, 1.04], np.float32)  # 실내 조명의 따뜻한 색감
    photo = _camera(img, rng, blur=case.get("blur", 1.1), noise=case.get("noise", 2.5))  # 카메라 흐림 + 노이즈
    return photo, dst  # 사진과 정답 꼭짓점


def _save_jpeg(path: Path, img: np.ndarray, quality: int = 90) -> Path:  # 휴대폰처럼 JPEG 압축 저장 (한글 경로 지원)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])  # 압축
    if not ok:  # 실패하면
        raise IOError(f"저장 실패: {path}")  # 오류
    buf.tofile(str(path))  # 파일 기록
    return path  # 경로 반환


def generate_dataset(folder: Path, seed: int = 2026, force: bool = False) -> list[Path]:  # 20장 + metadata.csv 생성
    folder.mkdir(parents=True, exist_ok=True)  # 폴더 준비
    existing = list_images(folder)  # 이미 있는 이미지
    if existing and not force:  # 사용자 사진이 있으면 덮어쓰지 않음
        raise FileExistsError(f"{folder} 에 이미지 {len(existing)}장이 이미 있습니다. 덮어쓰려면 --force 를 붙이세요.")  # 안내
    rng = np.random.default_rng(seed)  # 같은 seed 면 항상 같은 사진
    rows, paths = {}, []  # 메타데이터 / 경로 목록
    for case in CASES:  # 계획대로 한 장씩
        photo, dst = render_case(case, rng)  # 사진 만들기
        path = _save_jpeg(folder / f"{case['name']}.jpg", photo)  # 저장
        paths.append(path)  # 경로 기록
        _, kind = DOCS[case["doc"]]  # 문서 종류
        row = {c: "" for c in COLUMNS}  # 빈 행
        row.update(filename=path.name, category=case["category"], lighting=case["lighting"], angle=case["angle"],  # 조건 기록
                   background=case["background"], doc_type=kind, true_aspect=DOC_ASPECT[kind], notes=f"[합성] {case['notes']}")  # 문서/메모
        for col, val in zip(CORNER_COLS, order_points(dst).reshape(-1)):  # 정답 꼭짓점 8개
            row[col] = f"{val:.1f}"  # 소수 1자리로 기록
        rows[path.name] = row  # 사전에 추가
    write_metadata(rows, folder / "metadata.csv")  # CSV 저장
    return paths  # 생성한 경로 목록
