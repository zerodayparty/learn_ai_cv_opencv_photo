"""테스트 이미지 목록과 metadata.csv(촬영 조건 + 정답 꼭짓점)를 읽고 쓰는 모듈."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

import csv  # CSV 파일 읽기/쓰기
from dataclasses import dataclass  # 레코드 클래스 생성
from pathlib import Path  # 경로 처리

import numpy as np  # 좌표 배열

from .config import IMAGES_DIR, METADATA_CSV  # 기본 이미지 폴더와 CSV 경로

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}  # 이미지로 인정할 확장자
CORNER_COLS = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]  # 정답 꼭짓점 열 이름 (좌상→우상→우하→좌하)
COLUMNS = [  # metadata.csv의 전체 열 순서
    "filename",  # 이미지 파일 이름
    "category",  # 조건 분류: easy / shadow / tilt / complex / crumpled / occlusion / glare
    "lighting",  # 조명: uniform(균일) / dim(어두움) / shadow(그림자) / glare(반사)
    "angle",  # 촬영 각도: front(정면) / tilt30 / tilt45+ 등
    "background",  # 배경 복잡도: simple(단순) / complex(복잡)
    "doc_type",  # 문서 종류: a4 / receipt / card 등
    "true_aspect",  # 실제 문서 종횡비 = 긴 변 ÷ 짧은 변 (A4=1.414, 명함 90x50mm=1.8, 모르면 비움)
    "notes",  # 기타 메모
    *CORNER_COLS,  # 정답 꼭짓점 좌표 8개 (annotate 명령으로 클릭 입력 가능)
    "manual_result",  # 사람이 직접 판정한 결과 (success / partial / fail, 정답 좌표가 없을 때 사용)
]  # 열 목록 끝

CATEGORY_LABELS = {  # 보고서 표에 쓸 한글 이름
    "easy": "단순 배경 + 정면",  # 쉬운 케이스
    "shadow": "그림자 있음",  # 그림자
    "tilt": "45도 이상 기울어짐",  # 기울어짐
    "complex": "복잡한 배경",  # 복잡한 배경
    "crumpled": "구겨짐",  # 구겨짐
    "occlusion": "부분 가림",  # 가림
    "glare": "빛 반사",  # 반사
}  # 사전 끝


@dataclass  # 레코드 클래스 자동 생성
class Record:  # 이미지 한 장의 메타데이터
    row: dict  # CSV 한 줄 전체 (문자열 사전)
    path: Path  # 이미지 실제 경로

    @property  # 속성처럼 사용
    def name(self) -> str:  # 파일 이름
        return self.row["filename"]  # filename 열 값

    @property  # 속성처럼 사용
    def category(self) -> str:  # 조건 분류
        return (self.row.get("category") or "unknown").strip()  # 비어 있으면 unknown

    @property  # 속성처럼 사용
    def gt_corners(self) -> np.ndarray | None:  # 정답 꼭짓점 (없으면 None)
        vals = [self.row.get(c, "") for c in CORNER_COLS]  # 8개 좌표 문자열 꺼내기
        if any(v in ("", None) for v in vals):  # 하나라도 비어 있으면
            return None  # 정답 없음
        return np.array([float(v) for v in vals], dtype=np.float32).reshape(4, 2)  # (4, 2) 숫자 배열로 변환

    @property  # 속성처럼 사용
    def true_aspect(self) -> float | None:  # 실제 종횡비 (긴 변 ÷ 짧은 변)
        v = (self.row.get("true_aspect") or "").strip()  # 문자열 값
        return float(v) if v else None  # 숫자로 변환 (없으면 None)


def list_images(folder: Path = IMAGES_DIR) -> list[Path]:  # 폴더 안의 이미지 파일 목록
    if not folder.exists():  # 폴더가 없으면
        return []  # 빈 목록
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)  # 이미지 확장자만 이름순 정렬


def read_metadata(csv_path: Path = METADATA_CSV) -> dict[str, dict]:  # CSV를 {파일이름: 행} 사전으로 읽기
    if not csv_path.exists():  # 파일이 없으면
        return {}  # 빈 사전
    with csv_path.open(encoding="utf-8-sig", newline="") as f:  # BOM이 있어도 읽히도록 utf-8-sig 사용
        return {r["filename"]: {c: (r.get(c) or "") for c in COLUMNS} for r in csv.DictReader(f) if r.get("filename")}  # 열 이름 기준으로 정리


def write_metadata(rows: dict[str, dict], csv_path: Path = METADATA_CSV) -> None:  # 사전을 CSV로 저장
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:  # 엑셀에서도 한글이 깨지지 않게 BOM 포함 저장
        writer = csv.DictWriter(f, fieldnames=COLUMNS)  # 열 순서를 고정한 작성기
        writer.writeheader()  # 첫 줄에 열 이름 기록
        for name in sorted(rows):  # 파일 이름순으로
            writer.writerow({c: rows[name].get(c, "") for c in COLUMNS})  # 한 줄씩 기록


def sync_metadata(folder: Path = IMAGES_DIR, csv_path: Path = METADATA_CSV) -> dict[str, dict]:  # 이미지 폴더와 CSV를 맞춤
    rows = read_metadata(csv_path)  # 기존 CSV 읽기
    for img in list_images(folder):  # 폴더의 모든 이미지에 대해
        rows.setdefault(img.name, {c: "" for c in COLUMNS} | {"filename": img.name})  # CSV에 없으면 빈 행 추가
    write_metadata(rows, csv_path)  # 다시 저장
    return rows  # 최신 행들 반환


def load_records(folder: Path = IMAGES_DIR, csv_path: Path = METADATA_CSV) -> list[Record]:  # 평가용 레코드 목록
    rows = sync_metadata(folder, csv_path)  # 폴더와 CSV 동기화
    records = []  # 결과 목록
    for img in list_images(folder):  # 실제 존재하는 이미지만
        records.append(Record(row=rows[img.name], path=img))  # 레코드 생성
    return records  # 목록 반환
