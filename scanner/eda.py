"""이미지 탐색적 분석(EDA): 히스토그램, 밝기 통계, 쉬운/어려운 케이스 비교, 어두운 이미지와 엣지."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

import csv  # 통계 CSV 저장
from pathlib import Path  # 경로 처리

import cv2  # 영상 처리
import numpy as np  # 수치 계산

from .config import GENERATED_DIR, IMAGES_DIR  # 저장 폴더와 이미지 폴더
from .dataset import load_records  # 이미지 + 메타데이터
from .pipeline import load_image, resize_for_processing  # 이미지 읽기/축소
from .plotting import plt  # 그래프 도구 (한글 폰트 설정 포함)


def image_stats(gray: np.ndarray) -> dict:  # 밝기 분포 통계 계산
    g = gray.astype(np.float32)  # 계산 정밀도를 위해 실수형으로
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)  # 가로 방향 밝기 변화량 (x 미분)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)  # 세로 방향 밝기 변화량 (y 미분)
    mag = np.sqrt(gx ** 2 + gy ** 2)  # 기울기 크기 = 엣지 세기
    p5, p50, p95 = np.percentile(g, [5, 50, 95])  # 하위 5%, 중앙값, 상위 95% 밝기
    return {  # 통계 사전 반환
        "mean": round(float(g.mean()), 1),  # 평균 밝기 (0=검정, 255=흰색)
        "std": round(float(g.std()), 1),  # 표준편차 = 대비(contrast)
        "p5": round(float(p5), 1),  # 어두운 쪽 밝기
        "median": round(float(p50), 1),  # 중앙 밝기
        "p95": round(float(p95), 1),  # 밝은 쪽 밝기
        "dynamic_range": round(float(p95 - p5), 1),  # 실제로 쓰는 밝기 폭 (좁을수록 흐릿)
        "dark_ratio": round(float((g < 60).mean()), 3),  # 어두운 픽셀(60 미만) 비율
        "bright_ratio": round(float((g > 220).mean()), 3),  # 매우 밝은 픽셀(220 초과) 비율 (빛 반사 지표)
        "grad_mean": round(float(mag.mean()), 2),  # 평균 엣지 세기
        "grad_p99": round(float(np.percentile(mag, 99)), 1),  # 상위 1% 엣지 세기 (문서 테두리 같은 강한 경계)
    }  # 사전 끝


def dark_edge_demo(gray: np.ndarray, out: Path, low: int = 50, high: int = 150) -> list[dict]:  # 어두울수록 엣지가 사라지는 실험
    """같은 이미지의 밝기를 factor배로 줄이면 히스토그램이 왼쪽으로 압축되고,
    픽셀 간 차이(기울기)도 같은 비율로 줄어 Canny 고정 임계값을 넘지 못한다."""  # 함수 설명
    factors = [1.0, 0.6, 0.35, 0.2]  # 밝기 배율 (1.0=원본)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 노이즈 제거
    fig, axes = plt.subplots(3, len(factors), figsize=(4 * len(factors), 10))  # 3행(이미지/히스토그램/엣지) 그림
    rows = []  # 결과 표
    for k, f in enumerate(factors):  # 배율마다
        dark = np.clip(blurred.astype(np.float32) * f, 0, 255).astype(np.uint8)  # 밝기를 f배로 줄임
        edges = cv2.Canny(dark, low, high)  # 같은 임계값으로 엣지 검출
        st = image_stats(dark)  # 통계 계산
        edge_ratio = float((edges > 0).mean())  # 엣지 픽셀 비율
        rows.append({"factor": f, "mean": st["mean"], "std": st["std"], "grad_p99": st["grad_p99"], "edge_ratio": round(edge_ratio, 4)})  # 기록
        axes[0, k].imshow(dark, cmap="gray", vmin=0, vmax=255)  # 1행: 어두워진 이미지
        axes[0, k].set_title(f"밝기 ×{f}")  # 제목
        axes[0, k].axis("off")  # 축 숨김
        axes[1, k].hist(dark.ravel(), bins=64, range=(0, 255), color="gray")  # 2행: 히스토그램
        axes[1, k].set_title(f"평균 {st['mean']}, 표준편차 {st['std']}")  # 통계 제목
        axes[2, k].imshow(edges, cmap="gray")  # 3행: 엣지 결과
        axes[2, k].set_title(f"Canny({low},{high}) 엣지 {edge_ratio * 100:.2f}%")  # 엣지 비율 제목
        axes[2, k].axis("off")  # 축 숨김
    fig.suptitle("어두워질수록 히스토그램이 좁아지고, 기울기가 임계값 아래로 떨어져 엣지가 사라진다")  # 전체 제목
    fig.tight_layout()  # 여백 정리
    fig.savefig(out, dpi=120)  # 저장
    plt.close(fig)  # 메모리 해제
    return rows  # 결과 반환


def run_eda(folder: Path = IMAGES_DIR, report_dir: Path = GENERATED_DIR) -> None:  # EDA 전체 실행
    records = load_records(folder, folder / "metadata.csv")  # 이미지 목록
    if not records:  # 없으면
        print("[eda] images/ 폴더에 이미지가 없습니다.")  # 안내
        return  # 종료
    report_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    grays, stats = [], []  # 그레이 이미지와 통계 목록
    for rec in records:  # 이미지마다
        small, _ = resize_for_processing(load_image(rec.path), 800)  # 같은 해상도로 맞춰 비교 공정성 확보
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # 그레이 변환
        grays.append(gray)  # 저장
        stats.append({"filename": rec.name, "category": rec.category, "group": "easy" if rec.category == "easy" else "hard", **image_stats(gray)})  # 통계 저장
    with (report_dir / "eda_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:  # 통계 CSV
        w = csv.DictWriter(f, fieldnames=list(stats[0]))  # 작성기
        w.writeheader()  # 머리줄
        w.writerows(stats)  # 내용
    cols = 5  # 한 줄 5칸
    rows_n = int(np.ceil(len(records) / cols))  # 필요한 줄 수
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)  # 개별 히스토그램 격자
    for ax, gray, st in zip(axes.ravel(), grays, stats):  # 이미지마다
        ax.hist(gray.ravel(), bins=64, range=(0, 255), color="tab:blue" if st["group"] == "easy" else "tab:red")  # 쉬움=파랑, 어려움=빨강
        ax.set_title(f"{st['filename']}\n{st['category']} μ={st['mean']} σ={st['std']}", fontsize=8)  # 제목
    for ax in axes.ravel()[len(records):]:  # 남는 칸은
        ax.axis("off")  # 숨김
    fig.tight_layout()  # 여백 정리
    fig.savefig(report_dir / "eda_histograms.png", dpi=110)  # 저장
    plt.close(fig)  # 메모리 해제
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 쉬움 vs 어려움 비교 그림
    for group, color in (("easy", "tab:blue"), ("hard", "tab:red")):  # 두 그룹
        idx = [i for i, s in enumerate(stats) if s["group"] == group]  # 해당 그룹 이미지 번호
        if not idx:  # 그룹이 비었으면
            continue  # 건너뜀
        hists = np.array([np.histogram(grays[i], bins=64, range=(0, 255), density=True)[0] for i in idx])  # 정규화 히스토그램들
        centers = np.linspace(2, 253, 64)  # 막대 중심 위치
        axes[0].plot(centers, hists.mean(axis=0), color=color, label=f"{group} (n={len(idx)})")  # 평균 히스토그램
        axes[0].fill_between(centers, hists.min(axis=0), hists.max(axis=0), color=color, alpha=0.15)  # 최소~최대 범위
    axes[0].set_title("쉬운 vs 어려운 케이스 평균 히스토그램")  # 제목
    axes[0].set_xlabel("밝기 (0~255)")  # 가로축
    axes[0].legend()  # 범례
    for ax, key, title in ((axes[1], "std", "대비(표준편차)"), (axes[2], "grad_p99", "강한 엣지 세기(기울기 상위 1%)")):  # 상자 그림 2개
        data = [[s[key] for s in stats if s["group"] == g] for g in ("easy", "hard")]  # 그룹별 값
        ax.boxplot([d if d else [0] for d in data], tick_labels=["easy", "hard"])  # 상자 그림
        ax.set_title(title)  # 제목
    fig.tight_layout()  # 여백 정리
    fig.savefig(report_dir / "eda_easy_vs_hard.png", dpi=120)  # 저장
    plt.close(fig)  # 메모리 해제
    brightest = int(np.argmax([s["mean"] for s in stats]))  # 가장 밝은 이미지로 어둡게 만드는 실험
    demo = dark_edge_demo(grays[brightest], report_dir / "eda_dark_edges.png")  # 실험 실행
    md = ["# 🟩 EDA 결과 (자동 생성)", "", "## 🟢 1. 이미지별 밝기 통계", "",  # 문서 머리
          "| 파일 | 조건 | 평균 | 표준편차 | 밝기 폭(p95-p5) | 어두운 비율 | 밝은 비율 | 강한 엣지(p99) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]  # 표 머리
    for s in stats:  # 이미지마다
        md.append(f"| {s['filename']} | {s['category']} | {s['mean']} | {s['std']} | {s['dynamic_range']} | {s['dark_ratio']} | {s['bright_ratio']} | {s['grad_p99']} |")  # 한 줄
    md += ["", "![histograms](eda_histograms.png)", "", "## 🟢 2. 쉬운 vs 어려운 케이스", "", "![easy vs hard](eda_easy_vs_hard.png)", "",  # 그림 링크
           f"## 🟢 3. 어두운 이미지에서 엣지 검출이 어려운 이유 (실험 대상: {stats[brightest]['filename']})", "",  # 실험 제목
           "| 밝기 배율 | 평균 | 표준편차 | 기울기 p99 | 엣지 픽셀 비율 |", "| --- | --- | --- | --- | --- |"]  # 실험 표 머리
    for r in demo:  # 배율마다
        md.append(f"| ×{r['factor']} | {r['mean']} | {r['std']} | {r['grad_p99']} | {r['edge_ratio']} |")  # 한 줄
    md += ["", "- 밝기를 f배 하면 픽셀 값 차이(= 기울기)도 f배가 된다. Canny 임계값은 고정이므로 기울기가 low/high 아래로 떨어진 경계는 사라진다.",  # 해설 1
           "- 히스토그램 관점: 분포가 0 근처로 몰리고 폭(표준편차)이 좁아진다 → 문서와 배경의 밝기 차이가 줄어든다.",  # 해설 2
           "- 어두운 영역은 센서 노이즈 비중이 커서(신호 대 잡음비 감소) 임계값을 낮추면 노이즈 엣지가 함께 늘어난다.", "",  # 해설 3
           "![dark edges](eda_dark_edges.png)", ""]  # 그림 링크
    (report_dir / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")  # 마크다운 저장
    print(f"[eda] 저장: {report_dir / 'eda_summary.md'}")  # 완료 안내
