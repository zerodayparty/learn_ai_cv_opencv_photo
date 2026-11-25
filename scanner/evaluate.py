"""정량 평가: 20장 일괄 스캔, 성공 판정, 조건별 성공률 표, 파라미터 스윕(딜레마 분석)."""  # 모듈 설명

from __future__ import annotations  # 타입 힌트 호환성

import csv  # 결과 CSV 저장
import itertools  # 파라미터 조합 생성
from collections import defaultdict  # 그룹별 집계용 사전
from pathlib import Path  # 경로 처리

import cv2  # 이미지 그리기/합치기
import numpy as np  # 수치 계산

from .config import GENERATED_DIR, IMAGES_DIR, OUTPUT_DIR, ScanParams  # 이미지/저장 위치와 파라미터
from .dataset import CATEGORY_LABELS, Record, load_records  # 테스트 이미지 레코드
from .geometry import corner_error, quad_aspect  # 오차 계산
from .pipeline import ScanResult, load_image, run_pipeline, save_image  # 파이프라인

CORNER_TOL = 0.03  # 꼭짓점 허용 오차: 이미지 대각선 길이의 3% 이내면 '올바르게 검출'
ASPECT_TOL = 0.05  # 종횡비 허용 오차: 실제 비율과 5% 넘게 다르면 '부분 성공'


def judge(rec: Record, res: ScanResult) -> dict:  # 한 장의 성공/부분 성공/실패 판정
    h, w = res.original.shape[:2]  # 원본 크기
    diag = float(np.hypot(w, h))  # 대각선 길이 (오차를 비율로 바꾸기 위함)
    out = {"found": res.found, "corner_err_px": "", "corner_err_ratio": "", "aspect": "", "aspect_err": ""}  # 기본 값
    if res.found:  # 문서를 찾았으면
        out["aspect"] = round(quad_aspect(res.corners), 3)  # 결과 종횡비 (긴 변 ÷ 짧은 변)
    gt = rec.gt_corners  # 정답 좌표
    if gt is not None:  # 정답이 있으면 자동 판정
        if not res.found:  # 못 찾았으면
            out["status"], out["judged_by"] = "fail", "auto"  # 실패
            return out  # 종료
        err = corner_error(res.corners, gt)  # 최대 꼭짓점 오차(px)
        out["corner_err_px"], out["corner_err_ratio"] = round(err, 1), round(err / diag, 4)  # 오차 기록
        if err / diag > CORNER_TOL:  # 허용 오차보다 크면
            out["status"], out["judged_by"] = "fail", "auto"  # 엉뚱한 사각형을 찾은 것 → 실패
            return out  # 종료
        status = "success"  # 꼭짓점은 맞음
        if rec.true_aspect:  # 실제 문서 비율을 알고 있으면
            err_a = abs(out["aspect"] - rec.true_aspect) / rec.true_aspect  # 상대 오차
            out["aspect_err"] = round(err_a, 4)  # 기록
            if err_a > ASPECT_TOL:  # 5% 넘게 다르면
                status = "partial"  # 부분 성공 (원근 때문에 비율이 찌그러짐)
        out["status"], out["judged_by"] = status, "auto"  # 판정 기록
        return out  # 종료
    manual = (rec.row.get("manual_result") or "").strip().lower()  # 사람이 적은 판정
    if manual in ("success", "partial", "fail"):  # 올바른 값이면
        out["status"], out["judged_by"] = manual, "manual"  # 수동 판정 사용
    else:  # 정답도 수동 판정도 없으면
        out["status"], out["judged_by"] = ("detected" if res.found else "fail"), "unverified"  # 검출 여부만 기록
    return out  # 결과 반환


def _rate_table(groups: dict[str, list[dict]], title: str, labels: dict | None = None) -> str:  # 그룹별 성공률 마크다운 표
    lines = [f"| {title} | 테스트 수 | 성공 | 부분 성공 | 실패 | 미검증 | 성공률 |", "| --- | --- | --- | --- | --- | --- | --- |"]  # 표 머리
    total = defaultdict(int)  # 전체 합계
    for key in sorted(groups):  # 그룹마다
        rows = groups[key]  # 해당 그룹 결과
        c = {s: sum(r["status"] == s for r in rows) for s in ("success", "partial", "fail", "detected")}  # 상태별 개수
        n = len(rows)  # 테스트 수
        rate = c["success"] / n * 100 if n else 0  # 성공률(%)
        name = (labels or {}).get(key, key or "(미기록)")  # 표시 이름
        lines.append(f"| {name} | {n} | {c['success']} | {c['partial']} | {c['fail']} | {c['detected']} | {rate:.0f}% |")  # 한 줄
        for k, v in c.items():  # 합계 누적
            total[k] += v  # 상태별 더하기
        total["n"] += n  # 개수 더하기
    if total["n"]:  # 데이터가 있으면
        lines.append(f"| **전체** | {total['n']} | {total['success']} | {total['partial']} | {total['fail']} | {total['detected']} | {total['success'] / total['n'] * 100:.0f}% |")  # 합계 줄
    return "\n".join(lines)  # 문자열로 합쳐 반환


def _contact_sheet(items: list[tuple[str, str, np.ndarray]], path: Path, cell_h: int = 260) -> None:  # 결과 한눈에 보기 이미지
    if not items:  # 결과가 없으면
        return  # 아무것도 안 함
    colors = {"success": (0, 180, 0), "partial": (0, 200, 255), "fail": (0, 0, 255), "detected": (255, 160, 0)}  # 상태별 색
    cells = []  # 각 칸 이미지
    for name, status, img in items:  # 결과마다
        cell = cv2.resize(img, (int(img.shape[1] * cell_h / img.shape[0]), cell_h))  # 같은 높이로 축소
        cell = cv2.copyMakeBorder(cell, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=colors.get(status, (128, 128, 128)))  # 위쪽 색 띠
        cv2.putText(cell, f"{name} [{status}]", (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)  # 이름/상태 글씨
        cells.append(cell)  # 목록에 추가
    cell_w = max(c.shape[1] for c in cells)  # 가장 넓은 칸 너비
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, cell_w - c.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for c in cells]  # 너비 통일
    cols = 5  # 한 줄에 5칸
    while len(cells) % cols:  # 마지막 줄이 비면
        cells.append(np.full_like(cells[0], 40))  # 회색 빈칸으로 채움
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]  # 가로로 이어 붙여 줄 만들기
    save_image(path, np.vstack(rows))  # 세로로 쌓아 저장


def evaluate(params: ScanParams, folder: Path = IMAGES_DIR, out_dir: Path = OUTPUT_DIR / "batch", report_dir: Path = GENERATED_DIR) -> list[dict]:  # 일괄 평가
    records = load_records(folder, folder / "metadata.csv")  # 이미지 + 메타데이터
    if not records:  # 이미지가 없으면
        print("[evaluate] images/ 폴더에 이미지가 없습니다.")  # 안내
        return []  # 빈 결과
    results, sheet = [], []  # 결과 표와 모아보기 목록
    for rec in records:  # 이미지마다
        res = run_pipeline(load_image(rec.path), params)  # 파이프라인 실행
        j = judge(rec, res)  # 성공 판정
        stem = rec.path.stem  # 확장자 뺀 이름
        save_image(out_dir / f"{stem}_contours.jpg", res.contour_vis)  # 윤곽선 시각화 저장
        if res.found:  # 찾았으면
            save_image(out_dir / f"{stem}_warped.jpg", res.warped)  # 컬러 변환본 저장
            save_image(out_dir / f"scanned_{stem}.png", res.scanned)  # 최종 스캔본 저장
        row = {"filename": rec.name, **{k: rec.row.get(k, "") for k in ("category", "lighting", "angle", "background", "doc_type")}, **j, **res.info}  # 결과 한 줄
        results.append(row)  # 목록에 추가
        sheet.append((stem, j["status"], res.contour_vis))  # 모아보기에 추가
        print(f"  {rec.name:<28} {j['status']:<10} ({j['judged_by']}) err={j['corner_err_ratio']}")  # 진행 출력
    report_dir.mkdir(parents=True, exist_ok=True)  # 보고서 폴더 생성
    keys = sorted({k for r in results for k in r}, key=lambda k: list(results[0]).index(k) if k in results[0] else 99)  # 모든 열 이름
    with (report_dir / "evaluation_results.csv").open("w", encoding="utf-8-sig", newline="") as f:  # CSV 저장
        w = csv.DictWriter(f, fieldnames=keys)  # 작성기
        w.writeheader()  # 머리줄
        w.writerows(results)  # 모든 줄
    _contact_sheet(sheet, report_dir / "contact_sheet.jpg")  # 모아보기 이미지 저장
    by = lambda col: {k: [r for r in results if (r.get(col) or "") == k] for k in {r.get(col) or "" for r in results}}  # 열 기준 그룹 함수
    md = ["# 🟩 평가 결과 (자동 생성)", "", f"- 파라미터: `{params.to_dict()}`",  # 문서 머리
          f"- 성공 기준: 4꼭짓점 최대 오차 ≤ 대각선의 {CORNER_TOL * 100:.0f}% (정답 좌표가 있을 때 자동 판정)",  # 기준 1
          f"- 부분 성공: 꼭짓점은 맞지만 종횡비가 실제 대비 {ASPECT_TOL * 100:.0f}% 초과 차이", "- 미검증(detected): 정답 좌표/수동 판정이 없어 '검출됨'만 확인된 상태",  # 기준 2, 3
          "", "## 🟢 1. 조건(category)별 성공률", "", _rate_table(by("category"), "조건", CATEGORY_LABELS),  # 카테고리 표
          "", "## 🟢 2. 조명별 성공률", "", _rate_table(by("lighting"), "조명"),  # 조명 표
          "", "## 🟢 3. 각도별 성공률", "", _rate_table(by("angle"), "각도"),  # 각도 표
          "", "## 🟢 4. 배경별 성공률", "", _rate_table(by("background"), "배경"),  # 배경 표
          "", "## 🟢 5. 이미지별 결과", "", "| 파일 | 조건 | 상태 | 판정 | 꼭짓점 오차(대각선 비) | 종횡비 오차 | 큰 윤곽선 수 |", "| --- | --- | --- | --- | --- | --- | --- |"]  # 이미지별 표 머리
    for r in results:  # 이미지마다
        md.append(f"| {r['filename']} | {r.get('category', '')} | {r['status']} | {r['judged_by']} | {r['corner_err_ratio']} | {r['aspect_err']} | {r.get('contours_big', '')} |")  # 한 줄
    md += ["", "## 🟢 6. 모아보기", "", "![contact sheet](contact_sheet.jpg)", ""]  # 모아보기 이미지 링크
    (report_dir / "evaluation_summary.md").write_text("\n".join(md), encoding="utf-8")  # 마크다운 저장
    print(f"[evaluate] 저장: {report_dir / 'evaluation_summary.md'}")  # 완료 안내
    return results  # 결과 반환


SWEEP_BLUR = [1, 3, 5, 7, 9, 11, 15]  # 스윕할 블러 커널 크기
SWEEP_LOW = [10, 25, 50, 75, 100, 150]  # 스윕할 Canny low 값
SWEEP_RATIO = [2, 3]  # high = low × 비율 (Canny 권장 비율 2~3)


def sweep(base: ScanParams, folder: Path = IMAGES_DIR, report_dir: Path = GENERATED_DIR) -> None:  # 파라미터 딜레마 분석
    from .plotting import plt  # 그래프 도구 (필요할 때만 불러옴)
    records = load_records(folder, folder / "metadata.csv")  # 이미지 목록
    if not records:  # 없으면
        print("[sweep] images/ 폴더에 이미지가 없습니다.")  # 안내
        return  # 종료
    combos = list(itertools.product(SWEEP_BLUR, SWEEP_LOW, SWEEP_RATIO))  # 모든 파라미터 조합
    images = [(rec, load_image(rec.path)) for rec in records]  # 이미지를 미리 한 번만 읽음
    uses_gt = all(rec.gt_corners is not None for rec in records)  # 모든 이미지에 정답이 있는지
    ok = np.zeros((len(records), len(combos)), dtype=np.uint8)  # 성공 행렬 (이미지 × 조합)
    print(f"[sweep] {len(records)}장 × {len(combos)}조합, 판정={'정답 좌표' if uses_gt else '검출 여부(정답 좌표 없음)'}")  # 진행 안내
    for j, (blur, low, ratio) in enumerate(combos):  # 조합마다
        p = ScanParams(**base.to_dict())  # 기본 설정 복사
        p.blur_kernel, p.canny_low, p.canny_high = blur, low, low * ratio  # 조합 값 적용
        for i, (rec, img) in enumerate(images):  # 이미지마다
            res = run_pipeline(img, p)  # 파이프라인 실행
            status = judge(rec, res)["status"]  # 판정
            ok[i, j] = status in ("success", "partial") if uses_gt else res.found  # 성공 여부 기록
    labels = [f"b{b}/l{l}/h{l * r}" for b, l, r in combos]  # 조합 이름
    report_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    with (report_dir / "sweep_results.csv").open("w", encoding="utf-8-sig", newline="") as f:  # 행렬 CSV 저장
        w = csv.writer(f)  # 작성기
        w.writerow(["filename", *labels])  # 머리줄
        for rec, row in zip(records, ok):  # 이미지마다
            w.writerow([rec.name, *row.tolist()])  # 한 줄
    per_combo = ok.sum(axis=0).astype(np.int64)  # 조합별 성공 장수 (부호 있는 정수로 바꿔야 음수 정렬이 깨지지 않음)
    order = np.argsort(-per_combo)  # 성공 많은 순서
    n = len(records)  # 전체 장수
    md = ["# 🟩 파라미터 스윕 결과 (자동 생성)", "", f"- 판정 방식: {'정답 좌표 기반 (부분 성공 포함)' if uses_gt else '⚠️ 검출 여부만 (정답 좌표를 넣으면 정확해짐)'}",  # 머리
          f"- 조합 수: {len(combos)} (blur {SWEEP_BLUR} × low {SWEEP_LOW} × high=low×{SWEEP_RATIO})", "",  # 조합 설명
          "## 🟢 1. 모든 이미지에 통하는 단일 파라미터가 있는가?", "",  # 질문 1
          f"- 최고 조합 `{labels[order[0]]}` 성공 {per_combo[order[0]]}/{n}장 ({per_combo[order[0]] / n * 100:.0f}%)",  # 최고 조합
          f"- 모든 이미지 성공 조합 수: **{int((per_combo == n).sum())}개**",  # 만능 조합 개수
          f"- 어떤 조합으로도 성공 못한 이미지: {', '.join(r.name for r, row in zip(records, ok) if row.sum() == 0) or '없음'}", "",  # 불가능 이미지
          "## 🟢 2. 성공 상위 10개 조합", "", "| 순위 | 조합 | 성공 장수 | 성공률 |", "| --- | --- | --- | --- |"]  # 상위 표 머리
    for rank, j in enumerate(order[:10], 1):  # 상위 10개
        md.append(f"| {rank} | `{labels[j]}` | {per_combo[j]} | {per_combo[j] / n * 100:.0f}% |")  # 한 줄
    md += ["", "## 🟢 3. 이미지별 성공 가능한 조합 수", "", "| 파일 | 조건 | 성공 조합 수 | 예시 성공 조합 |", "| --- | --- | --- | --- |"]  # 이미지별 표 머리
    for rec, row in zip(records, ok):  # 이미지마다
        good = [labels[j] for j in np.flatnonzero(row)][:3]  # 성공 조합 예시 3개
        md.append(f"| {rec.name} | {rec.category} | {int(row.sum())}/{len(combos)} | {', '.join(good) or '-'} |")  # 한 줄
    md += ["", "## 🟢 4. 이미지 A의 최적값이 이미지 B에서 실패하는 사례", ""]  # 충돌 사례 제목
    conflicts = 0  # 사례 개수
    for a, row_a in enumerate(ok):  # 이미지 A마다
        if row_a.sum() == 0:  # A가 어떤 조합으로도 안 되면
            continue  # 건너뜀
        best_a = int(np.argmax(np.where(row_a == 1, per_combo, -1)))  # A에서 성공한 조합 중 전체 성적이 가장 좋은 것
        fails = [records[b].name for b in range(n) if ok[b, best_a] == 0]  # 그 조합으로 실패하는 이미지들
        if fails and conflicts < 10:  # 실패 이미지가 있으면 최대 10건 기록
            md.append(f"- `{records[a].name}` 최적 `{labels[best_a]}` → 실패: {', '.join(fails)}")  # 한 줄
            conflicts += 1  # 개수 증가
    if conflicts == 0:  # 충돌이 없으면
        md.append("- 충돌 사례 없음")  # 안내
    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 0.18), max(4, n * 0.35)))  # 그림 크기
    ax.imshow(ok, cmap="Greens", aspect="auto", vmin=0, vmax=1)  # 성공 행렬을 초록색 지도로
    ax.set_yticks(range(n), [r.name for r in records], fontsize=7)  # 세로축: 이미지 이름
    ax.set_xticks(range(len(combos)), labels, rotation=90, fontsize=5)  # 가로축: 조합 이름
    ax.set_title("파라미터 조합별 성공 여부 (초록=성공)")  # 제목
    fig.tight_layout()  # 여백 정리
    fig.savefig(report_dir / "sweep_matrix.png", dpi=150)  # 저장
    plt.close(fig)  # 메모리 해제
    md += ["", "## 🟢 5. 성공 행렬", "", "![sweep matrix](sweep_matrix.png)", ""]  # 그림 링크
    (report_dir / "sweep_summary.md").write_text("\n".join(md), encoding="utf-8")  # 마크다운 저장
    print(f"[sweep] 저장: {report_dir / 'sweep_summary.md'}")  # 완료 안내
