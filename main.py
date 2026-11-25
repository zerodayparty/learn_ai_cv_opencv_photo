"""문서 스캐너 실행 진입점 (명령줄 인터페이스).

사용 예:
    uv run python main.py                       # 이미지 경로를 물어본 뒤 GUI 실행
    uv run python main.py scan images/a.jpg     # 트랙바 GUI로 실행
    uv run python main.py scan images/a.jpg --no-gui --blur 7 --low 50 --high 150
    uv run python main.py batch                 # 20장 일괄 평가 + 성공률 표
    uv run python main.py sweep                 # 파라미터 딜레마 분석
    uv run python main.py eda                   # 히스토그램 EDA
    uv run python main.py annotate              # 정답 꼭짓점 클릭 입력
    uv run python main.py bench images/a.jpg    # 보너스1 컨볼루션 속도 비교
    uv run python main.py pdf                   # 보너스3 스캔본을 PDF로 묶기
    uv run python main.py generate              # 과제용 합성 테스트 사진 20장 + metadata.csv 생성
"""  # 모듈 설명 (사용법)

from __future__ import annotations  # 타입 힌트 호환성

import argparse  # 명령줄 인자 해석
import sys  # 프로그램 종료 코드
from pathlib import Path  # 경로 처리

from scanner.config import GENERATED_DIR, IMAGES_DIR, OUTPUT_DIR, load_params  # 경로 상수와 설정 로더


def _add_param_args(p: argparse.ArgumentParser) -> None:  # 파라미터 관련 공통 인자 추가
    p.add_argument("--config", help="파라미터 JSON 파일 경로 (예: config.example.json)")  # 설정 파일
    p.add_argument("--blur", type=int, dest="blur_kernel", help="가우시안 블러 커널 크기 (홀수)")  # 블러 커널
    p.add_argument("--low", type=int, dest="canny_low", help="Canny 낮은 임계값")  # Canny low
    p.add_argument("--high", type=int, dest="canny_high", help="Canny 높은 임계값")  # Canny high
    p.add_argument("--dilate", type=int, dest="dilate_iter", help="엣지 팽창 반복 횟수")  # 팽창 횟수
    p.add_argument("--min-area", type=float, dest="min_area_ratio", help="문서 최소 면적 비율 (0~1)")  # 최소 면적
    p.add_argument("--block", type=int, dest="thresh_block", help="적응형 이진화 블록 크기 (홀수)")  # 이진화 블록
    p.add_argument("--c", type=int, dest="thresh_c", help="적응형 이진화 상수 C")  # 이진화 상수
    p.add_argument("--orientation", choices=["portrait", "landscape", "auto"], help="결과 방향 (기본 portrait)")  # 결과 방향


def _params_from(args: argparse.Namespace):  # 인자에서 ScanParams 생성
    keys = ["blur_kernel", "canny_low", "canny_high", "dilate_iter", "min_area_ratio", "thresh_block", "thresh_c", "orientation"]  # 파라미터 이름 목록
    return load_params(args.config, **{k: getattr(args, k, None) for k in keys})  # 파일 + 명령줄 값 합치기


def cmd_scan(args: argparse.Namespace) -> int:  # scan 명령: 이미지 한 장 스캔
    from scanner.pipeline import load_image, run_pipeline, save_image  # 파이프라인 함수
    path = args.image  # 명령줄에서 받은 경로
    if not path:  # 경로가 없으면
        try:  # 입력이 끊길 수 있으므로 보호
            path = input("이미지 경로를 입력하세요: ").strip().strip("'\"")  # 직접 입력받음 (드래그 시 붙는 따옴표 제거)
        except EOFError:  # 입력 스트림이 없으면 (파이프/자동 실행)
            path = ""  # 빈 경로로 처리
    if not path or not Path(path).exists():  # 경로가 비었거나 파일이 없으면
        print(f"[오류] 파일이 없습니다: {path}")  # 오류 안내
        return 1  # 실패 코드
    image = load_image(path)  # 이미지 읽기
    params = _params_from(args)  # 파라미터 준비
    out_dir = Path(args.out)  # 저장 폴더
    stem = Path(path).stem  # 확장자 뺀 파일 이름
    if args.no_gui:  # GUI 없이 실행하면 (Colab/서버용)
        res = run_pipeline(image, params)  # 파이프라인 1회 실행
        for tag, img in (("1_original", res.resized), ("2_preprocessed", res.blurred), ("3_edges", res.edges), ("4_contours", res.contour_vis)):  # 단계별 이미지
            save_image(out_dir / f"{stem}_{tag}.jpg", img)  # 단계별로 파일 저장 (imshow 대체)
        if not res.found:  # 못 찾았으면
            print(f"[실패] 문서를 찾지 못했습니다. info={res.info}")  # 실패 안내
            return 2  # 실패 코드
        save_image(out_dir / f"{stem}_5_warped.jpg", res.warped)  # 컬러 변환본 저장
        out = save_image(out_dir / f"scanned_{stem}.png", res.scanned)  # 최종 스캔본 저장
        if args.pdf:  # PDF 옵션이면
            from scanner.pdf_export import export_pdf  # PDF 모듈
            export_pdf([res.scanned], out_dir / f"scanned_{stem}.pdf")  # PDF 저장
        print(f"[완료] {out} corners={res.corners.round(1).tolist()}")  # 결과 안내
        return 0  # 성공 코드
    from scanner.gui import run_gui  # GUI 모듈 (화면이 있을 때만 불러옴)
    run_gui(image, params, stem, out_dir)  # 트랙바 GUI 실행
    return 0  # 성공 코드


def cmd_batch(args: argparse.Namespace) -> int:  # batch 명령: 일괄 평가
    from scanner.evaluate import evaluate  # 평가 함수
    results = evaluate(_params_from(args), Path(args.images), Path(args.out) / "batch", Path(args.report))  # 실행
    return 0 if results else 1  # 결과가 있으면 성공


def cmd_sweep(args: argparse.Namespace) -> int:  # sweep 명령: 파라미터 스윕
    from scanner.evaluate import sweep  # 스윕 함수
    sweep(_params_from(args), Path(args.images), Path(args.report))  # 실행
    return 0  # 성공 코드


def cmd_eda(args: argparse.Namespace) -> int:  # eda 명령: 탐색적 분석
    from scanner.eda import run_eda  # EDA 함수
    run_eda(Path(args.images), Path(args.report))  # 실행
    return 0  # 성공 코드


def cmd_annotate(args: argparse.Namespace) -> int:  # annotate 명령: 정답 좌표 입력
    from scanner.annotate import annotate  # 입력 도구
    annotate(Path(args.images), only_missing=not args.all)  # 실행
    return 0  # 성공 코드


def cmd_init(args: argparse.Namespace) -> int:  # init 명령: metadata.csv 생성/동기화
    from scanner.dataset import sync_metadata  # 동기화 함수
    folder = Path(args.images)  # 이미지 폴더
    rows = sync_metadata(folder, folder / "metadata.csv")  # 폴더의 이미지를 CSV에 등록
    print(f"[init] {folder / 'metadata.csv'} 에 {len(rows)}장 등록")  # 결과 안내
    return 0  # 성공 코드


def cmd_bench(args: argparse.Namespace) -> int:  # bench 명령: 보너스1 속도 비교
    import cv2  # 그레이 변환
    from scanner.convolution import benchmark  # 벤치마크 함수
    from scanner.pipeline import load_image, resize_for_processing  # 이미지 읽기/축소
    small, _ = resize_for_processing(load_image(args.image), args.height)  # 적당한 크기로 축소
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # 그레이 변환
    print(benchmark(gray, args.ksize, Path(args.report) / "bench_convolution.md", args.loop_size))  # 실행 후 결과 출력
    return 0  # 성공 코드


def cmd_pdf(args: argparse.Namespace) -> int:  # pdf 명령: 보너스3 스캔본 PDF 묶기
    from scanner.pdf_export import export_pdf  # PDF 함수
    from scanner.pipeline import load_image  # 이미지 읽기
    src = Path(args.src)  # 스캔본 폴더
    files = sorted(src.glob("scanned_*.png"))  # 최종 스캔본 파일 목록
    if not files:  # 없으면
        print(f"[pdf] {src} 에 scanned_*.png 가 없습니다. 먼저 batch를 실행하세요.")  # 안내
        return 1  # 실패 코드
    out = export_pdf([load_image(f) for f in files], Path(args.output))  # PDF 저장
    print(f"[pdf] {len(files)}페이지 → {out}")  # 결과 안내
    return 0  # 성공 코드


def cmd_sample(args: argparse.Namespace) -> int:  # sample 명령: 합성 테스트 이미지 생성
    from scanner.synthetic import make_samples  # 생성 함수
    paths = make_samples(Path(args.out))  # 생성
    print(f"[sample] {len(paths)}장 생성 → {args.out}")  # 결과 안내
    return 0  # 성공 코드


def cmd_generate(args: argparse.Namespace) -> int:  # generate 명령: 과제용 합성 사진 20장 생성
    from scanner.photo_factory import generate_dataset  # 생성 함수
    try:  # 기존 사진 보호 오류를 잡기 위해
        paths = generate_dataset(Path(args.images), seed=args.seed, force=args.force)  # 20장 + metadata.csv 생성
    except FileExistsError as e:  # 이미 사진이 있으면
        print(f"[generate] {e}")  # 안내
        return 1  # 실패 코드
    print(f"[generate] {len(paths)}장 + metadata.csv 생성 → {args.images}")  # 결과 안내
    return 0  # 성공 코드


def build_parser() -> argparse.ArgumentParser:  # 명령줄 구조 정의
    parser = argparse.ArgumentParser(description="OpenCV 규칙 기반 문서 스캐너")  # 최상위 파서
    sub = parser.add_subparsers(dest="command")  # 하위 명령 묶음
    s = sub.add_parser("scan", help="이미지 한 장 스캔 (기본: 트랙바 GUI)")  # scan 명령
    s.add_argument("image", nargs="?", help="이미지 경로 (생략 시 입력받음)")  # 이미지 경로
    s.add_argument("--no-gui", action="store_true", help="GUI 없이 단계별 이미지를 파일로 저장")  # GUI 끄기
    s.add_argument("--pdf", action="store_true", help="결과를 PDF로도 저장 (보너스3)")  # PDF 저장
    s.add_argument("--out", default=str(OUTPUT_DIR), help="저장 폴더")  # 저장 폴더
    _add_param_args(s)  # 파라미터 인자
    s.set_defaults(func=cmd_scan)  # 실행 함수 연결
    for name, func, help_ in (("batch", cmd_batch, "일괄 평가 + 조건별 성공률"), ("sweep", cmd_sweep, "파라미터 스윕(딜레마 분석)"), ("eda", cmd_eda, "히스토그램 EDA")):  # 폴더 단위 명령 3개
        c = sub.add_parser(name, help=help_)  # 하위 명령 생성
        c.add_argument("--images", default=str(IMAGES_DIR), help="테스트 이미지 폴더")  # 이미지 폴더
        c.add_argument("--report", default=str(GENERATED_DIR), help="자동 생성 보고서 폴더")  # 보고서 폴더
        c.add_argument("--out", default=str(OUTPUT_DIR), help="결과 이미지 폴더")  # 출력 폴더
        _add_param_args(c)  # 파라미터 인자
        c.set_defaults(func=func)  # 실행 함수 연결
    a = sub.add_parser("annotate", help="정답 꼭짓점 클릭 입력")  # annotate 명령
    a.add_argument("--images", default=str(IMAGES_DIR), help="테스트 이미지 폴더")  # 이미지 폴더
    a.add_argument("--all", action="store_true", help="이미 입력된 이미지도 다시 입력")  # 전체 재입력
    a.set_defaults(func=cmd_annotate)  # 실행 함수 연결
    i = sub.add_parser("init", help="images/metadata.csv 생성/동기화")  # init 명령
    i.add_argument("--images", default=str(IMAGES_DIR), help="테스트 이미지 폴더")  # 이미지 폴더
    i.set_defaults(func=cmd_init)  # 실행 함수 연결
    b = sub.add_parser("bench", help="보너스1: NumPy 컨볼루션 속도 비교")  # bench 명령
    b.add_argument("image", help="테스트 이미지 경로")  # 이미지 경로
    b.add_argument("--ksize", type=int, default=5, help="커널 크기 (홀수)")  # 커널 크기
    b.add_argument("--height", type=int, default=600, help="축소 높이")  # 축소 높이
    b.add_argument("--loop-size", type=int, default=120, help="이중 for문 비교 조각 크기")  # 조각 크기
    b.add_argument("--report", default=str(GENERATED_DIR), help="보고서 폴더")  # 보고서 폴더
    b.set_defaults(func=cmd_bench)  # 실행 함수 연결
    pd = sub.add_parser("pdf", help="보너스3: 스캔본을 하나의 PDF로")  # pdf 명령
    pd.add_argument("--src", default=str(OUTPUT_DIR / "batch"), help="scanned_*.png 폴더")  # 입력 폴더
    pd.add_argument("--output", default=str(OUTPUT_DIR / "scanned_all.pdf"), help="PDF 저장 경로")  # 출력 파일
    pd.set_defaults(func=cmd_pdf)  # 실행 함수 연결
    sm = sub.add_parser("sample", help="동작 확인용 합성 이미지 생성")  # sample 명령
    sm.add_argument("--out", default=str(OUTPUT_DIR / "synthetic"), help="생성 폴더")  # 생성 폴더
    sm.set_defaults(func=cmd_sample)  # 실행 함수 연결
    g = sub.add_parser("generate", help="과제용 합성 테스트 사진 20장 + 정답 꼭짓점 생성")  # generate 명령
    g.add_argument("--images", default=str(IMAGES_DIR), help="생성 폴더")  # 생성 폴더
    g.add_argument("--seed", type=int, default=2026, help="난수 seed (같으면 같은 사진)")  # 재현용 seed
    g.add_argument("--force", action="store_true", help="폴더에 이미지가 있어도 덮어쓰기")  # 덮어쓰기 허용
    g.set_defaults(func=cmd_generate)  # 실행 함수 연결
    return parser  # 완성된 파서 반환


def main(argv: list[str] | None = None) -> int:  # 프로그램 시작 함수
    parser = build_parser()  # 파서 생성
    args = parser.parse_args(argv)  # 인자 해석
    if args.command is None:  # 하위 명령 없이 실행하면
        args = parser.parse_args(["scan", *(argv or [])])  # 기본으로 scan(경로 입력받기) 실행
    return args.func(args)  # 선택된 명령 실행


if __name__ == "__main__":  # 직접 실행했을 때만
    sys.exit(main())  # main 실행 결과를 종료 코드로 사용
