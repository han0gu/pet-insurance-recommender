"""
혼동 행렬(Confusion Matrix) 계산 및 결과 리포트 저장 모듈.

평가 파이프라인 실행 후:
  1) TP, TN, FP, FN 개수를 계산하여 rich 테이블로 터미널 출력
  2) FP(보장 안 되는데 된다고 판단한 위험 케이스)를 빨간색으로 강조
  3) Pandas DataFrame으로 변환하여 CSV 파일로 자동 저장
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from rich import print as rprint
from rich.table import Table

from app.evaluation.schemas import EvaluationRecord

# 결과 CSV 저장 디렉토리
RESULTS_DIR = Path(__file__).parent / "results"


# ==========================================
# 1. 혼동 행렬 계산 + Rich 테이블 출력
# ==========================================


def compute_and_display_metrics(records: list[EvaluationRecord]) -> dict[str, int]:
    """평가 결과에서 혼동 행렬을 계산하고 Rich 테이블로 출력합니다.

    혼동 행렬 정의 (Positive = 보장됨, Negative = 보장 안 됨):
      - TP: Judge=보장 & Evaluator=보장 → 정확한 보장 판정
      - TN: Judge=면책 & Evaluator=면책 → 정확한 면책 판정
      - FP: Judge=보장 & Evaluator=면책 → ⚠️ 위험! 실제 면책인데 보장이라 함
      - FN: Judge=면책 & Evaluator=보장 → 보수적 판단 (실제 보장인데 면책이라 함)

    Args:
        records: 평가 결과 레코드 리스트

    Returns:
        {"TP": n, "TN": n, "FP": n, "FN": n} 딕셔너리
    """
    # 라벨별 개수 집계
    counts: dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1

    total = len(records)

    # ── 혼동 행렬 테이블 ──
    matrix_table = Table(
        title="🔍 혼동 행렬 (Confusion Matrix)",
        show_header=True,
        header_style="bold magenta",
    )
    matrix_table.add_column("", style="bold", width=25)
    matrix_table.add_column("Evaluator: 보장(P)", justify="center", width=20)
    matrix_table.add_column("Evaluator: 면책(N)", justify="center", width=20)

    # Judge가 보장이라 한 행
    matrix_table.add_row(
        "Judge: 보장(P)",
        f"[green]TP = {counts['TP']}[/green]",
        f"[bold red]FP = {counts['FP']}[/bold red]",  # 위험 강조
    )
    # Judge가 면책이라 한 행
    matrix_table.add_row(
        "Judge: 면책(N)",
        f"[yellow]FN = {counts['FN']}[/yellow]",
        f"[blue]TN = {counts['TN']}[/blue]",
    )

    rprint()
    rprint(matrix_table)

    # ── 성능 지표 테이블 ──
    accuracy = (counts["TP"] + counts["TN"]) / total if total > 0 else 0
    precision = (
        counts["TP"] / (counts["TP"] + counts["FP"])
        if (counts["TP"] + counts["FP"]) > 0
        else 0
    )
    recall = (
        counts["TP"] / (counts["TP"] + counts["FN"])
        if (counts["TP"] + counts["FN"]) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    metrics_table = Table(
        title="📊 성능 지표",
        show_header=True,
        header_style="bold cyan",
    )
    metrics_table.add_column("지표", style="bold", width=15)
    metrics_table.add_column("값", justify="center", width=15)

    metrics_table.add_row("총 테스트 수", str(total))
    metrics_table.add_row("Accuracy", f"{accuracy:.2%}")
    metrics_table.add_row("Precision", f"{precision:.2%}")
    metrics_table.add_row("Recall", f"{recall:.2%}")
    metrics_table.add_row("F1 Score", f"{f1:.2%}")

    rprint()
    rprint(metrics_table)

    # ── FP 위험 케이스 상세 출력 ──
    fp_records = [r for r in records if r.label == "FP"]
    if fp_records:
        rprint()
        rprint(
            f"[bold red]⚠️  FP(위험 케이스) {len(fp_records)}건 상세 "
            f"— 보장 안 되는데 보장된다고 판단한 건[/bold red]"
        )

        fp_table = Table(
            title="⚠️ FP (False Positive) 상세",
            show_header=True,
            header_style="bold red",
        )
        fp_table.add_column("파일명", width=12)
        fp_table.add_column("품종", width=12)
        fp_table.add_column("질병명", width=20)
        fp_table.add_column("Judge 이유", width=40)
        fp_table.add_column("Evaluator 이유", width=40)

        for r in fp_records:
            fp_table.add_row(
                r.test_case.file_name,
                r.test_case.breed,
                r.test_case.disease_name,
                r.judge_prediction.reason[:80],
                r.evaluator_ground_truth.reason[:80],
            )

        rprint(fp_table)

    return counts


# ==========================================
# 2. CSV 저장
# ==========================================


def records_to_dataframe(records: list[EvaluationRecord]) -> pd.DataFrame:
    """평가 레코드 리스트를 Pandas DataFrame으로 변환합니다.

    CSV 컬럼:
      파일이름, 견/묘종, 나이, 기저질환(history), 추출질병명, 약관원문,
      Judge예측(O/X), Judge이유, Evaluator정답(O/X), Evaluator이유, 라벨
    """
    rows: list[dict[str, str | int]] = []

    for record in records:
        tc = record.test_case
        jp = record.judge_prediction
        eg = record.evaluator_ground_truth

        rows.append(
            {
                "파일이름": tc.file_name,
                "견묘종": tc.breed,
                "나이": tc.age,
                "기저질환": tc.disease_surgery_history,
                "추출질병명": tc.disease_name,
                "약관원문": tc.policy_text[:200],  # CSV 가독성을 위해 200자 제한
                "Judge예측": "O" if jp.is_covered else "X",
                "Judge이유": jp.reason,
                "Evaluator정답": "O" if eg.is_covered else "X",
                "Evaluator이유": eg.reason,
                "라벨": record.label,
            }
        )

    return pd.DataFrame(rows)


def save_results_to_csv(records: list[EvaluationRecord]) -> Path:
    """평가 결과를 타임스탬프가 포함된 CSV 파일로 저장합니다.

    저장 경로: app/evaluation/results/eval_result_{YYYYMMDDHHmmss}.csv

    Args:
        records: 평가 결과 레코드 리스트

    Returns:
        저장된 CSV 파일의 Path 객체
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_path = RESULTS_DIR / f"eval_result_{timestamp}.csv"

    df = records_to_dataframe(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    rprint(f"\n[bold green]📁 결과 저장 완료: {csv_path}[/bold green]")
    rprint(f"   총 {len(records)}건, 컬럼: {list(df.columns)}")

    return csv_path
