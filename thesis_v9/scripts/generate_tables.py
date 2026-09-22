#!/usr/bin/env python3
"""Generate LaTeX table fragments from project CSV/JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "tables"

def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")


def tab_phase5() -> None:
    rows = [
        ("شاخه A — بیومارکر (XGBoost)", "0.800", "0.646", "0.762", "0.442"),
        ("شاخه B — CNN (EfficientNet-B5)", "0.928", "0.861", "0.803", "0.590"),
        ("شاخه C — ترکیب زودهنگام (LightGBM)", "0.912", "0.837", "0.785", "0.560"),
        ("شاخه C — فقط embedding (آبلاسیون)", "0.916", "0.799", "0.851", "0.615"),
        ("Legacy leaky hybrid (نامعتبر)", "0.980", "—", "—", "—"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{نتایج قفل‌شده فاز ۵ روی مجموعه آزمون (Plus-OvR)}",
        r"  \label{tab:phase5_locked}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{مدل} & \textbf{AUC} & \textbf{حساسیت} & \textbf{اختصاصیت} & \textbf{F1} \\",
        r"    \midrule",
    ]
    for model, auc, sens, spec, f1 in rows:
        lines.append(f"    {model} & {auc} & {sens} & {spec} & {f1} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    write("tab_phase5.tex", "\n".join(lines))


def tab_split() -> None:
    p = ROOT / "data/splits/split_counts.csv"
    if not p.exists():
        p = ROOT / "THESIS_COMPLETE_DOCUMENTATION/artifacts/tables/split_counts.csv"
    df = pd.read_csv(p)
    if "n" in df.columns:
        agg = df.groupby("split")["n"].sum().reset_index()
    else:
        agg = pd.read_csv(ROOT / "data/splits/split_counts.csv")
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{تعداد تصاویر در تقسیم‌بندی رسمی گروه‌محور}",
        r"  \label{tab:split_counts}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    \textbf{بخش} & \textbf{تصاویر} & \textbf{Normal} & \textbf{Pre\_Plus} & \textbf{Plus} \\",
        r"    \midrule",
    ]
    for _, r in agg.iterrows():
        split = str(r["split"])
        if "images" in r:
            n, n0, n1, n2 = int(r["images"]), int(r["normal"]), int(r["pre_plus"]), int(r["plus"])
        else:
            sub = df[df.split == split]
            n = int(sub["n"].sum())
            n0 = int(sub[sub.label == 0]["n"].sum()) if "label" in sub else 0
            n1 = int(sub[sub.label == 1]["n"].sum()) if "label" in sub else 0
            n2 = int(sub[sub.label == 2]["n"].sum()) if "label" in sub else 0
        lines.append(f"    {split} & {n} & {n0} & {n1} & {n2} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write("tab_split.tex", "\n".join(lines))


def tab_e2_seeds() -> None:
    p = ROOT / "THESIS_COMPLETE_DOCUMENTATION/artifacts/tables/all_runs.csv"
    df = pd.read_csv(p)
    df = df[(df.experiment == "E2") & (df.fold == "official")].sort_values("seed")
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{آزمایش E2 — دقت اعتبارسنجی برای پنج seed (384px، سر Plus دودویی)}",
        r"  \label{tab:e2_seeds}",
        r"  \begin{tabular}{ccccc}",
        r"    \toprule",
        r"    \textbf{Seed} & \textbf{Val AUC} & \textbf{بدترین منبع} & \textbf{Epoch} \\",
        r"    \midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"    {int(r.seed)} & {r.val_auc:.4f} & {r.worst_source_auc:.3f} & {int(r.best_epoch)} \\\\"
        )
    mean = df.val_auc.mean()
    std = df.val_auc.std()
    lines.append(r"    \midrule")
    lines.append(f"    \\textbf{{میانگین $\\pm$ انحراف}} & {mean:.4f} $\\pm$ {std:.4f} & — & — \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write("tab_e2_seeds.tex", "\n".join(lines))


def tab_loso() -> None:
    p = ROOT / "THESIS_COMPLETE_DOCUMENTATION/artifacts/tables/e2_e8_loso.csv"
    df = pd.read_csv(p)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{ارزیابی LOSO — مقایسه E2 و E8 (seed 42)}",
        r"  \label{tab:loso}",
        r"  \small",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Holdout} & \textbf{E2 AUC} & \textbf{E8 AUC} & \textbf{$\Delta$} & \textbf{n} \\",
        r"    \midrule",
    ]
    for _, r in df.iterrows():
        holdout = str(r.holdout).replace("_", r"\_")
        lines.append(
            f"    \\lr{{{holdout}}} & {r.e2_holdout_auc:.3f} & {r.e8_holdout_auc:.3f} & {r.delta_holdout_auc:+.3f} & {int(r.e2_n)} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write("tab_loso.tex", "\n".join(lines))


def tab_e4b() -> None:
    p = ROOT / "THESIS_COMPLETE_DOCUMENTATION/artifacts/tables/e4b_vs_e2_official.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{E4B در برابر E2 — تفاوت AUC اعتبارسنجی}",
        r"  \label{tab:e4b}",
        r"  \begin{tabular}{ccccc}",
        r"    \toprule",
        r"    \textbf{Seed} & \textbf{E2} & \textbf{E4B} & \textbf{$\Delta$} \\",
        r"    \midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"    {int(r.seed)} & {r.e2_val_auc:.4f} & {r.e4b_val_auc:.4f} & {r.delta_e4b_minus_e2:+.4f} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write("tab_e4b.tex", "\n".join(lines))


def tab_integrity() -> None:
    p = ROOT / "THESIS_COMPLETE_DOCUMENTATION/artifacts/json/LOCAL_SYNC_INTEGRITY.json"
    data = json.loads(p.read_text())
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{اثبات یکپارچگی داده و نتایج (بازمحاسبه محلی)}",
        r"  \label{tab:integrity}",
        r"  \begin{tabular}{ll}",
        r"    \toprule",
        r"    \textbf{مورد} & \textbf{مقدار} \\",
        r"    \midrule",
        f"    SHA-256 split & \\lr{{{data['split_sha256'][:16]}...}} \\\\",
        f"    تعداد تصاویر & {data['n_rows']} \\\\",
        f"    تعداد گروه & {data['n_groups']} \\\\",
        f"    A / B / C test AUC & {data['phase5_test_auc']['A']:.3f} / {data['phase5_test_auc']['B']:.3f} / {data['phase5_test_auc']['C']:.3f} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    write("tab_integrity.tex", "\n".join(lines))


if __name__ == "__main__":
    tab_phase5()
    tab_split()
    tab_e2_seeds()
    tab_loso()
    tab_e4b()
    tab_integrity()
    print("Wrote tables to", OUT)
