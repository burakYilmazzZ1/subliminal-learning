#!/usr/bin/env python3
"""HF tabanlı görev kataloğunu çalıştır, distilasyon sonuçlarını topla ve raporla."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from experiments.common import run_task_experiment
except ImportError:  
    from common import run_task_experiment


def load_tasks(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_output_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)


def summarize_tasks(tasks, seed: int = 42):
    results = []
    for task in tasks:
        result = run_task_experiment(task, seed=seed)
        results.append(
            {
                "task_id": task.get("id"),
                "task": result.task,
                "turkish_name": task.get("turkish_name", task.get("name")),
                "turkish_description": task.get("turkish_description", ""),
                "family": task.get("family", "bilinmeyen"),
                "angle_tr": task.get("angle_tr", "genel"),
                "dataset_name": task.get("dataset_name", ""),
                "teacher_model": task.get("teacher_model", ""),
                "student_model": task.get("student_model", ""),
                "accuracy": result.accuracy,
                "f1": result.f1,
                "teacher_acc": result.teacher_acc,
                "teacher_f1": result.teacher_f1,
                "agreement": result.agreement,
                "kl_divergence": result.kl_divergence,
                "teacher_ece": result.teacher_ece,
                "student_ece": result.student_ece,
                "gain": result.accuracy - result.teacher_acc,
                "learned_flag": result.learned_flag,
                "notes": result.notes,
            }
        )
    return pd.DataFrame(results)


def add_family_summary(summary_df):
    numeric_cols = [
        "accuracy",
        "f1",
        "teacher_acc",
        "teacher_f1",
        "agreement",
        "kl_divergence",
        "teacher_ece",
        "student_ece",
        "gain",
    ]
    family_df = summary_df.groupby("family")[numeric_cols].mean(numeric_only=True).reset_index()
    family_df["task_sayisi"] = summary_df.groupby("family").size().values
    return family_df


def add_angle_summary(summary_df):
    numeric_cols = [
        "accuracy",
        "f1",
        "teacher_acc",
        "teacher_f1",
        "agreement",
        "kl_divergence",
        "teacher_ece",
        "student_ece",
        "gain",
    ]
    angle_df = summary_df.groupby("angle_tr")[numeric_cols].mean(numeric_only=True).reset_index()
    angle_df["task_sayisi"] = summary_df.groupby("angle_tr").size().values
    return angle_df


def write_report(summary_df, family_df, out_path):
    top_tasks = summary_df.sort_values("accuracy", ascending=False).head(3)
    best_gain = summary_df.sort_values("gain", ascending=False).head(3)
    top_family = family_df.sort_values("accuracy", ascending=False).head(1)

    lines = [
        "# HF Tabanlı Subliminal Öğrenme Raporu",
        "",
        "## Kısa yorum",
        "Bu rapor her görevi ayrı Hugging Face veri kümesi ve teacher-student model çifti ile değerlendirir.",
        "",
        "## En yüksek öğrenci doğruluğu",
    ]
    for _, row in top_tasks.iterrows():
        lines.append(
            f"- {row['turkish_name']} ({row['task']}): student_acc={row['accuracy']:.3f}, teacher_acc={row['teacher_acc']:.3f}, gain={row['gain']:.3f}"
        )

    lines.extend(["", "## En yüksek aktarım kazancı"])
    for _, row in best_gain.iterrows():
        lines.append(
            f"- {row['turkish_name']} ({row['task']}): gain={row['gain']:.3f}, agreement={row['agreement']:.3f}, KL={row['kl_divergence']:.3f}"
        )

    lines.extend(["", "## En güçlü aile"])
    if not top_family.empty:
        row = top_family.iloc[0]
        lines.append(f"- {row['family']}: student_acc={row['accuracy']:.3f}, task_sayisi={int(row['task_sayisi'])}")

    lines.extend([
        "",
        "## Yorumlama notu",
        "- Öğretmen ve öğrenci arasındaki fark, distilasyonun bilgi sıkıştırma etkisini gösterir.",
        "- ECE düşüyorsa modelin güven kalibrasyonu iyileşmiştir.",
        "- Agreement yüksek ama student_acc düşükse öğrenci öğretmeni taklit ediyor ama genellemesi zayıf olabilir.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(summary_df, family_df, out_dir):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.sort_values("accuracy")
    plt.figure(figsize=(11, max(5, 0.35 * len(plot_df))))
    sns.barplot(data=plot_df, x="accuracy", y="turkish_name", hue="family", dodge=False)
    plt.title("Görev bazında öğrenci doğruluğu")
    plt.xlabel("Student accuracy")
    plt.ylabel("")
    plt.legend(title="Aile", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "gorev_student_accuracy.png", dpi=160)
    plt.close()

    family_plot = family_df.sort_values("accuracy")
    plt.figure(figsize=(9, max(4, 0.35 * len(family_plot))))
    sns.barplot(data=family_plot, x="accuracy", y="family", color="#2a9d8f")
    plt.title("Aile bazında student accuracy")
    plt.xlabel("Student accuracy")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(plots_dir / "aile_student_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=summary_df, x="teacher_acc", y="accuracy", hue="family", s=90)
    min_val = min(summary_df["teacher_acc"].min(), summary_df["accuracy"].min())
    max_val = max(summary_df["teacher_acc"].max(), summary_df["accuracy"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    plt.title("Teacher vs Student accuracy")
    plt.xlabel("Teacher")
    plt.ylabel("Student")
    plt.tight_layout()
    plt.savefig(plots_dir / "teacher_student_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary_df, x="agreement", y="accuracy", hue="family", s=100)
    plt.title("Agreement vs Student accuracy")
    plt.xlabel("Agreement")
    plt.ylabel("Student accuracy")
    plt.tight_layout()
    plt.savefig(plots_dir / "agreement_vs_accuracy.png", dpi=160)
    plt.close()

    angle_df = summary_df.groupby("angle_tr", as_index=False)["accuracy"].mean().sort_values("accuracy")
    plt.figure(figsize=(10, max(4, 0.35 * len(angle_df))))
    sns.barplot(data=angle_df, x="accuracy", y="angle_tr", color="#264653")
    plt.title("12 açı bazında student accuracy")
    plt.xlabel("Student accuracy")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(plots_dir / "angle_student_accuracy.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gorev_katalogu.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_output_dirs(out_dir)
    tasks = load_tasks(args.config)

    summary_df = summarize_tasks(tasks, seed=args.seed).sort_values("accuracy", ascending=False)
    summary_df.to_csv(out_dir / "gorev_karsilastirma.csv", index=False)

    family_df = add_family_summary(summary_df)
    family_df.to_csv(out_dir / "aile_karsilastirma.csv", index=False)

    angle_df = add_angle_summary(summary_df)
    angle_df.to_csv(out_dir / "angle_karsilastirma.csv", index=False)

    plot_outputs(summary_df, family_df, out_dir)
    write_report(summary_df, family_df, out_dir / "rapor_tr.md")

    print(f"Yazıldı: {out_dir / 'gorev_karsilastirma.csv'}")
    print(f"Yazıldı: {out_dir / 'aile_karsilastirma.csv'}")
    print(f"Yazıldı: {out_dir / 'angle_karsilastirma.csv'}")
    print(f"Yazıldı: {out_dir / 'rapor_tr.md'}")


if __name__ == "__main__":
    main()
