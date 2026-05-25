#!/usr/bin/env python3
"""Generate synthetic trial-level data for many subliminal-learning tasks."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


def make_task_data(task, n_subjects=30, trials=120, seed=None, out_dir="data"):
    rng = np.random.RandomState(seed)
    rows = []
    for subj in range(1, n_subjects + 1):
        conditions = np.tile(["control", "subliminal"], int(np.ceil(trials / 2)))[:trials]
        rng.shuffle(conditions)
        for t in range(trials):
            cond = conditions[t]
            base_acc = task.get("base_acc", 0.75)
            base_rt = task.get("base_rt", 600)
            effect = task.get("default_effect", 0.05)

            acc_prob = base_acc + (effect if cond == "subliminal" else 0.0)
            acc = rng.binomial(1, np.clip(acc_prob, 0.01, 0.99))
            rt = rng.normal(loc=base_rt - (15 if cond == "subliminal" else 0), scale=60)
            rt = max(150, rt)
            rows.append({
                "subject": subj,
                "trial": t + 1,
                "task_id": task.get("id"),
                "task": task.get("name"),
                "condition": cond,
                "accuracy": int(acc),
                "rt": float(rt),
            })

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{task.get('name')}.csv"
    df.to_csv(fname, index=False)
    print(f"Wrote {fname} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=30)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--out", dest="out_dir", default="data")
    parser.add_argument("--tasks-file", default="tasks_config.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.tasks_file, "r") as f:
        tasks = json.load(f)

    for task in tasks:
        make_task_data(task, n_subjects=args.n_subjects, trials=args.trials, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
