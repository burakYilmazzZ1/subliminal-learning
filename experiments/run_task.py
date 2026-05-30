#!/usr/bin/env python3
"""Run a single task environment from its local config.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import run_experiment, write_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True, help="Path to a task folder under experiments/tasks/")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    config_path = task_dir / "config.json"
    result = run_experiment(config_path=config_path, data_dir=Path(args.data_dir))
    out_path = write_result(result, Path(args.output_dir))
    print(f"{result.task}: {result.architecture_name} -> accuracy={result.accuracy:.3f}, f1={result.f1:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
