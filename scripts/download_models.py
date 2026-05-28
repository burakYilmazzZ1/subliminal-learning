"""Download teacher/student models referenced by experiment task configs.

Scans `experiments/tasks/*/config.json` and downloads tokenizers and
weights to a specified cache directory (default: `.hf_cache`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def find_task_configs(root: Path) -> Set[Path]:
    configs = set()
    tasks_dir = root / "experiments" / "tasks"
    if not tasks_dir.exists():
        return configs
    for child in tasks_dir.iterdir():
        cfg = child / "config.json"
        if cfg.exists():
            configs.add(cfg)
    return configs


def collect_model_ids(config_paths: Set[Path], which: str = "all") -> Set[str]:
    models = set()
    for cfg in config_paths:
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        if which in ("all", "teacher"):
            tm = data.get("teacher_model") or data.get("teacher_tokenizer")
            if tm:
                models.add(tm)
        if which in ("all", "student"):
            sm = data.get("student_model")
            if sm:
                models.add(sm)
    return models


def collect_from_catalog(root: Path, which: str = "all") -> Set[str]:
    catalog_path = root / "configs" / "gorev_katalogu.json"
    models = set()
    if not catalog_path.exists():
        return models
    try:
        entries = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return models
    for entry in entries:
        if which in ("all", "teacher"):
            tm = entry.get("teacher_model") or entry.get("teacher_tokenizer")
            if tm:
                models.add(tm)
        if which in ("all", "student"):
            sm = entry.get("student_model")
            if sm:
                models.add(sm)
    return models


def download_models(model_ids: Set[str], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for mid in sorted(model_ids):
        print(f"Downloading tokenizer for: {mid}")
        try:
            AutoTokenizer.from_pretrained(mid, cache_dir=str(cache_dir))
        except Exception as exc:
            print(f"  Failed tokenizer {mid}: {exc}")
            continue
        print(f"Downloading model weights for: {mid}")
        try:
            AutoModelForSequenceClassification.from_pretrained(mid, cache_dir=str(cache_dir))
        except Exception as exc:
            print(f"  Failed model {mid}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download HF models referenced by task configs")
    p.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"), help="cache directory for downloaded models")
    p.add_argument("--which", choices=["all", "teacher", "student"], default="all", help="which models to download")
    p.add_argument("--dry-run", action="store_true", help="just list models to download")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs = find_task_configs(repo_root)
    catalog_models = collect_from_catalog(repo_root, which=args.which)
    configs_models = collect_model_ids(configs, which=args.which)
    models = catalog_models.union(configs_models)
    if not models:
        print("No models found in configs for the selected option")
        return
    if not models:
        print("No models found in configs for the selected option")
        return
    print(f"Found {len(models)} unique model ids")
    for m in sorted(models):
        print(" -", m)
    if args.dry_run:
        return
    download_models(models, args.cache_dir)


if __name__ == "__main__":
    main()
