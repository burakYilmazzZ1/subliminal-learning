"""Common utilities for task-specific experiment environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from scripts.hf_distillation import DistillationMetrics, run_task_distillation


@dataclass(frozen=True)
class ExperimentResult:
    task: str
    architecture_key: str
    architecture_name: str
    architecture_description: str
    accuracy: float
    f1: float
    teacher_acc: float
    teacher_f1: float
    agreement: float
    kl_divergence: float
    teacher_ece: float
    student_ece: float
    learned_flag: bool
    notes: str


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_catalog() -> Dict[str, Dict[str, Any]]:
    catalog_path = Path("configs/gorev_katalogu.json")
    tasks = load_json(catalog_path)
    return {task["name"]: task for task in tasks}


def read_task_config(config_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    task_name = config.get("task_name", config_path.parent.name)
    config["task_name"] = task_name
    catalog = _load_catalog()
    if task_name in catalog:
        merged = dict(config)
        merged.update(catalog[task_name])
        config = merged
    config.setdefault("data_file", f"data/{task_name}.csv")
    return config


def run_experiment(config_path: Path, seed: int = 42) -> ExperimentResult:
    config_path = Path(config_path)
    config = read_task_config(config_path)
    metrics = run_task_distillation(config, seed=seed)
    architecture_key = config.get("architecture_key", config.get("model_family", "hf_distill"))
    architecture_name = config.get("architecture_name", config.get("teacher_model", architecture_key))
    architecture_description = config.get("architecture_description", config.get("distillation_method", "Hugging Face distillation"))
    return ExperimentResult(
        task=metrics.task,
        architecture_key=architecture_key,
        architecture_name=architecture_name,
        architecture_description=architecture_description,
        accuracy=metrics.student_acc,
        f1=metrics.student_f1,
        teacher_acc=metrics.teacher_acc,
        teacher_f1=metrics.teacher_f1,
        agreement=metrics.agreement,
        kl_divergence=metrics.kl_divergence,
        teacher_ece=metrics.teacher_ece,
        student_ece=metrics.student_ece,
        learned_flag=metrics.learned_flag,
        notes=metrics.notes,
    )


def run_task_experiment(task_meta: Dict[str, Any], seed: int = 42) -> ExperimentResult:
    metrics = run_task_distillation(task_meta, seed=seed)
    architecture_key = task_meta.get("architecture_key", task_meta.get("model_family", "hf_distill"))
    architecture_name = task_meta.get("architecture_name", task_meta.get("teacher_model", architecture_key))
    architecture_description = task_meta.get("architecture_description", task_meta.get("distillation_method", "Hugging Face distillation"))
    return ExperimentResult(
        task=metrics.task,
        architecture_key=architecture_key,
        architecture_name=architecture_name,
        architecture_description=architecture_description,
        accuracy=metrics.student_acc,
        f1=metrics.student_f1,
        teacher_acc=metrics.teacher_acc,
        teacher_f1=metrics.teacher_f1,
        agreement=metrics.agreement,
        kl_divergence=metrics.kl_divergence,
        teacher_ece=metrics.teacher_ece,
        student_ece=metrics.student_ece,
        learned_flag=metrics.learned_flag,
        notes=metrics.notes,
    )


def write_result(result: ExperimentResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": result.task,
        "architecture_key": result.architecture_key,
        "architecture_name": result.architecture_name,
        "architecture_description": result.architecture_description,
        "accuracy": result.accuracy,
        "f1": result.f1,
        "teacher_acc": result.teacher_acc,
        "teacher_f1": result.teacher_f1,
        "agreement": result.agreement,
        "kl_divergence": result.kl_divergence,
        "teacher_ece": result.teacher_ece,
        "student_ece": result.student_ece,
        "learned_flag": result.learned_flag,
        "notes": result.notes,
    }
    out_path = output_dir / f"{result.task}_result.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return out_path
