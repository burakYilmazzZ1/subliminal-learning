"""Hugging Face tabanlı task-specific fine-tuning and distillation engine.

This module loads the task catalog entry, fetches the real HF dataset,
finetunes a teacher model, then distills knowledge into a student model using
logit matching + hard-label supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass(frozen=True)
class DistillationMetrics:
    task: str
    dataset_name: str
    dataset_config: str
    teacher_model: str
    student_model: str
    teacher_acc: float
    student_acc: float
    teacher_f1: float
    student_f1: float
    agreement: float
    kl_divergence: float
    teacher_ece: float
    student_ece: float
    learned_flag: bool
    notes: str


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        hard_loss = F.cross_entropy(student_logits, labels)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        return (loss, student_outputs) if return_outputs else loss


def _safe_get(task_meta: Dict[str, Any], key: str, default: Any = None) -> Any:
    return task_meta.get(key, default)


def _load_dataset(task_meta: Dict[str, Any]) -> DatasetDict:
    dataset_name = _safe_get(task_meta, "dataset_name")
    dataset_config = _safe_get(task_meta, "dataset_config")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})
    return dataset


def _resolve_split_names(dataset: DatasetDict, task_meta: Dict[str, Any]) -> Tuple[str, str]:
    train_split = _safe_get(task_meta, "train_split", "train")
    eval_split = _safe_get(task_meta, "eval_split")
    if eval_split and eval_split in dataset:
        return train_split, eval_split
    if "validation" in dataset:
        return train_split, "validation"
    if "validation_matched" in dataset:
        return train_split, "validation_matched"
    return train_split, "train"


def _apply_sample_limit(ds: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit <= 0 or len(ds) <= limit:
        return ds
    return ds.shuffle(seed=seed).select(range(limit))


def _resolve_text_columns(task_meta: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    text_field = _safe_get(task_meta, "text_field", "text")
    text_pair_field = _safe_get(task_meta, "text_pair_field")
    if text_pair_field in ("", None):
        text_pair_field = None
    return text_field, text_pair_field


def _resolve_label_column(dataset: Dataset, task_meta: Dict[str, Any]) -> str:
    label_field = _safe_get(task_meta, "label_field", "label")
    if label_field in dataset.column_names:
        return label_field
    for candidate in ("label", "labels", "coarse_label", "fine_label", "label-coarse", "label_fine"):
        if candidate in dataset.column_names:
            return candidate
    raise KeyError(f"Label column not found for {task_meta.get('name', 'unknown_task')}")


def _normalise_text_batch(batch: Dict[str, Any], text_field: str, text_pair_field: Optional[str], noise_prob: float = 0.0) -> Dict[str, Any]:
    texts = batch[text_field]
    if text_pair_field is None:
        combined = texts
    else:
        pairs = batch[text_pair_field]
        combined = [f"{left} [SEP] {right}" for left, right in zip(texts, pairs)]

    if noise_prob > 0.0:
        rng = np.random.RandomState(42)
        noisy = []
        for text in combined:
            tokens = str(text).split()
            if len(tokens) > 3:
                keep = [tok for tok in tokens if rng.rand() > noise_prob]
                if keep:
                    tokens = keep
            noisy.append(" ".join(tokens))
        combined = noisy

    return {"text": combined}


def _encode_labels(dataset: Dataset, label_field: str) -> Dataset:
    if label_field not in dataset.column_names:
        return dataset
    if dataset.features[label_field].dtype.kind in "biu":
        return dataset.rename_column(label_field, "labels")
    # string labels are factorized deterministically
    labels = dataset[label_field]
    classes = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(classes)}
    return dataset.map(lambda row: {"labels": mapping[row[label_field]]})


def _tokenize_dataset(dataset: Dataset, tokenizer, text_field: str, text_pair_field: Optional[str], max_length: int, noise_prob: float = 0.0) -> Dataset:
    def transform(batch):
        normalised = _normalise_text_batch(batch, text_field, text_pair_field, noise_prob=noise_prob)
        return tokenizer(normalised["text"], truncation=True, max_length=max_length)

    remove_columns = [col for col in dataset.column_names if col not in {"labels", "label"}]
    tokenized = dataset.map(transform, batched=True, remove_columns=remove_columns)
    if "label" in tokenized.column_names and "labels" not in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float(np.mean(preds == labels))
    f1 = _weighted_f1(labels, preds)
    return {"accuracy": accuracy, "f1": f1}


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    weights = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(f1)
        weights.append(np.sum(y_true == cls))
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(np.asarray(scores) * weights) / max(1.0, weights.sum()))


def _ece_from_probs(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        mask = (confidences >= bin_edges[idx]) & (confidences < bin_edges[idx + 1])
        if not np.any(mask):
            continue
        acc = np.mean(predictions[mask] == labels[mask])
        conf = np.mean(confidences[mask])
        ece += np.abs(acc - conf) * np.mean(mask)
    return float(ece)


def _predict_probs(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    output = trainer.predict(dataset)
    logits = output.predictions
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()


def _prepare_splits(task_meta: Dict[str, Any], seed: int) -> Tuple[Dataset, Dataset]:
    dataset = _load_dataset(task_meta)
    train_split, eval_split = _resolve_split_names(dataset, task_meta)

    if train_split not in dataset:
        raise KeyError(f"Train split not found: {train_split}")
    train_ds = dataset[train_split]
    eval_ds = dataset[eval_split] if eval_split in dataset else dataset[train_split].train_test_split(test_size=0.2, seed=seed)["test"]

    train_limit = _safe_get(task_meta, "max_train_samples")
    eval_limit = _safe_get(task_meta, "max_eval_samples")
    train_ds = _apply_sample_limit(train_ds.shuffle(seed=seed), train_limit, seed)
    eval_ds = _apply_sample_limit(eval_ds.shuffle(seed=seed), eval_limit, seed)
    return train_ds, eval_ds


def _trainer_args(task_meta: Dict[str, Any], output_dir: str, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(_safe_get(task_meta, "learning_rate", 5e-5)),
        per_device_train_batch_size=int(_safe_get(task_meta, "train_batch_size", 16)),
        per_device_eval_batch_size=int(_safe_get(task_meta, "eval_batch_size", 32)),
        num_train_epochs=float(_safe_get(task_meta, "num_train_epochs", 2.0)),
        weight_decay=float(_safe_get(task_meta, "weight_decay", 0.01)),
        warmup_steps=int(_safe_get(task_meta, "warmup_steps", 0)),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=seed,
        remove_unused_columns=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=False,
    )


def _train_teacher(task_meta: Dict[str, Any], train_ds: Dataset, eval_ds: Dataset, num_labels: int, tokenizer, seed: int):
    teacher_model_name = _safe_get(task_meta, "teacher_model")
    model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.config.label2id = {str(i): i for i in range(num_labels)}
    model.config.id2label = {i: str(i) for i in range(num_labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    args = _trainer_args(task_meta, output_dir=f".dist/{task_meta['name']}/teacher", seed=seed)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    probs = _predict_probs(trainer, eval_ds)
    preds = np.argmax(probs, axis=-1)
    labels = np.asarray(eval_ds["labels"])
    return trainer, metrics, probs, preds, labels


def _train_student(task_meta: Dict[str, Any], train_ds: Dataset, eval_ds: Dataset, num_labels: int, tokenizer, teacher_trainer: Trainer, seed: int):
    student_model_name = _safe_get(task_meta, "student_model")
    model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.config.label2id = {str(i): i for i in range(num_labels)}
    model.config.id2label = {i: str(i) for i in range(num_labels)}
    teacher_model = teacher_trainer.model
    teacher_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    teacher_model.to(device)
    args = _trainer_args(task_meta, output_dir=f".dist/{task_meta['name']}/student", seed=seed)
    trainer = DistillationTrainer(
        model=model,
        teacher_model=teacher_model,
        temperature=float(_safe_get(task_meta, "temperature", 2.0)),
        alpha=float(_safe_get(task_meta, "alpha", 0.5)),
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    probs = _predict_probs(trainer, eval_ds)
    preds = np.argmax(probs, axis=-1)
    labels = np.asarray(eval_ds["labels"])
    return trainer, metrics, probs, preds, labels


def run_task_distillation(task_meta: Dict[str, Any], seed: int = 42) -> DistillationMetrics:
    set_seed(seed)
    train_ds, eval_ds = _prepare_splits(task_meta, seed=seed)
    text_field = _safe_get(task_meta, "text_field", "text")
    text_pair_field = _safe_get(task_meta, "text_pair_field")
    if text_pair_field in ("", None):
        text_pair_field = None
    max_length = int(_safe_get(task_meta, "max_length", 256))
    noise_prob = float(_safe_get(task_meta, "noise_probability", 0.0))
    label_field = _resolve_label_column(train_ds, task_meta)

    tokenizer = AutoTokenizer.from_pretrained(_safe_get(task_meta, "teacher_tokenizer", _safe_get(task_meta, "teacher_model")))

    if label_field != "labels":
        if label_field in train_ds.column_names:
            train_ds = train_ds.rename_column(label_field, "labels")
        if label_field in eval_ds.column_names:
            eval_ds = eval_ds.rename_column(label_field, "labels")
    elif "labels" not in train_ds.column_names and "label" in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
        eval_ds = eval_ds.rename_column("label", "labels")

    if "labels" not in train_ds.column_names:

        classes = sorted(set(train_ds[label_field] if label_field in train_ds.column_names else train_ds["label"]))
        mapping = {label: idx for idx, label in enumerate(classes)}

        def map_labels(batch):
            raw = batch[label_field] if label_field in batch else batch["label"]
            return {"labels": [mapping[item] for item in raw]}

        train_ds = train_ds.map(map_labels, batched=True)
        eval_ds = eval_ds.map(map_labels, batched=True)

    num_labels = int(_safe_get(task_meta, "num_labels", len(set(train_ds["labels"]))))
    train_ds = _tokenize_dataset(train_ds, tokenizer, text_field, text_pair_field, max_length=max_length, noise_prob=noise_prob)
    eval_ds = _tokenize_dataset(eval_ds, tokenizer, text_field, text_pair_field, max_length=max_length, noise_prob=0.0)
    train_ds.set_format(type="torch", columns=[c for c in train_ds.column_names if c in {"input_ids", "token_type_ids", "attention_mask", "labels"}])
    eval_ds.set_format(type="torch", columns=[c for c in eval_ds.column_names if c in {"input_ids", "token_type_ids", "attention_mask", "labels"}])

    teacher_trainer, teacher_metrics, teacher_probs, teacher_preds, labels = _train_teacher(
        task_meta, train_ds, eval_ds, num_labels=num_labels, tokenizer=tokenizer, seed=seed
    )
    student_trainer, student_metrics, student_probs, student_preds, _ = _train_student(
        task_meta, train_ds, eval_ds, num_labels=num_labels, tokenizer=tokenizer, teacher_trainer=teacher_trainer, seed=seed
    )

    teacher_acc = float(teacher_metrics.get("eval_accuracy", _compute_metrics((teacher_probs, labels))["accuracy"]))
    student_acc = float(student_metrics.get("eval_accuracy", _compute_metrics((student_probs, labels))["accuracy"]))
    teacher_f1 = float(teacher_metrics.get("eval_f1", _compute_metrics((teacher_probs, labels))["f1"]))
    student_f1 = float(student_metrics.get("eval_f1", _compute_metrics((student_probs, labels))["f1"]))
    agreement = float(np.mean(teacher_preds == student_preds))
    kl_div = float(np.mean(np.sum(teacher_probs * (np.log(np.clip(teacher_probs, 1e-9, 1.0)) - np.log(np.clip(student_probs, 1e-9, 1.0))), axis=1)))
    teacher_ece = _ece_from_probs(teacher_probs, labels)
    student_ece = _ece_from_probs(student_probs, labels)
    learned_flag = bool((student_acc > 0.05) and (student_acc >= teacher_acc - 0.01 or student_f1 >= teacher_f1 - 0.01))

    return DistillationMetrics(
        task=_safe_get(task_meta, "name", _safe_get(task_meta, "task_name", "unknown_task")),
        dataset_name=_safe_get(task_meta, "dataset_name", "unknown_dataset"),
        dataset_config=str(_safe_get(task_meta, "dataset_config", "") or ""),
        teacher_model=_safe_get(task_meta, "teacher_model", "unknown_teacher"),
        student_model=_safe_get(task_meta, "student_model", "unknown_student"),
        teacher_acc=teacher_acc,
        student_acc=student_acc,
        teacher_f1=teacher_f1,
        student_f1=student_f1,
        agreement=agreement,
        kl_divergence=kl_div,
        teacher_ece=teacher_ece,
        student_ece=student_ece,
        learned_flag=learned_flag,
        notes=f"train={len(train_ds)} eval={len(eval_ds)} labels={num_labels} temp={_safe_get(task_meta, 'temperature', 2.0)}",
    )
