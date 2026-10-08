"""Structured, dependency-free summaries for environment-backed generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_item_id_hash(item_ids: list[str]) -> str:
    payload = json.dumps(item_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_artifact_manifest(model_path: str) -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_files = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in weight_files if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing indexed model weights: {missing}")
        return {
            "config_sha256": sha256_file(str(config_path)),
            "index_sha256": sha256_file(str(index_path)),
            "weight_files": {
                name: {
                    "size": (root / name).stat().st_size,
                    "sha256": sha256_file(str(root / name)),
                }
                for name in weight_files
            },
        }
    single_weight = root / "model.safetensors"
    if not single_weight.is_file():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {root}"
        )
    return {
        "config_sha256": sha256_file(str(config_path)),
        "index_sha256": None,
        "weight_files": {
            single_weight.name: {
                "size": single_weight.stat().st_size,
                "sha256": sha256_file(str(single_weight)),
            }
        },
    }


def resolve_bank_path(agentgym_config: Any) -> str | None:
    harness = _config_value(agentgym_config, "harness")
    if harness is None:
        return None
    for key in ("bank_snapshot", "bank_path"):
        value = _config_value(harness, key)
        if value and os.path.isfile(value):
            return os.path.abspath(value)
    return None


def write_generation_summary(
    *,
    output_path: str,
    checkpoint_id: str | None,
    model_path: str,
    task_role: str,
    task_file: str,
    item_ids: list[str],
    output_scores: list[list[float]],
    category_map: dict[str, str],
    decoding: dict[str, Any],
    harness: dict[str, Any],
    bank_path: str | None,
) -> Path:
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Cannot summarize duplicate task IDs")
    if len(output_scores) != len(item_ids):
        raise ValueError(
            f"Expected {len(item_ids)} score rows, got {len(output_scores)}"
        )
    if not output_scores or any(not scores for scores in output_scores):
        raise ValueError("Every task must have at least one score")

    output = Path(output_path)
    if output.suffix.lower() != ".json":
        output = output / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    normalized_scores = [
        [float(score) for score in task_scores]
        for task_scores in output_scores
    ]
    flattened_scores = [score for scores in normalized_scores for score in scores]
    best_scores = [max(scores) for scores in normalized_scores]
    per_task = {
        item_id: {
            "scores": task_scores,
            "reward": task_scores[0] if len(task_scores) == 1 else None,
            "success": max(task_scores) > 0,
            "official_evaluation_completed": True,
            "category": category_map.get(item_id),
        }
        for item_id, task_scores in zip(item_ids, normalized_scores)
    }
    categories = {}
    for category in sorted({value for value in category_map.values()}):
        category_scores = [
            max(scores)
            for item_id, scores in zip(item_ids, normalized_scores)
            if category_map.get(item_id) == category
        ]
        categories[category] = {
            "n_tasks": len(category_scores),
            "mean_reward": sum(category_scores) / len(category_scores),
            "success_rate": (
                sum(score > 0 for score in category_scores) / len(category_scores)
            ),
        }

    summary = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "model": os.path.abspath(model_path),
        "model_artifact": model_artifact_manifest(model_path),
        "task_role": task_role,
        "task_file": os.path.abspath(task_file),
        "task_file_sha256": sha256_file(task_file),
        "task_ids_sha256": semantic_item_id_hash(item_ids),
        "n_tasks": len(item_ids),
        "n_unique_tasks": len(set(item_ids)),
        "n_samples": len(normalized_scores[0]),
        "mean_reward": sum(flattened_scores) / len(flattened_scores),
        "success_rate": sum(score > 0 for score in best_scores) / len(best_scores),
        "n_success": sum(score > 0 for score in best_scores),
        "decoding": decoding,
        "harness": harness,
        "bank": {
            "path": os.path.abspath(bank_path) if bank_path else None,
            "sha256": sha256_file(bank_path) if bank_path else None,
        },
        "categories": categories,
        "per_task": per_task,
    }
    with open(output, "w") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2, sort_keys=True)
    return output


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)
