from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunArtifacts:
    run_name: str
    model_dir: Path
    log_dir: Path
    result_dir: Path
    best_checkpoint: Path
    log_file: Path
    config_file: Path
    metrics_file: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_run_artifacts(config) -> RunArtifacts:
    run_name = config.runtime.run_name or _timestamp()
    model_dir = config.runtime.output_root / "models" / config.model_name / config.task_type / config.dataset / run_name
    log_dir = config.runtime.output_root / "logs" / config.model_name / config.task_type / config.dataset / run_name
    result_dir = config.runtime.output_root / "results" / config.model_name / config.task_type / config.dataset / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_name=run_name,
        model_dir=model_dir,
        log_dir=log_dir,
        result_dir=result_dir,
        best_checkpoint=model_dir / "best.pt",
        log_file=log_dir / "run.log",
        config_file=result_dir / "config.json",
        metrics_file=result_dir / "metrics.json",
    )


def create_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_json(path: Path, content) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False, default=str)
