"""Trainer callbacks: metric logging and best-checkpoint tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MetricLoggerCallback(TrainerCallback):
    """Accumulate eval metrics and write them to ``training_metrics.json``.

    Also records the best evaluation loss seen so far so the README / reports
    can quote it without parsing trainer logs.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.history: list[dict[str, Any]] = []
        self.best_eval_loss: float | None = None
        self.output_dir = Path(output_dir)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any, **kwargs: Any) -> None:
        entry = {"step": state.global_step, **metrics}
        self.history.append(entry)
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and (
            self.best_eval_loss is None or eval_loss < self.best_eval_loss
        ):
            self.best_eval_loss = float(eval_loss)
            entry["best"] = True
        logger.info("Step %s eval metrics: %s", state.global_step, metrics)

    def save(self) -> Path | None:
        """Write the accumulated history to disk (no-op when nothing recorded)."""
        if not self.history:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "training_metrics.json"
        path.write_text(
            json.dumps({"history": self.history, "best_eval_loss": self.best_eval_loss}, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved training metrics history to %s", path)
        return path
