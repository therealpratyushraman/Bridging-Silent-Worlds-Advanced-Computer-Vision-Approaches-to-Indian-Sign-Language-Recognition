"""Training callbacks for model checkpointing and early stopping."""

import os
import json
from typing import Dict, Optional


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        monitor: Metric to monitor ('loss' or 'map50').
        mode: 'min' for loss, 'max' for accuracy/mAP.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """Check if training should stop.

        Args:
            current_value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """Save model weights when a monitored metric improves.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric to monitor.
        mode: 'min' for loss, 'max' for accuracy/mAP.
        save_best_only: Only save when metric improves.
    """

    def __init__(
        self,
        save_dir: str = "runs/checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = None

        os.makedirs(save_dir, exist_ok=True)

    def __call__(
        self, model, epoch: int, current_value: float, metrics: Dict = None
    ) -> Optional[str]:
        """Check and potentially save model.

        Args:
            model: Model to save (must have a save/export method or be a torch module).
            epoch: Current epoch number.
            current_value: Current metric value.
            metrics: Additional metrics to log.

        Returns:
            Path to saved model if saved, None otherwise.
        """
        should_save = True

        if self.save_best_only:
            if self.best_value is None:
                self.best_value = current_value
            elif self.mode == "min" and current_value >= self.best_value:
                should_save = False
            elif self.mode == "max" and current_value <= self.best_value:
                should_save = False

        if should_save:
            self.best_value = current_value
            save_path = os.path.join(self.save_dir, f"best.pt")

            if hasattr(model, "save"):
                model.save(save_path)
            elif hasattr(model, "state_dict"):
                import torch
                torch.save(model.state_dict(), save_path)

            if metrics:
                metrics_path = os.path.join(self.save_dir, "best_metrics.json")
                log_entry = {"epoch": epoch, self.monitor: current_value}
                log_entry.update(metrics)
                with open(metrics_path, "w") as f:
                    json.dump(log_entry, f, indent=2)

            print(
                f"Checkpoint saved: epoch {epoch}, "
                f"{self.monitor}={current_value:.4f} -> {save_path}"
            )
            return save_path

        return None


class MetricLogger:
    """Log training metrics to a JSON file.

    Args:
        log_dir: Directory to save logs.
    """

    def __init__(self, log_dir: str = "runs/logs"):
        self.log_dir = log_dir
        self.history = []
        os.makedirs(log_dir, exist_ok=True)

    def log(self, epoch: int, metrics: Dict):
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric name to value.
        """
        entry = {"epoch": epoch}
        entry.update(metrics)
        self.history.append(entry)

        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self):
        """Return full training history."""
        return self.history

    def get_best(self, metric: str, mode: str = "max") -> Dict:
        """Get the best epoch for a given metric.

        Args:
            metric: Metric name to find best for.
            mode: 'max' or 'min'.

        Returns:
            Dictionary of the best epoch's metrics.
        """
        if not self.history:
            return {}

        valid = [e for e in self.history if metric in e]
        if not valid:
            return {}

        if mode == "max":
            return max(valid, key=lambda x: x[metric])
        return min(valid, key=lambda x: x[metric])
