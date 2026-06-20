"""
config.py

Loads the YAML training configuration into a simple, attribute-accessible
object so the rest of the codebase can do `cfg.data.scale_factor` instead
of `cfg["data"]["scale_factor"]`.

Why this exists:
    Hardcoding paths/hyperparameters across multiple files makes a project
    hard to maintain and hard to reason about during a code review or
    interview walkthrough. A single config file + loader is standard
    practice in production ML codebases.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert nested dicts into SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = _dict_to_namespace(value)
        setattr(ns, key, value)
    return ns


def load_config(config_path: str | Path = "configs/train_config.yaml") -> SimpleNamespace:
    """
    Load the YAML config file and return it as a nested namespace object.

    Args:
        config_path: path to the YAML config file.

    Returns:
        SimpleNamespace with dot-accessible config values, e.g. cfg.data.scale_factor

    Raises:
        FileNotFoundError: if the config file does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            f"Run this script from the project root, or pass an explicit path."
        )

    with open(config_path, "r") as f:
        raw_dict = yaml.safe_load(f)

    return _dict_to_namespace(raw_dict)


if __name__ == "__main__":
    # Quick manual check: run `python -m src.utils.config` from project root
    cfg = load_config()
    print("Scale factor:", cfg.data.scale_factor)
    print("Batch size:", cfg.training.batch_size)
    print("Checkpoint dir:", cfg.training.checkpoint_dir)
