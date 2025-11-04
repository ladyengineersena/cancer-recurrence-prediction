import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_feature_columns(prefix: str, columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith(prefix)]

