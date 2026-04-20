"""Minimal experiment logger — saves JSON records to experiments/<subdir>/."""
import json
import os
from datetime import datetime

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


def save(subdir: str, data: dict, label: str = "") -> str:
    """Persist *data* as JSON under experiments/<subdir>/.

    Returns the path of the written file.
    """
    out_dir = os.path.join(_EXPERIMENTS_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data.setdefault("timestamp", timestamp)

    slug = label.replace(" ", "_") if label else ""
    filename = f"{timestamp}_{slug}.json" if slug else f"{timestamp}.json"
    path = os.path.join(out_dir, filename)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path
