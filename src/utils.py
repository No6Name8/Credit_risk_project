import json
from pathlib import Path
from datetime import datetime

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def latest_run_dir(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda x: x.name)[-1]
