import json
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@contextmanager
def timer_ms(name: str, store: Dict[str, float]):
    t0 = time.perf_counter()
    yield
    store[name] = round((time.perf_counter() - t0) * 1000.0, 2)


def citation_coverage_percent(answer: str) -> float:
    lines = [l.strip() for l in (answer or "").splitlines() if l.strip()]
    if not lines:
        return 0.0
    cited = sum(1 for l in lines if re.search(r"\[[^\]]+:\d+\]", l))
    return round((cited / len(lines)) * 100.0, 1)


def safe_mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def log_metrics_jsonl(path: str, payload: Dict[str, Any]) -> None:
    safe_mkdir(str(Path(path).parent))
    out = {"ts_utc": datetime.utcnow().isoformat(), **payload}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


def docs_coverage_ratio(retrieved_metadatas: List[Dict[str, Any]]) -> float:
    doc_ids = [m.get("doc_id") for m in retrieved_metadatas if m.get("doc_id")]
    if not doc_ids:
        return 0.0
    return round(len(set(doc_ids)) / max(len(doc_ids), 1), 3)
