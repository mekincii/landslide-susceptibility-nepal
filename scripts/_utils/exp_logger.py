# scripts/_utils/exp_logger.py
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _jsonify(v: Any) -> str:
    """Make values safe for CSV (strings)."""
    if v is None:
        return ""
    if isinstance(v, (int, float, str, bool)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def log_experiment(row: Dict[str, Any], out_csv: str | Path = "outputs/experiments/experiments.csv") -> Path:
    """
    Append a single experiment result row to a CSV file (creates file + header if missing).

    Parameters
    ----------
    row : dict
        Your experiment record. Values can be any type; they'll be serialized safely.
    out_csv : str | Path
        CSV file path.

    Returns
    -------
    Path to the CSV written.
    """
    out_path = Path(out_csv)
    _ensure_parent(out_path)

    # Add a timestamp if user didn't provide one
    if "timestamp_utc" not in row:
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # We keep a stable header: union of existing header + new keys (in a deterministic order)
    row_keys = list(row.keys())

    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        header = list(existing_header)
        # Append any new keys at the end (preserve previous column order)
        for k in row_keys:
            if k not in header:
                header.append(k)
        needs_rewrite = (header != existing_header)
    else:
        header = row_keys
        needs_rewrite = False

    # If header expanded, rewrite file with new header (preserving old rows)
    if needs_rewrite:
        tmp_path = out_path.with_suffix(".tmp.csv")
        with out_path.open("r", newline="", encoding="utf-8") as fin, tmp_path.open("w", newline="", encoding="utf-8") as fout:
            old_reader = csv.DictReader(fin)
            new_writer = csv.DictWriter(fout, fieldnames=header)
            new_writer.writeheader()
            for r in old_reader:
                new_writer.writerow({k: r.get(k, "") for k in header})
        tmp_path.replace(out_path)

    # Append the row
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({k: _jsonify(row.get(k)) for k in header})

    return out_path