#!/usr/bin/env python3
"""
aggregate_leaderboard.py
========================
Parses results.json files from multiple team evaluation outputs and generates
a ranked CSV leaderboard sorted by OSIPI Silver Score (descending).

Usage:
    python aggregate_leaderboard.py \\
        --results_dir  ./all_outputs \\
        --output_csv   ./leaderboard.csv

Expected directory structure:
    all_outputs/
        TeamA/results.json
        TeamB/results.json
        TeamC/results.json

Each results.json must contain the keys written by run_pipeline.py.

Also runnable inside Docker container on mounted outputs:
    docker run --rm \\
        -v $(pwd)/all_outputs:/app/all_outputs:ro \\
        -v $(pwd):/app/leaderboard_out \\
        --entrypoint python \\
        osipi-eval-pipeline \\
        /app/aggregate_leaderboard.py \\
        --results_dir /app/all_outputs \\
        --output_csv  /app/leaderboard_out/leaderboard.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("osipi_leaderboard")

# ── CSV columns (in exact order) ──────────────────────────────────────────────
COLUMNS = [
    "Rank",
    "Team",
    "Validation",
    "Silver_Score_%",
    "Gold_Score_%",
    "Accuracy",
    "Repeatability",
    "Reproducibility",
    "Validation_Errors",
]


def _safe(value, fmt=None) -> str:
    """Convert a value to a display-safe string, handling None / NaN."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and value != value:   # NaN check
        return "N/A"
    return fmt.format(value) if fmt else str(value)


def parse_team_result(team_dir: str) -> Optional[dict]:
    """
    Read results.json from a single team output directory.
    Returns a flat dict ready for CSV row output, or None on failure.
    """
    json_path = os.path.join(team_dir, "results.json")
    team_name = os.path.basename(team_dir.rstrip(os.sep))

    if not os.path.isfile(json_path):
        log.warning(f"[{team_name}] No results.json found — skipping.")
        return None

    try:
        # Try utf-8-sig first (handles BOM from Windows tools like PowerShell)
        # then fall back to plain utf-8
        for enc in ("utf-8-sig", "utf-8"):
            try:
                with open(json_path, encoding=enc) as f:
                    data = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                data = None
        if data is None:
            raise ValueError("Could not parse JSON with utf-8 or utf-8-sig")
    except (ValueError, OSError) as exc:
        log.warning(f"[{team_name}] Failed to parse results.json: {exc} — skipping.")
        return None

    # Use the 'team' field from JSON if present, otherwise use folder name
    name = data.get("team") or team_name

    # Silver score (primary ranking key) — use -inf if missing so they sort last
    silver = data.get("osipi_score_silver")
    silver_sort = silver if (silver is not None and silver == silver) else float("-inf")

    errors = data.get("validation_errors") or []
    err_summary = "; ".join(errors[:3])
    if len(errors) > 3:
        err_summary += f" (+{len(errors) - 3} more)"

    return {
        "team_name":        name,
        "silver_sort":      silver_sort,
        "Validation":       "PASS" if data.get("validation_passed") else "FAIL",
        "Silver_Score_%":   _safe(silver, "{:.2f}"),
        "Gold_Score_%":     _safe(data.get("osipi_score_gold"), "{:.2f}"),
        "Accuracy":         _safe(data.get("accuracy_score"), "{:.4f}"),
        "Repeatability":    _safe(data.get("repeatability_score"), "{:.4f}"),
        "Reproducibility":  _safe(data.get("reproducibility_score"), "{:.4f}"),
        "Validation_Errors": err_summary or "None",
    }


def build_leaderboard(results_dir: str, output_csv: str) -> int:
    """
    Walk results_dir for sub-directories containing results.json.
    Write ranked CSV to output_csv.
    Returns the number of teams included.
    """
    if not os.path.isdir(results_dir):
        log.error(f"results_dir does not exist: {results_dir}")
        sys.exit(1)

    teams = []
    for entry in sorted(os.listdir(results_dir)):
        team_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(team_dir):
            continue
        row = parse_team_result(team_dir)
        if row:
            teams.append(row)

    if not teams:
        log.error("No valid results.json files found — CSV not written.")
        sys.exit(1)

    # Sort by silver score descending (highest = rank 1)
    teams.sort(key=lambda r: r["silver_sort"], reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()
        for rank, row in enumerate(teams, start=1):
            writer.writerow({
                "Rank":             rank,
                "Team":             row["team_name"],
                "Validation":       row["Validation"],
                "Silver_Score_%":   row["Silver_Score_%"],
                "Gold_Score_%":     row["Gold_Score_%"],
                "Accuracy":         row["Accuracy"],
                "Repeatability":    row["Repeatability"],
                "Reproducibility":  row["Reproducibility"],
                "Validation_Errors": row["Validation_Errors"],
            })

    log.info(f"Leaderboard written ({len(teams)} teams) -> {output_csv}")
    log.info("")
    log.info(f"  {'Rank':<5} {'Team':<30} {'Silver':>8}  {'Validation':>12}")
    log.info(f"  {'-'*5} {'-'*30} {'-'*8}  {'-'*12}")
    for rank, row in enumerate(teams, 1):
        log.info(f"  {rank:<5} {row['team_name']:<30} {row['Silver_Score_%']:>8}%  {row['Validation']:>12}")

    return len(teams)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a ranked CSV leaderboard from OSIPI pipeline outputs"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Parent directory containing one subfolder per team, each with results.json"
    )
    parser.add_argument(
        "--output_csv",
        default="leaderboard.csv",
        help="Path to write the ranked CSV (default: leaderboard.csv)"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  OSIPI DCE-MRI Challenge — Leaderboard Generator")
    log.info("=" * 60)
    log.info(f"  Scanning : {os.path.abspath(args.results_dir)}")
    log.info(f"  Output   : {os.path.abspath(args.output_csv)}")
    log.info("")

    count = build_leaderboard(args.results_dir, args.output_csv)
    log.info(f"\nDone. {count} team(s) ranked.")


if __name__ == "__main__":
    main()
