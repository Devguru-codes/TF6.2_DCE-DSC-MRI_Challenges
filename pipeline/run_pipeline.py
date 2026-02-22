#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master orchestrator for the OSIPI DCE-MRI automated evaluation pipeline.
This is the ENTRYPOINT of the Docker container.

Steps executed in order:
    1. Validate submitted NIfTI files (validator.py)
    2. Run scoring against ground truth (scorer_wrapper.py)
    3. Generate PDF report (report_gen.py)

Default Docker paths (can be overridden for local testing):
    --submission_dir   /app/submission
    --ground_truth_dir /app/ground_truth
    --output_dir       /app/output

Usage (Docker — default paths):
    docker run --rm \\
        -v $(pwd)/my_submission:/app/submission:ro \\
        -v $(pwd)/DROKtransNifti:/app/ground_truth:ro \\
        -v $(pwd)/output:/app/output \\
        osipi-eval-pipeline

Usage (local testing — override paths):
    python run_pipeline.py \\
        --submission_dir  ../Scoring/entryDirectories/constantKtransModel \\
        --ground_truth_dir ../Scoring/DROKtransNifti \\
        --output_dir      ./test_output \\
        --scoring_script  ../Scoring/challengeScoring.py \\
        --masks_dir       ../Scoring/Masks
"""

import argparse
import json
import logging
import os
import sys
import shutil
from typing import List

# Force UTF-8 output so box-drawing chars work even on Windows cp1252 consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure pipeline/ dir is on the path when run from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validator
import scorer_wrapper
import report_gen


# -------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure dual-handler logging: console + pipeline.log in output_dir."""
    log_path = os.path.join(output_dir, "pipeline.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger("osipi_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers if re-used

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+) — captures everything including tracebacks
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "-" * 60


def header(log: logging.Logger, title: str):
    log.info(SEPARATOR)
    log.info(f"  {title}")
    log.info(SEPARATOR)


def success(log: logging.Logger, msg: str):
    log.info(f"[OK] {msg}")


def fail(log: logging.Logger, msg: str):
    log.error(f"[ERR] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OSIPI DCE-MRI Automated Evaluation Pipeline"
    )
    parser.add_argument(
        "--submission_dir",
        default="/app/submission",
        help="Directory containing submitted NIfTI files (default: /app/submission)"
    )
    parser.add_argument(
        "--ground_truth_dir",
        default="/app/DROKtransNifti",
        help="Directory containing DROKtransNifti ground truth (default: /app/DROKtransNifti)"
    )
    parser.add_argument(
        "--output_dir",
        default="/app/output",
        help="Directory for results.json and Evaluation_Report.pdf (default: /app/output)"
    )
    parser.add_argument(
        "--team_name",
        default="submission",
        help="Entry/team name used in scoring (default: submission)"
    )
    parser.add_argument(
        "--scoring_script",
        default=None,
        help="Explicit path to challengeScoring.py (auto-detected if omitted)"
    )
    parser.add_argument(
        "--masks_dir",
        default=None,
        help="Explicit path to Masks directory (auto-detected if omitted)"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip file validation (not recommended)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logger — MUST happen after output_dir is created
    log = setup_logging(args.output_dir)

    log.info("=" * 60)
    log.info("  OSIPI DCE-MRI Challenge -- Automated Evaluation Pipeline")
    log.info("=" * 60)
    log.info(f"  Submission dir  : {args.submission_dir}")
    log.info(f"  Ground truth dir: {args.ground_truth_dir}")
    log.info(f"  Output dir      : {args.output_dir}")
    log.info(f"  Log file        : {os.path.join(args.output_dir, 'pipeline.log')}")

    validation_passed = False
    validation_errors: List[str] = []

    # ── STEP 1: Validate ──────────────────────────────────────────────────────
    header(log, "STEP 1 / 3 — File Validation")
    if args.skip_validation:
        log.warning("Validation skipped (--skip_validation flag set).")
        validation_passed = True
    else:
        is_valid, errors = validator.validate_submission(args.submission_dir)
        if is_valid:
            success(log, "All submission files are valid.")
            validation_passed = True
        else:
            fail(log, f"{len(errors)} validation error(s) found:")
            for err in errors:
                log.error(f"     - {err}")
            validation_errors = errors
            # We do NOT abort here — we continue to scoring and include the
            # FAIL status in the PDF report. This gives the submitter a full
            # report even when files are missing.
            log.warning("Continuing pipeline to generate partial report...")

    # ── STEP 2: Score ─────────────────────────────────────────────────────────
    header(log, "STEP 2 / 3 — Scoring")

    # If masks_dir is provided, patch scorer_wrapper to find it
    if args.masks_dir:
        os.environ["OSIPI_MASKS_DIR"] = os.path.abspath(args.masks_dir)

    # Resolve scoring script
    scoring_script = args.scoring_script
    if scoring_script is None:
        candidates = [
            "/app/challengeScoring.py",
            os.path.join(os.path.dirname(__file__), "..", "Scoring", "challengeScoring.py"),
            os.path.join(os.path.dirname(__file__), "challengeScoring.py"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                scoring_script = os.path.normpath(candidate)
                break
        if scoring_script is None:
            fail(log, "Cannot locate challengeScoring.py. Use --scoring_script to specify its location.")
            sys.exit(1)

    log.info(f"Using scoring script: {scoring_script}")

    try:
        results = scorer_wrapper.run_scoring(
            submission_dir=args.submission_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_dir=args.output_dir,
            team_name=args.team_name,
            scoring_script=scoring_script,
        )
        # Inject validation result into results.json
        results["validation_passed"] = validation_passed
        results["validation_errors"] = validation_errors

        results_json_path = os.path.join(args.output_dir, "results.json")
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        success(log, f"Scoring complete -> {results_json_path}")
        log.info(f"    Accuracy Score        : {results.get('accuracy_score')}")
        log.info(f"    Repeatability Score   : {results.get('repeatability_score')}")
        log.info(f"    Reproducibility Score : {results.get('reproducibility_score')}")
        log.info(f"    OSIPI Silver Score    : {results.get('osipi_score_silver')}%")
        log.info(f"    OSIPI Gold Score      : {results.get('osipi_score_gold')}%")

    except Exception as e:
        fail(log, f"Scoring failed: {e}")
        import traceback
        log.debug(traceback.format_exc())
        sys.exit(1)

    # ── STEP 3: Generate PDF Report ───────────────────────────────────────────
    header(log, "STEP 3 / 3 — Report Generation")
    try:
        pdf_path = report_gen.generate_report(
            results_json_path=results_json_path,
            output_dir=args.output_dir,
        )
        success(log, f"Report generated: {pdf_path}")
    except Exception as e:
        fail(log, f"Report generation failed: {e}")
        import traceback
        log.debug(traceback.format_exc())
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Pipeline Complete")
    log.info("=" * 60)
    log.info(f"  [JSON] results.json        -> {results_json_path}")
    log.info(f"  [PDF] Evaluation_Report    -> {pdf_path}")
    log.info(f"  [LOG] pipeline.log         -> {os.path.join(args.output_dir, 'pipeline.log')}")
    val_status = "PASS" if validation_passed else "FAIL"
    log.info(f"  [VAL] Validation Status    -> {val_status}")
    log.info("=" * 60)

    sys.exit(0 if validation_passed else 2)  # exit 2 = scoring done but validation failed


if __name__ == "__main__":
    main()
