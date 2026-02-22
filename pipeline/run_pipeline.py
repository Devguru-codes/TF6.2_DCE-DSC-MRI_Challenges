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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "-" * 60


def header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def success(msg: str):
    print(f"  [OK] {msg}")


def fail(msg: str):
    print(f"  [ERR] {msg}", file=sys.stderr)


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

    print("\n" + "=" * 60)
    print("  OSIPI DCE-MRI Challenge -- Automated Evaluation Pipeline")
    print("=" * 60)
    print(f"  Submission dir  : {args.submission_dir}")
    print(f"  Ground truth dir: {args.ground_truth_dir}")
    print(f"  Output dir      : {args.output_dir}")

    validation_passed = False
    validation_errors: List[str] = []

    # ── STEP 1: Validate ──────────────────────────────────────────────────────
    header("STEP 1 / 3 — File Validation")
    if args.skip_validation:
        print("  [WARN] Validation skipped (--skip_validation flag set).")
        validation_passed = True
    else:
        is_valid, errors = validator.validate_submission(args.submission_dir)
        if is_valid:
            success("All submission files are valid.")
            validation_passed = True
        else:
            fail(f"{len(errors)} validation error(s) found:")
            for err in errors:
                print(f"     - {err}", file=sys.stderr)
            validation_errors = errors
            # We do NOT abort here — we continue to scoring and include the
            # FAIL status in the PDF report. This gives the submitter a full
            # report even when files are missing.
            print("  [WARN] Continuing pipeline to generate partial report...", file=sys.stderr)

    # ── STEP 2: Score ─────────────────────────────────────────────────────────
    header("STEP 2 / 3 — Scoring")

    # If masks_dir is provided, patch scorer_wrapper to find it
    if args.masks_dir:
        # Inject a tiny shim so scorer_wrapper._link_or_copy resolves Masks correctly
        os.environ["OSIPI_MASKS_DIR"] = os.path.abspath(args.masks_dir)

    # Resolve scoring script
    scoring_script = args.scoring_script
    if scoring_script is None:
        # Auto-detect: check standard Docker path first, then local repo path
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
            fail("Cannot locate challengeScoring.py. Use --scoring_script to specify its location.")
            sys.exit(1)

    print(f"  Using scoring script: {scoring_script}")

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

        success(f"Scoring complete. Results: {results_json_path}")
        print()
        print(f"    Accuracy Score        : {results.get('accuracy_score')}")
        print(f"    Repeatability Score   : {results.get('repeatability_score')}")
        print(f"    Reproducibility Score : {results.get('reproducibility_score')}")
        print(f"    OSIPI Silver Score    : {results.get('osipi_score_silver')}%")
        print(f"    OSIPI Gold Score      : {results.get('osipi_score_gold')}%")

    except Exception as e:
        fail(f"Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── STEP 3: Generate PDF Report ───────────────────────────────────────────
    header("STEP 3 / 3 — Report Generation")
    try:
        pdf_path = report_gen.generate_report(
            results_json_path=results_json_path,
            output_dir=args.output_dir,
        )
        success(f"Report generated: {pdf_path}")
    except Exception as e:
        fail(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  [JSON] results.json        -> {results_json_path}")
    print(f"  [PDF] Evaluation_Report    -> {pdf_path}")
    val_status = "[PASS]" if validation_passed else "[FAIL]"
    print(f"  [VAL] Validation Status    -> {val_status}")
    print("=" * 60 + "\n")

    sys.exit(0 if validation_passed else 2)  # exit 2 = scoring done but validation failed


if __name__ == "__main__":
    main()
