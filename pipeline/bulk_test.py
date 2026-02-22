#!/usr/bin/env python3
"""
bulk_test.py
============
Comprehensive pipeline stress-test.

Tests (in order):
  A. REAL DATA: 1 run using the actual constantKtransModel from the repo
  B. SYNTHETIC (10 submissions): varying constant Ktrans values
  C. EDGE CASES (validator/scorer hardening):
      - Missing 1 file (19/20)
      - Compressed .nii.gz files
      - Wrong filenames (invalid format)
      - Mix of valid + invalid filenames
      - Out-of-range patient numbers (P9 clinical)
      - Empty directory
      - Duplicate patient-visit (same patient as .nii and .nii.gz)
      - NaN values in NIfTI
      - All-zero NIfTI values
      - Very large Ktrans (1000x the expected value)
      - Negative Ktrans values
      - Non-existent submission dir
      - Non-existent ground truth dir
      - Special characters in team name (should be sanitized)

Usage (from pipeline/ dir):
    venv\\Scripts\\python bulk_test.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

# =============================================================================
# Config
# =============================================================================
HERE            = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT       = os.path.normpath(os.path.join(HERE, ".."))
REFERENCE_DIR   = os.path.join(REPO_ROOT, "Scoring", "entryDirectories", "constantKtransModel")
GROUND_TRUTH    = os.path.join(REPO_ROOT, "Scoring", "DROKtransNifti")
MASKS_DIR       = os.path.join(REPO_ROOT, "Scoring", "Masks")
SCORING_SCRIPT  = os.path.join(REPO_ROOT, "Scoring", "challengeScoring.py")
PYTHON          = os.path.join(HERE, "venv", "Scripts", "python.exe")
PIPELINE_SCRIPT = os.path.join(HERE, "run_pipeline.py")

CLINICAL_PATIENTS  = 8
SYNTHETIC_PATIENTS = 2
VISITS             = 2

print("=" * 72)
print("  OSIPI DCE-MRI Pipeline -- Comprehensive Edge Case Test Suite")
print("=" * 72)
print()

# =============================================================================
# Load reference NIfTI
# =============================================================================
print("[Setup] Loading reference NIfTI ...", end=" ", flush=True)
ref_file   = os.path.join(REFERENCE_DIR, "Clinical_P1_Visit1.nii")
ref_img    = nib.load(ref_file)
ref_affine = ref_img.affine
ref_header = ref_img.header.copy()
ref_shape  = ref_img.shape
print(f"shape={ref_shape}  dtype={ref_img.get_data_dtype()}")


def _make_nifti(data: np.ndarray, path: str):
    """Save a NIfTI file with the same geometry as the reference."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref_affine, ref_header), path)


def _make_submission(base_dir: str, name: str, ktrans_val: float = 0.1,
                     missing: list = None, extra_files: list = None,
                     use_gz: bool = False, data_override: dict = None) -> str:
    """
    Create a submission directory.
    missing      : list of (type, p, v) tuples to skip
    extra_files  : list of filenames to add (with dummy data)
    use_gz       : save all as .nii.gz
    data_override: dict mapping filename -> numpy array (no extension)
    """
    sub_dir = os.path.join(base_dir, name)
    os.makedirs(sub_dir, exist_ok=True)
    missing = missing or []
    data_override = data_override or {}

    ext = ".nii.gz" if use_gz else ".nii"

    for p in range(1, CLINICAL_PATIENTS + 1):
        for v in range(1, VISITS + 1):
            fname = f"Clinical_P{p}_Visit{v}"
            if ("Clinical", p, v) in missing:
                continue
            data = data_override.get(fname, np.full(ref_shape, ktrans_val, dtype=np.float32))
            _make_nifti(data, os.path.join(sub_dir, fname + ext))
    for p in range(1, SYNTHETIC_PATIENTS + 1):
        for v in range(1, VISITS + 1):
            fname = f"Synthetic_P{p}_Visit{v}"
            if ("Synthetic", p, v) in missing:
                continue
            data = data_override.get(fname, np.full(ref_shape, ktrans_val, dtype=np.float32))
            _make_nifti(data, os.path.join(sub_dir, fname + ext))

    for ef in (extra_files or []):
        _make_nifti(np.zeros(ref_shape, np.float32), os.path.join(sub_dir, ef))

    return sub_dir


def run_pipeline(submission_dir: str, output_dir: str, label: str,
                 extra_args: list = None) -> dict:
    """Run the pipeline subprocess and return a summary dict."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        PYTHON, PIPELINE_SCRIPT,
        "--submission_dir",   submission_dir,
        "--ground_truth_dir", GROUND_TRUTH,
        "--output_dir",       output_dir,
        "--scoring_script",   SCORING_SCRIPT,
        "--masks_dir",        MASKS_DIR,
    ]
    if extra_args:
        cmd += extra_args

    t0 = time.time()
    ret = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    result_path = os.path.join(output_dir, "results.json")
    pdf_path    = os.path.join(output_dir, "Evaluation_Report.pdf")

    data = {}
    if os.path.isfile(result_path):
        with open(result_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"_json_error": True}

    # Capture relevant error snippet from stderr
    stderr_lines = [ln for ln in ret.stderr.splitlines() if ln.strip()]
    err_snippet = stderr_lines[-1][:120] if stderr_lines else ""

    return {
        "label":               label,
        "exit_code":           ret.returncode,
        "elapsed_s":           elapsed,
        "validation_passed":   data.get("validation_passed"),
        "accuracy_score":      data.get("accuracy_score"),
        "silver":              data.get("osipi_score_silver"),
        "json_exists":         os.path.isfile(result_path),
        "pdf_exists":          os.path.isfile(pdf_path),
        "stderr":              err_snippet,
    }


# =============================================================================
# Build all test cases
# =============================================================================
tmpdir = tempfile.mkdtemp(prefix="osipi_edge_")
print(f"[Setup] Temp dir: {tmpdir}\n")

results = []
assertions_ok = True

def run_case(sub_dir, out_dir, label, expect_exit, expect_val=None, extra_args=None):
    """Run a test case and print progress."""
    global assertions_ok
    idx = len(results) + 1
    print(f"[{idx:02d}] {label[:55]:<55}", end=" ... ", flush=True)
    r = run_pipeline(sub_dir, out_dir, label, extra_args=extra_args)
    results.append(r)

    status_parts = []
    passed = True

    # Check exit code
    if expect_exit == "nonzero":
        if r["exit_code"] == 0:
            status_parts.append("FAIL(expected_nonzero)")
            passed = False
            assertions_ok = False
    elif expect_exit is not None:
        if r["exit_code"] != expect_exit:
            status_parts.append(f"FAIL(exit={r['exit_code']} expected={expect_exit})")
            passed = False
            assertions_ok = False

    # Check validation_passed
    if expect_val is not None and r["validation_passed"] is not None:
        if r["validation_passed"] != expect_val:
            status_parts.append(f"FAIL(val={r['validation_passed']} expected={expect_val})")
            passed = False
            assertions_ok = False

    status = "PASS" if passed else " | ".join(status_parts)
    silver_str = f"Silver={r['silver']}%" if r["silver"] is not None else ""
    print(f"{status:<30} {silver_str}  ({r['elapsed_s']}s)")
    return r


try:
    # =========================================================================
    # A. REAL DATA TEST
    # =========================================================================
    print("--- A: Real Repo Data ---")
    run_case(
        sub_dir=REFERENCE_DIR,
        out_dir=os.path.join(tmpdir, "out_real"),
        label="REAL: constantKtransModel",
        expect_exit=0,
        expect_val=True,
    )

    # =========================================================================
    # B. SYNTHETIC (10 KTRANS VALUES)
    # =========================================================================
    print("\n--- B: 10 Synthetic Submissions (varying Ktrans) ---")
    KTRANS_VALUES = [0.005, 0.01, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    for i, kval in enumerate(KTRANS_VALUES):
        sub = _make_submission(tmpdir, f"syn_k{kval}", ktrans_val=kval)
        run_case(
            sub_dir=sub,
            out_dir=os.path.join(tmpdir, f"out_syn_{i}"),
            label=f"Ktrans={kval:.3f} min^-1",
            expect_exit=0,
            expect_val=True,
        )

    # =========================================================================
    # C. EDGE CASES
    # =========================================================================
    print("\n--- C: Edge Cases ---")

    # C1: Missing one file (Clinical_P5_Visit2)
    sub = _make_submission(tmpdir, "miss1", missing=[("Clinical", 5, 2)])
    run_case(sub, os.path.join(tmpdir, "out_miss1"),
             "C1: Missing Clinical_P5_Visit2",
             expect_exit="nonzero", expect_val=False)

    # C2: Missing multiple files (all Synthetic)
    sub = _make_submission(tmpdir, "miss_syn",
                           missing=[("Synthetic", 1, 1), ("Synthetic", 1, 2),
                                    ("Synthetic", 2, 1), ("Synthetic", 2, 2)])
    run_case(sub, os.path.join(tmpdir, "out_miss_syn"),
             "C2: All Synthetic files missing",
             expect_exit="nonzero", expect_val=False)

    # C3: Compressed .nii.gz files (should fail validation)
    sub = _make_submission(tmpdir, "gz_files", use_gz=True)
    run_case(sub, os.path.join(tmpdir, "out_gz"),
             "C3: All files as .nii.gz (not supported)",
             expect_exit="nonzero", expect_val=False)

    # C4: Wrong filename format
    sub_dir = os.path.join(tmpdir, "bad_names")
    os.makedirs(sub_dir)
    for name in ["patient1_v1.nii", "Ktrans_P1.nii", "CLINICAL_p1_v1.nii",
                 "Synthetic_1_Visit_1.nii", "Clinical_Patient1_Visit1.nii"]:
        _make_nifti(np.zeros(ref_shape, np.float32), os.path.join(sub_dir, name))
    run_case(sub_dir, os.path.join(tmpdir, "out_bad_names"),
             "C4: All wrong filename formats",
             expect_exit="nonzero", expect_val=False)

    # C5: Mix of valid + invalid filenames
    sub = _make_submission(tmpdir, "mix_names")
    # Add a badly-named extra file
    _make_nifti(np.zeros(ref_shape, np.float32),
                os.path.join(sub, "badname_extra.nii"))
    run_case(sub, os.path.join(tmpdir, "out_mix"),
             "C5: Valid names + 1 extra bad file",
             expect_exit="nonzero", expect_val=False)

    # C6: Out-of-range patient number (Clinical_P9)
    sub = _make_submission(tmpdir, "out_range")
    _make_nifti(np.zeros(ref_shape, np.float32),
                os.path.join(sub, "Clinical_P9_Visit1.nii"))
    run_case(sub, os.path.join(tmpdir, "out_range"),
             "C6: Clinical_P9 (out of range)",
             expect_exit="nonzero", expect_val=False)

    # C7: Empty directory
    empty_dir = os.path.join(tmpdir, "empty")
    os.makedirs(empty_dir)
    run_case(empty_dir, os.path.join(tmpdir, "out_empty"),
             "C7: Empty submission directory",
             expect_exit="nonzero", expect_val=False)

    # C8: Duplicate patient-visit (Clinical_P1_Visit1 as both .nii and .nii.gz)
    sub = _make_submission(tmpdir, "dup_pv")
    # Add .nii.gz version of P1_V1 (creating a duplicate)
    _make_nifti(np.zeros(ref_shape, np.float32),
                os.path.join(sub, "Clinical_P1_Visit1.nii.gz"))
    run_case(sub, os.path.join(tmpdir, "out_dup"),
             "C8: Duplicate C_P1_V1 (.nii + .nii.gz)",
             expect_exit="nonzero", expect_val=False)

    # C9: NaN values in submission NiFTI
    nan_data = np.full(ref_shape, float('nan'), dtype=np.float32)
    nan_override = {f"Clinical_P{p}_Visit{v}": nan_data
                    for p in range(1, 9) for v in (1, 2)}
    nan_override.update({f"Synthetic_P{p}_Visit{v}": nan_data
                         for p in (1, 2) for v in (1, 2)})
    sub = _make_submission(tmpdir, "nan_vals", data_override=nan_override)
    run_case(sub, os.path.join(tmpdir, "out_nan"),
             "C9: NaN values in all NIfTIs (validator catches)",
             expect_exit="nonzero", expect_val=False)  # validator now detects all-NaN arrays

    # C10: All-zero Ktrans (should score; silver may be null if scoring returns nan)
    sub = _make_submission(tmpdir, "zeros", ktrans_val=0.0)
    run_case(sub, os.path.join(tmpdir, "out_zeros"),
             "C10: All-zero Ktrans values",
             expect_exit=0, expect_val=True)  # silver may be null (0/0 = NaN in scorer)

    # C11: Very large Ktrans (1000x expected)
    sub = _make_submission(tmpdir, "huge", ktrans_val=100.0)
    run_case(sub, os.path.join(tmpdir, "out_huge"),
             "C11: Very large Ktrans (100.0 min^-1)",
             expect_exit=0, expect_val=True)

    # C12: Negative Ktrans values
    neg_data = np.full(ref_shape, -0.5, dtype=np.float32)
    neg_override = {f"Clinical_P{p}_Visit{v}": neg_data
                    for p in range(1, 9) for v in (1, 2)}
    neg_override.update({f"Synthetic_P{p}_Visit{v}": neg_data
                         for p in (1, 2) for v in (1, 2)})
    sub = _make_submission(tmpdir, "neg_vals", data_override=neg_override)
    run_case(sub, os.path.join(tmpdir, "out_neg"),
             "C12: Negative Ktrans values (-0.5)",
             expect_exit=0, expect_val=True)

    # C13: Non-existent submission directory
    run_case("/nonexistent/submission/path",
             os.path.join(tmpdir, "out_nodir"),
             "C13: Non-existent submission dir",
             expect_exit="nonzero")

    # C14: Non-existent ground truth directory
    sub = _make_submission(tmpdir, "real_sub_bad_gt")
    r14_dir = os.path.join(tmpdir, "out_bad_gt")
    os.makedirs(r14_dir, exist_ok=True)
    idx = len(results) + 1
    print(f"[{idx:02d}] {'C14: Non-existent ground truth dir':<55}", end=" ... ", flush=True)
    cmd = [
        PYTHON, PIPELINE_SCRIPT,
        "--submission_dir",   sub,
        "--ground_truth_dir", "/nonexistent/ground/truth",
        "--output_dir",       r14_dir,
        "--scoring_script",   SCORING_SCRIPT,
        "--masks_dir",        MASKS_DIR,
    ]
    t0 = time.time()
    ret14 = subprocess.run(cmd, capture_output=True, text=True)
    elapsed14 = round(time.time() - t0, 1)
    r14 = {"label": "C14: Non-existent GT dir", "exit_code": ret14.returncode,
           "elapsed_s": elapsed14, "validation_passed": None,
           "accuracy_score": None, "silver": None,
           "json_exists": False, "pdf_exists": False, "stderr": ""}
    results.append(r14)
    if ret14.returncode == 0:
        print(f"FAIL(exit=0 expected=nonzero)")
        assertions_ok = False
    else:
        print(f"PASS                           ({elapsed14}s)")

    # C15: Special characters in team name (should be sanitized)
    sub = _make_submission(tmpdir, "special_team")
    run_case(sub, os.path.join(tmpdir, "out_special_team"),
             "C15: Special chars in team name",
             expect_exit=0, expect_val=True,
             extra_args=["--team_name", "my/team & name!"])

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print()
    print("=" * 100)
    print(f"{'#':>3}  {'Submission':<45}  {'Exit':>4}  {'Val':>5}  {'Acc':>6}  {'Silver':>7}  {'JSON':>4}  {'PDF':>4}")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        val    = ("PASS" if r["validation_passed"] else
                  ("FAIL" if r["validation_passed"] is False else " -- "))
        acc    = f"{r['accuracy_score']:.3f}" if r["accuracy_score"] is not None else " N/A"
        silver = f"{r['silver']:.1f}%" if r["silver"] is not None else "  N/A"
        json_  = "YES" if r["json_exists"] else " NO"
        pdf_   = "YES" if r["pdf_exists"]  else " NO"
        print(f"{i:>3}  {r['label']:<45}  {r['exit_code']:>4}  {val:>5}  {acc:>6}  {silver:>7}  {json_:>4}  {pdf_:>4}")
    print("=" * 100)

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    # Specific score sanity checks
    silver_scores = [r["silver"] for r in results[1:11] if r["silver"] is not None]
    if silver_scores:
        peak = max(silver_scores)
        low  = min(silver_scores)
        if peak <= low:
            print("[Sanity] FAIL: Silver scores don't vary with Ktrans (all equal)")
            assertions_ok = False
        else:
            print(f"[Sanity] PASS: Silver scores vary from {low:.1f}% to {peak:.1f}% across Ktrans values")

    # Real data score check
    real_silver = results[0]["silver"]
    if real_silver is not None:
        print(f"[Real data] Silver={real_silver}%  (constantKtransModel reference)")
    else:
        print("[Real data] FAIL: No score for constantKtransModel")
        assertions_ok = False

    print()
    if assertions_ok:
        print("ALL ASSERTIONS PASSED")
    else:
        print("SOME ASSERTIONS FAILED -- review table above")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[Cleanup] Temp directory removed.")
