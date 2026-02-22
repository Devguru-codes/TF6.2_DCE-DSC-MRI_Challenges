#!/usr/bin/env python3
"""
validator.py
============
Validates NIfTI file submissions for the OSIPI DCE-MRI Challenge.

Naming convention (from README):
    Synthetic_P#_Visit#.nii   or   Clinical_P#_Visit#.nii

Usage (standalone):
    python validator.py /path/to/submission/dir
"""

import os
import re
import sys
from typing import Optional

# Expected file sets for one complete submission
# 2 synthetic patients x 2 visits + 8 clinical patients x 2 visits
EXPECTED_SYNTHETIC_PATIENTS = [1, 2]
EXPECTED_CLINICAL_PATIENTS = list(range(1, 9))    # 1-8
EXPECTED_VISITS = [1, 2]

# Regex: matches Synthetic_P1_Visit1.nii or Clinical_P3_Visit2.nii (case-insensitive)
# NOTE: .nii.gz is accepted by the challenge but challengeScoring.py only loads .nii
VALID_NAME_PATTERN = re.compile(
    r'^(Synthetic|Clinical)_P(\d+)_Visit(\d+)\.nii(\.gz)?$',
    re.IGNORECASE
)


def _collect_nifti_files(submission_dir: str) -> list:
    """Return all .nii / .nii.gz filenames (basename only) found in submission_dir (top-level only)."""
    files = []
    # Only look at top-level files — challengeScoring.py does os.walk which picks
    # up subdirs, but mixing nested and top-level causes duplicate-file issues.
    try:
        for f in os.listdir(submission_dir):
            full = os.path.join(submission_dir, f)
            if os.path.isfile(full):
                if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz'):
                    files.append(f)
    except PermissionError as e:
        return []
    return files


def _check_nifti_readable(filepath: str) -> Optional[str]:
    """
    Try to load the NIfTI header. Returns an error string on failure, None on success.
    Only imports nibabel if needed (avoids hard dep at module import time).
    """
    try:
        import nibabel as nib
        img = nib.load(filepath)
        # Force header + affine to load (detects truncated files)
        _ = img.header
        _ = img.affine
        # Also check shape makes sense
        if any(d == 0 for d in img.shape):
            return f"NIfTI image has zero-size dimension: shape={img.shape}"
        # Check for all-NaN or all-Inf arrays which crash the scoring script
        import numpy as np
        data = np.array(img.dataobj)
        if np.all(np.isnan(data)):
            return "NIfTI data array is entirely NaN — the scoring script cannot process this."
        if np.all(np.isinf(data)):
            return "NIfTI data array is entirely Inf — the scoring script cannot process this."
        return None
    except Exception as e:
        return str(e)


def validate_submission(submission_dir: str, check_readable: bool = True) -> tuple:
    """
    Validate the submission directory.

    Parameters
    ----------
    submission_dir : str
    check_readable : bool
        If True, attempt to load each NIfTI header to detect corruption.

    Returns
    -------
    (is_valid, errors)
        is_valid : bool
        errors   : list of human-readable error strings (empty if valid)
    """
    errors: list = []
    warnings: list = []

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not os.path.exists(submission_dir):
        return False, [f"Submission directory does not exist: {submission_dir}"]
    if not os.path.isdir(submission_dir):
        return False, [f"Submission path is not a directory: {submission_dir}"]

    nifti_files = _collect_nifti_files(submission_dir)

    if len(nifti_files) == 0:
        return False, ["No NIfTI (.nii / .nii.gz) files found in the submission directory (top level)."]

    # ── 1. Check every file matches the naming convention ─────────────────────
    gz_files = []
    invalid_names = []
    for f in nifti_files:
        if not VALID_NAME_PATTERN.match(f):
            invalid_names.append(f)
        elif f.lower().endswith('.nii.gz'):
            gz_files.append(f)

    for name in invalid_names:
        errors.append(
            f"Invalid filename: '{name}'. "
            f"Expected format: Synthetic_P#_Visit#.nii or Clinical_P#_Visit#.nii"
        )

    # Warn about .nii.gz — the scoring script only handles .nii natively
    for name in gz_files:
        warnings.append(
            f"File '{name}' uses .nii.gz format. "
            f"The scoring script expects plain .nii files. Please decompress."
        )
        # Treat .gz as a validation error to prevent scoring failures
        errors.append(
            f"File '{name}' is compressed (.nii.gz). The scoring script requires uncompressed .nii files."
        )

    # Build sets of (type, patient_num, visit_num) from valid-name files
    found: dict = {"Synthetic": set(), "Clinical": set()}
    for f in nifti_files:
        m = VALID_NAME_PATTERN.match(f)
        if m:
            ftype = m.group(1).capitalize()
            patient = int(m.group(2))
            visit = int(m.group(3))
            found[ftype].add((patient, visit))

    # ── 2. Check all expected Synthetic files are present ─────────────────────
    for p in EXPECTED_SYNTHETIC_PATIENTS:
        for v in EXPECTED_VISITS:
            if (p, v) not in found["Synthetic"]:
                errors.append(f"Missing file: Synthetic_P{p}_Visit{v}.nii")

    # ── 3. Check all expected Clinical files are present ──────────────────────
    for p in EXPECTED_CLINICAL_PATIENTS:
        for v in EXPECTED_VISITS:
            if (p, v) not in found["Clinical"]:
                errors.append(f"Missing file: Clinical_P{p}_Visit{v}.nii")

    # ── 4. Check for unexpected out-of-range patient numbers ──────────────────
    for p, v in found["Synthetic"]:
        if p not in EXPECTED_SYNTHETIC_PATIENTS:
            errors.append(f"Unexpected synthetic patient number: Synthetic_P{p}_Visit{v}.nii "
                          f"(expected P1 or P2)")
    for p, v in found["Clinical"]:
        if p not in EXPECTED_CLINICAL_PATIENTS:
            errors.append(f"Unexpected clinical patient number: Clinical_P{p}_Visit{v}.nii "
                          f"(expected P1-P8)")

    # ── 5. Check for duplicate patient-visit combinations (e.g. both .nii and .nii.gz) ──
    seen: dict = {}
    for f in nifti_files:
        m = VALID_NAME_PATTERN.match(f)
        if m:
            key = (m.group(1).capitalize(), int(m.group(2)), int(m.group(3)))
            if key in seen:
                errors.append(
                    f"Duplicate entry for {key[0]}_P{key[1]}_Visit{key[2]}: "
                    f"found both '{seen[key]}' and '{f}'"
                )
            else:
                seen[key] = f

    # ── 6. NIfTI readability check (only if no name errors so far) ────────────
    if check_readable and not errors:
        for f in nifti_files:
            full_path = os.path.join(submission_dir, f)
            err = _check_nifti_readable(full_path)
            if err:
                errors.append(f"Cannot read NIfTI file '{f}': {err}")

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("Usage: python validator.py <submission_dir>")
        sys.exit(1)

    sub_dir = sys.argv[1]
    print(f"[Validator] Scanning: {sub_dir}")
    valid, errs = validate_submission(sub_dir)

    if valid:
        print("[Validator] PASS - Submission is VALID -- all required files present and correctly named.")
        sys.exit(0)
    else:
        print(f"[Validator] FAIL - Submission is INVALID -- {len(errs)} error(s) found:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
