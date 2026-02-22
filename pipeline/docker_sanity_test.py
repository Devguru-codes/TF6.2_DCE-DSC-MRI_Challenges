#!/usr/bin/env python3
"""
docker_sanity_test.py
=====================
Comprehensive end-to-end Docker sanity test for osipi-eval-pipeline.
Creates real NIfTI submissions, runs them through the Docker image,
and validates every output.

Run from repo root:
    python pipeline/docker_sanity_test.py
"""

import json, os, shutil, subprocess, sys, tempfile, time
import nibabel as nib
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.normpath(os.path.join(HERE, ".."))
REFERENCE   = os.path.join(REPO_ROOT, "Scoring", "entryDirectories", "constantKtransModel")
IMAGE       = "osipi-eval-pipeline"
CLINICAL    = 8
SYNTHETIC   = 2
VISITS      = 2

# Load reference geometry
ref = nib.load(os.path.join(REFERENCE, "Clinical_P1_Visit1.nii"))
REF_AFFINE  = ref.affine
REF_HEADER  = ref.header.copy()
REF_SHAPE   = ref.shape

def nifti(data, path):
    nib.save(nib.Nifti1Image(data.astype(np.float32), REF_AFFINE, REF_HEADER), path)

def make_submission(base, name, ktrans=0.1, missing=(), extra=(), use_gz=False,
                    overrides=None):
    d = os.path.join(base, name); os.makedirs(d, exist_ok=True)
    overrides = overrides or {}
    ext = ".nii.gz" if use_gz else ".nii"
    for p in range(1, CLINICAL+1):
        for v in range(1, VISITS+1):
            fn = f"Clinical_P{p}_Visit{v}"
            if ("Clinical", p, v) in missing: continue
            nifti(overrides.get(fn, np.full(REF_SHAPE, ktrans, np.float32)),
                  os.path.join(d, fn + ext))
    for p in range(1, SYNTHETIC+1):
        for v in range(1, VISITS+1):
            fn = f"Synthetic_P{p}_Visit{v}"
            if ("Synthetic", p, v) in missing: continue
            nifti(overrides.get(fn, np.full(REF_SHAPE, ktrans, np.float32)),
                  os.path.join(d, fn + ext))
    for ef in extra:
        nifti(np.zeros(REF_SHAPE, np.float32), os.path.join(d, ef))
    return d

def docker_run(sub_dir, out_dir, extra_docker_args=None):
    os.makedirs(out_dir, exist_ok=True)
    # Docker needs forward slashes for volume mounts on Windows
    sub_fwd  = sub_dir.replace("\\", "/")
    out_fwd  = out_dir.replace("\\", "/")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{sub_fwd}:/app/submission:ro",
        "-v", f"{out_fwd}:/app/output",
    ]
    if extra_docker_args:
        cmd += extra_docker_args
    cmd.append(IMAGE)
    t0 = time.time()
    ret = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    result_path = os.path.join(out_dir, "results.json")
    pdf_path    = os.path.join(out_dir, "Evaluation_Report.pdf")
    data = {}
    if os.path.isfile(result_path):
        try:
            with open(result_path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {"_json_error": True}

    return {
        "exit":     ret.returncode,
        "elapsed":  elapsed,
        "val":      data.get("validation_passed"),
        "acc":      data.get("accuracy_score"),
        "silver":   data.get("osipi_score_silver"),
        "json_ok":  os.path.isfile(result_path) and "_json_error" not in data,
        "pdf_ok":   os.path.isfile(pdf_path),
        "stdout":   ret.stdout[-400:] if ret.stdout else "",
        "stderr":   ret.stderr[-400:] if ret.stderr else "",
    }

# ─── Test runner ──────────────────────────────────────────────────────────────
PASS_MARK = "\u2714"
FAIL_MARK = "\u2718"
results = []
assertions_ok = True

def run_case(label, sub_dir, out_dir, expect_exit, expect_val=None,
             expect_silver_range=None, expect_json=None, expect_pdf=None,
             extra_docker_args=None):
    global assertions_ok
    idx = len(results) + 1
    print(f"[{idx:02d}] {label[:58]:<58}", end=" ... ", flush=True)
    r = docker_run(sub_dir, out_dir, extra_docker_args)

    fails = []
    # Exit code
    if expect_exit == "nonzero":
        if r["exit"] == 0: fails.append(f"exit=0 (expected nonzero)")
    elif expect_exit is not None:
        if r["exit"] != expect_exit: fails.append(f"exit={r['exit']} (expected {expect_exit})")
    # Validation flag
    if expect_val is not None and r["val"] is not None:
        if r["val"] != expect_val: fails.append(f"val={r['val']} (expected {expect_val})")
    # Silver range
    if expect_silver_range and r["silver"] is not None:
        lo, hi = expect_silver_range
        if not (lo <= r["silver"] <= hi):
            fails.append(f"silver={r['silver']} not in [{lo},{hi}]")
    # JSON/PDF presence
    if expect_json is not None:
        if r["json_ok"] != expect_json: fails.append(f"json_ok={r['json_ok']} (expected {expect_json})")
    if expect_pdf is not None:
        if r["pdf_ok"] != expect_pdf: fails.append(f"pdf_ok={r['pdf_ok']} (expected {expect_pdf})")

    ok = not fails
    if not ok:
        assertions_ok = False

    silver_str = f"Silver={r['silver']}%" if r["silver"] is not None else ""
    status     = "PASS" if ok else "FAIL: " + "; ".join(fails)
    print(f"{status:<36} {silver_str} ({r['elapsed']}s)")
    results.append({**r, "label": label, "ok": ok})
    return r


# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  OSIPI Docker Image — Comprehensive End-to-End Sanity Check")
print("=" * 80)
print()

# Verify image exists
check = subprocess.run(["docker", "image", "inspect", IMAGE],
                       capture_output=True, text=True)
if check.returncode != 0:
    print(f"[ERROR] Docker image '{IMAGE}' not found. Build it first.")
    sys.exit(2)
print(f"[OK] Image '{IMAGE}' found.\n")

tmpdir = tempfile.mkdtemp(prefix="osipi_docker_sanity_")
print(f"[Setup] Temp dir: {tmpdir}\n")

try:
    # ── A: REAL DATA ─────────────────────────────────────────────────────────
    print("--- A: Real Data ---")
    run_case("REAL: constantKtransModel",
             REFERENCE, os.path.join(tmpdir, "out_real"),
             expect_exit=0, expect_val=True,
             expect_silver_range=(50, 75),
             expect_json=True, expect_pdf=True)

    # ── B: SYNTHETIC (3 Ktrans values) ───────────────────────────────────────
    print("\n--- B: Synthetic Submissions ---")
    for kval in [0.05, 0.10, 0.30]:
        sub = make_submission(tmpdir, f"syn_{kval}", ktrans=kval)
        run_case(f"Synthetic Ktrans={kval:.2f}",
                 sub, os.path.join(tmpdir, f"out_syn_{kval}"),
                 expect_exit=0, expect_val=True,
                 expect_json=True, expect_pdf=True)

    # ── C: EDGE CASES ────────────────────────────────────────────────────────
    print("\n--- C: Edge Cases ---")

    # C1: Missing 1 required file
    sub = make_submission(tmpdir, "miss1", missing=[("Clinical", 5, 2)])
    run_case("C1: Missing Clinical_P5_Visit2",
             sub, os.path.join(tmpdir, "out_c1"),
             expect_exit="nonzero", expect_val=False,
             expect_json=False, expect_pdf=False)

    # C2: All synthetic missing
    sub = make_submission(tmpdir, "miss_syn",
                          missing=[(t,p,v) for t in ["Synthetic"] for p in [1,2] for v in [1,2]])
    run_case("C2: All Synthetic files missing",
             sub, os.path.join(tmpdir, "out_c2"),
             expect_exit="nonzero", expect_val=False,
             expect_json=False, expect_pdf=False)

    # C3: Compressed .nii.gz files
    sub = make_submission(tmpdir, "gz_files", use_gz=True)
    run_case("C3: All .nii.gz (validator rejects)",
             sub, os.path.join(tmpdir, "out_c3"),
             expect_exit="nonzero", expect_val=False)

    # C4: Entirely wrong filenames
    bad_dir = os.path.join(tmpdir, "bad_names")
    os.makedirs(bad_dir)
    for n in ["patient1.nii", "Ktrans_map.nii", "scan_v1.nii"]:
        nifti(np.zeros(REF_SHAPE, np.float32), os.path.join(bad_dir, n))
    run_case("C4: All wrong filename formats",
             bad_dir, os.path.join(tmpdir, "out_c4"),
             expect_exit="nonzero", expect_val=False,
             expect_json=False, expect_pdf=False)

    # C5: Valid files + 1 extra badly-named file
    sub = make_submission(tmpdir, "mix_names",
                          extra=["badname_extra.nii"])
    run_case("C5: Valid + 1 extra badly-named file",
             sub, os.path.join(tmpdir, "out_c5"),
             expect_exit="nonzero", expect_val=False,
             expect_json=True, expect_pdf=True)

    # C6: Out-of-range patient number (Clinical_P9)
    sub = make_submission(tmpdir, "out_range")
    nifti(np.zeros(REF_SHAPE, np.float32),
          os.path.join(sub, "Clinical_P9_Visit1.nii"))
    run_case("C6: Clinical_P9 (out of range)",
             sub, os.path.join(tmpdir, "out_c6"),
             expect_exit="nonzero", expect_val=False,
             expect_json=True, expect_pdf=True)

    # C7: Empty directory
    empty = os.path.join(tmpdir, "empty")
    os.makedirs(empty)
    run_case("C7: Empty submission directory",
             empty, os.path.join(tmpdir, "out_c7"),
             expect_exit="nonzero", expect_val=False,
             expect_json=False, expect_pdf=False)

    # C8: Duplicate patient-visit (.nii + .nii.gz)
    sub = make_submission(tmpdir, "dup_pv")
    nifti(np.zeros(REF_SHAPE, np.float32),
          os.path.join(sub, "Clinical_P1_Visit1.nii.gz"))
    run_case("C8: Duplicate P1_V1 (.nii + .nii.gz)",
             sub, os.path.join(tmpdir, "out_c8"),
             expect_exit="nonzero", expect_val=False,
             expect_json=True, expect_pdf=True)

    # C9: All-NaN NIfTI values (validator must catch BEFORE scorer)
    nan_data = np.full(REF_SHAPE, float('nan'), np.float32)
    ov = {f"Clinical_P{p}_Visit{v}": nan_data
          for p in range(1,9) for v in (1,2)}
    ov.update({f"Synthetic_P{p}_Visit{v}": nan_data
               for p in (1,2) for v in (1,2)})
    sub = make_submission(tmpdir, "nan_vals", overrides=ov)
    run_case("C9: All-NaN NIfTIs (validator catches)",
             sub, os.path.join(tmpdir, "out_c9"),
             expect_exit="nonzero", expect_val=False,
             expect_json=False, expect_pdf=False)

    # C10: All-zero Ktrans (scores but silver may be null)
    sub = make_submission(tmpdir, "zeros", ktrans=0.0)
    run_case("C10: All-zero Ktrans (silver may be N/A)",
             sub, os.path.join(tmpdir, "out_c10"),
             expect_exit=0, expect_val=True,
             expect_json=True, expect_pdf=True)

    # C11: Very large Ktrans
    sub = make_submission(tmpdir, "huge", ktrans=100.0)
    run_case("C11: Very large Ktrans (100.0)",
             sub, os.path.join(tmpdir, "out_c11"),
             expect_exit=0, expect_val=True,
             expect_json=True, expect_pdf=True)

    # C12: Negative Ktrans
    neg = np.full(REF_SHAPE, -0.5, np.float32)
    ov  = {f"Clinical_P{p}_Visit{v}": neg for p in range(1,9) for v in (1,2)}
    ov.update({f"Synthetic_P{p}_Visit{v}": neg for p in (1,2) for v in (1,2)})
    sub = make_submission(tmpdir, "neg_vals", overrides=ov)
    run_case("C12: Negative Ktrans (-0.5)",
             sub, os.path.join(tmpdir, "out_c12"),
             expect_exit=0, expect_val=True,
             expect_json=True, expect_pdf=True)

    # C13: Special characters in team name (should be sanitised)
    sub = make_submission(tmpdir, "special_team")
    run_case("C13: Special chars in team name (my/team & name!)",
             sub, os.path.join(tmpdir, "out_c13"),
             expect_exit=0, expect_val=True,
             expect_json=True, expect_pdf=True,
             extra_docker_args=["--team_name", "my/team & name!"])

    # C14: results.json is valid JSON
    print("\n--- D: Output Integrity ---")
    out_real = os.path.join(tmpdir, "out_real")
    json_path = os.path.join(out_real, "results.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            parsed = json.load(f)
        required_keys = ["accuracy_score", "repeatability_score",
                         "osipi_score_silver", "validation_passed",
                         "validation_errors", "per_patient_ktrans"]
        missing_keys = [k for k in required_keys if k not in parsed]
        ok = not missing_keys and not any(
            v != v for v in [parsed.get("accuracy_score"), parsed.get("osipi_score_silver")]
            if v is not None)  # NaN check (NaN != NaN)
        idx = len(results) + 1
        print(f"[{idx:02d}] {'D1: results.json has all required keys':<58}", end=" ... ")
        if ok and not missing_keys:
            print("PASS")
        else:
            print(f"FAIL: missing keys {missing_keys}")
            assertions_ok = False
        results.append({"label": "D1: results.json structure", "ok": ok})

    # C15: PDF file is non-empty
    pdf_path = os.path.join(out_real, "Evaluation_Report.pdf")
    pdf_sz = os.path.getsize(pdf_path) if os.path.isfile(pdf_path) else 0
    ok = pdf_sz > 1000
    idx = len(results) + 1
    print(f"[{idx:02d}] {'D2: Evaluation_Report.pdf non-empty':<58}", end=" ... ")
    print(f"PASS ({pdf_sz} bytes)" if ok else f"FAIL (size={pdf_sz})")
    results.append({"label": "D2: PDF non-empty", "ok": ok})
    if not ok: assertions_ok = False

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  {'#':>2}  {'Case':<52}  {'Exit':>4}  {'Val':>5}  {'Silver':>8}  {'JSON':>4}  {'PDF':>4}  {'Result':>6}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        exit_s   = str(r.get("exit", "--"))
        val_s    = ("PASS" if r.get("val") else ("FAIL" if r.get("val") is False else "--"))
        silver_s = f"{r['silver']}%" if r.get("silver") is not None else "N/A"
        json_s   = "YES" if r.get("json_ok") else " NO"
        pdf_s    = "YES" if r.get("pdf_ok") else " NO"
        result_s = "PASS" if r.get("ok", True) else "FAIL"
        print(f"  {i:>2}  {r['label']:<52}  {exit_s:>4}  {val_s:>5}  {silver_s:>8}  {json_s:>4}  {pdf_s:>4}  {result_s:>6}")
    print("=" * 90)

    pass_count = sum(1 for r in results if r.get("ok", True))
    fail_count = len(results) - pass_count
    print()
    if assertions_ok:
        print(f"FINAL VERDICT: ALL {len(results)} CHECKS PASSED")
    else:
        print(f"FINAL VERDICT: {fail_count} FAILED / {len(results)} TOTAL -- review table above")

    sys.exit(0 if assertions_ok else 1)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("[Cleanup] Temp dir removed.")
