#!/usr/bin/env python3
"""
scorer_wrapper.py
=================
Wraps challengeScoring.py so it accepts CLI arguments instead of hardcoded paths.
Saves numeric results to results.json in the output directory.

Strategy: uses runpy.run_path() with a patched globals dict to override the
module-level path/config variables in challengeScoring.py — zero changes to
the original scoring math.

Usage:
    python scorer_wrapper.py \\
        --submission_dir  /app/submission \\
        --ground_truth_dir /app/ground_truth \\
        --output_dir      /app/output \\
        [--team_name      my_team]
"""

import argparse
import json
import os
import re
import runpy
import shutil
import sys
import tempfile
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_score_file(score_file: str) -> dict:
    """Parse scoringOutputs/OSIPI_score_tabular.txt into a dict."""
    scores = {}
    if not os.path.isfile(score_file):
        return scores
    with open(score_file, 'r') as f:
        lines = f.readlines()

    def _safe_float(s: str):
        """Convert a string to float; return None for 'nan', 'inf', or bad values."""
        s = s.strip().lower()
        if s in ('nan', 'inf', '-inf', '', 'none', 'n/a'):
            return None
        try:
            v = float(s)
            import math
            return None if (math.isnan(v) or math.isinf(v)) else v
        except ValueError:
            return None

    # Header line is first; data lines follow
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split('\t')]
        if len(parts) >= 6 and parts[0]:
            team = parts[0]
            scores[team] = {
                "accuracy_score":          _safe_float(parts[1]),
                "repeatability_score":     _safe_float(parts[2]),
                "reproducibility_score":   _safe_float(parts[3]),
                "osipi_score_silver":      _safe_float(parts[4]),
                "osipi_score_gold":        _safe_float(parts[5]),
            }
    return scores


def _parse_ktrans_table(ktrans_file: str, team_name: str) -> dict:
    """
    Parse scoringOutputs/TMROI_Ktrans.txt and extract per-label entry values.

    File format (tab-separated, each cell may have surrounding spaces):
        Team         C1_v1  C1_v2  ...  S2_v2
        gt           NA     NA     ...  0.069
        {team}_entry 0.1    0.1    ...  0.1
        {team}_sd    ...
        {team}_repro ...
    """
    result: dict = {}
    if not os.path.isfile(ktrans_file):
        return result

    labels: list[str] = []
    entry_values: list[str] = []

    with open(ktrans_file, 'r') as f:
        for line in f:
            # Strip and split on tab; strip each cell too
            parts = [p.strip() for p in line.split('\t')]
            if not parts or not parts[0]:
                continue
            row_key = parts[0]
            if row_key == 'Team':
                labels = [p for p in parts[1:] if p]   # column headers
            elif row_key == f'{team_name}_entry':
                entry_values = parts[1:]                # submission values

    for label, val in zip(labels, entry_values):
        try:
            result[label] = float(val)
        except ValueError:
            result[label] = val.strip() if val else val

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_scoring(
    submission_dir: str,
    ground_truth_dir: str,
    output_dir: str,
    team_name: str = "submission",
    scoring_script: Optional[str] = None,
) -> dict:
    """
    Run the scoring pipeline and return the results dict that was written to
    output_dir/results.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve ALL paths to absolute BEFORE any chdir() calls
    submission_dir   = os.path.abspath(submission_dir)
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    output_dir       = os.path.abspath(output_dir)

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not os.path.isdir(submission_dir):
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")
    if not os.path.isdir(ground_truth_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    # Sanitise team_name: must be a valid directory component (no path separators, slashes, etc.)
    import re as _re_sanitize
    safe_team = _re_sanitize.sub(r'[^\w\-]', '_', team_name).strip('_') or 'submission'
    if safe_team != team_name:
        print(f"[Scorer] WARNING: team_name '{team_name}' sanitized to '{safe_team}'")
    team_name = safe_team

    # Locate challengeScoring.py
    if scoring_script is None:
        here = os.path.dirname(os.path.abspath(__file__))
        scoring_script = os.path.join(here, "challengeScoring.py")
    scoring_script = os.path.abspath(scoring_script)   # MUST be absolute before chdir
    if not os.path.isfile(scoring_script):
        raise FileNotFoundError(f"challengeScoring.py not found at: {scoring_script}")
    if not scoring_script.endswith('.py'):
        raise ValueError(f"scoring_script must be a Python .py file: {scoring_script}")

    # ── Build a temporary working directory mimicking the expected layout ──────
    workdir = tempfile.mkdtemp(prefix="osipi_scoring_")
    try:
        scoring_outputs_dir = os.path.join(workdir, "scoringOutputs")
        os.makedirs(scoring_outputs_dir, exist_ok=True)

        entry_dir     = os.path.join(workdir, "entryDirectories")
        os.makedirs(entry_dir, exist_ok=True)

        # challengeScoring.py walks: entryDirectories/<team>/  and  entryDirectories/<team>_neutral/
        # We point both to the same submission_dir (neutral = reproducibility data; same dir = fine for PoC)
        def _link_or_copy(src, dst):
            """Symlink on Linux/Mac; on Windows try junction then copytree."""
            src = os.path.abspath(src)
            if os.name == 'nt':
                try:
                    import subprocess as _sp
                    result = _sp.run(
                        ['cmd', '/c', 'mklink', '/J', dst, src],
                        stdout=_sp.DEVNULL, stderr=_sp.PIPE
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr.decode(errors='replace'))
                except Exception as je:
                    print(f"[Scorer] Junction failed ({je}), falling back to copytree")
                    if os.path.exists(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(src, dst)
            else:
                os.symlink(src, dst)

        sub_link         = os.path.join(entry_dir, team_name)
        sub_neutral_link = os.path.join(entry_dir, f"{team_name}_neutral")
        _link_or_copy(submission_dir, sub_link)
        _link_or_copy(submission_dir, sub_neutral_link)

        # DROKtransNifti  →  ground_truth_dir
        dro_link = os.path.join(workdir, "DROKtransNifti")
        _link_or_copy(ground_truth_dir, dro_link)

        # Masks — check OSIPI_MASKS_DIR env var first (set by run_pipeline.py)
        masks_link = os.path.join(workdir, "Masks")
        env_masks  = os.environ.get("OSIPI_MASKS_DIR", "")
        if env_masks and os.path.isdir(env_masks):
            masks_src = os.path.abspath(env_masks)
        else:
            # Auto-detect: look next to challengeScoring.py, then parent/Scoring/Masks
            script_dir = os.path.dirname(scoring_script)
            masks_src  = os.path.join(script_dir, "Masks")
            if not os.path.isdir(masks_src):
                alt = os.path.normpath(os.path.join(script_dir, "..", "Scoring", "Masks"))
                if os.path.isdir(alt):
                    masks_src = alt
                else:
                    raise FileNotFoundError(
                        f"Cannot find Masks directory. Searched:\n  {masks_src}\n  {alt}\n"
                        f"Set OSIPI_MASKS_DIR env var or use --masks_dir."
                    )
        _link_or_copy(masks_src, masks_link)

        # ── Create a patched copy of challengeScoring.py ─────────────────────
        # Strategy:
        #  1. Suppress original module-level config assignments (FIRST, so
        #     count=1 targets only the originals, not anything we inject).
        #  2. Replace the now-commented list_dir line with our full override block.
        import re as _re

        with open(scoring_script, 'r', encoding='utf-8') as f:
            script_src = f.read()

        patched_src = script_src

        # Step 1: Comment out original config variable assignments.
        # These lines are identifiable by their variable names at start of line.
        for var in ('plotting', 'list_dir', 'entry_list', 'entry_list_rep',
                    'dro_dir', 'mask_dir'):
            patched_src = _re.sub(
                rf'^({var}\s*=)',
                r'# [WRAPPER-SUPPRESSED] \1',
                patched_src,
                count=1,
                flags=_re.MULTILINE
            )

        # Step 2: Inject our override block in place of the suppressed list_dir line.
        # CRITICAL: challengeScoring.py accesses files by index (c_fnames[i*2]),
        # which requires files to be sorted alphabetically by patient and visit.
        # os.walk() order is NOT guaranteed on Windows, so we pre-sort via override.
        #
        # The sorted() call is injected into the script so that after the os.walk
        # collects all filenames, they get sorted before index-based access.
        # We do this by appending sort calls after the file-collection loops.
        sorted_fix = (
            "\n    # [WRAPPER] Sort file lists so index-based access is deterministic\n"
            "    c_fnames = sorted(c_fnames)\n"
            "    s_fnames = sorted(s_fnames)\n"
            "    all_fnames = sorted(all_fnames)\n"
            "    c_mask_fnames = sorted(c_mask_fnames)\n"
            "    s_mask_fnames = sorted(s_mask_fnames)\n"
            "    all_mask_fnames = sorted(all_mask_fnames)\n"
        )
        # Find the line that starts '    clinical_P = np.arange' (4-space indent inside for loop)
        patched_src = patched_src.replace(
            '    clinical_P = np.arange',
            sorted_fix + '    clinical_P = np.arange',
            1
        )


        override_block = (
            "\n# -- PIPELINE WRAPPER OVERRIDES --\n"
            "list_dir       = 'entryDirectories'\n"
            "entry_list     = " + repr([team_name]) + "\n"
            "entry_list_rep = " + repr([team_name]) + "\n"
            "dro_dir        = 'DROKtransNifti'\n"
            "mask_dir       = 'Masks'\n"
            "plotting       = 'Off'\n"
            "# -- END WRAPPER OVERRIDES --\n"
        )
        patched_src = patched_src.replace(
            "# [WRAPPER-SUPPRESSED] list_dir =",
            override_block + "# [WRAPPER-SUPPRESSED] list_dir =",
            1  # only first occurrence
        )

        patched_script = os.path.join(workdir, "challengeScoring_patched.py")
        with open(patched_script, 'w', encoding='utf-8') as f:
            f.write(patched_src)

        # ── Run the patched script ────────────────────────────────────────────
        orig_dir = os.getcwd()
        os.chdir(workdir)
        try:
            print(f"[Scorer] Running patched challengeScoring from workdir: {workdir}")
            runpy.run_path(patched_script, run_name="__main__")
        finally:
            os.chdir(orig_dir)



        # ── Validate scoring outputs ──────────────────────────────────────────
        tabular_path = os.path.join(scoring_outputs_dir, "OSIPI_score_tabular.txt")
        ktrans_path  = os.path.join(scoring_outputs_dir, "TMROI_Ktrans.txt")

        if not os.path.isfile(tabular_path):
            raise RuntimeError(
                f"challengeScoring.py did not produce expected output: {tabular_path}\n"
                f"Check the scoring script ran to completion without errors."
            )

        scores     = _parse_score_file(tabular_path)
        team_scores = scores.get(team_name, {})

        if not team_scores:
            # Dump any available output for diagnosis
            with open(tabular_path) as _f:
                _contents = _f.read()
            raise RuntimeError(
                f"Scores for team '{team_name}' not found in tabular output.\n"
                f"Available teams: {list(scores.keys())}\n"
                f"Tabular file contents:\n{_contents}"
            )

        ktrans_table = _parse_ktrans_table(ktrans_path, team_name)

        results = {
            "team":                    team_name,
            "accuracy_score":          team_scores.get("accuracy_score"),
            "repeatability_score":     team_scores.get("repeatability_score"),
            "reproducibility_score":   team_scores.get("reproducibility_score"),
            "osipi_score_silver":      team_scores.get("osipi_score_silver"),
            "osipi_score_gold":        team_scores.get("osipi_score_gold"),
            "per_patient_ktrans":      ktrans_table,
            "validation_passed":       True,   # set by run_pipeline.py
            "raw_outputs_dir":         scoring_outputs_dir,
        }

        # Copy all raw text outputs to output_dir for reference
        for fname in os.listdir(scoring_outputs_dir):
            shutil.copy2(
                os.path.join(scoring_outputs_dir, fname),
                os.path.join(output_dir, fname)
            )

        # Write results.json — use custom serializer to handle NaN/Inf as null
        def _json_safe(obj):
            import math
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        results_json_path = os.path.join(output_dir, "results.json")
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2, default=_json_safe)
        print(f"[Scorer] Results written to: {results_json_path}")

        return results

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSIPI DCE-MRI Scoring Wrapper")
    parser.add_argument("--submission_dir",   required=True,  help="Path to submission NIfTI files")
    parser.add_argument("--ground_truth_dir", required=True,  help="Path to DROKtransNifti directory")
    parser.add_argument("--output_dir",       required=True,  help="Path to write results.json and text outputs")
    parser.add_argument("--team_name",        default="submission", help="Team/entry name (default: submission)")
    parser.add_argument("--scoring_script",   default=None,   help="Explicit path to challengeScoring.py")
    args = parser.parse_args()

    try:
        run_scoring(
            submission_dir=args.submission_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_dir=args.output_dir,
            team_name=args.team_name,
            scoring_script=args.scoring_script,
        )
    except FileNotFoundError as e:
        print(f"[Scorer] PATH ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[Scorer] SCORING ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"[Scorer] UNEXPECTED ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
