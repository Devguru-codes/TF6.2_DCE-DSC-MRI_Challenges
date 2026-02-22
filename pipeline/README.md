# OSIPI DCE-MRI Challenge Automated Evaluation Pipeline

This directory contains the production-ready automated evaluation pipeline for the OSIPI TF6.2 DCE-DSC-MRI Challenge. It wraps the core `challengeScoring.py` logic natively, adding robustness, file validation, custom execution paths without mutating the original files, and automated PDF report generation.

It is fully containerized with **Docker**, making it completely plug-and-play for evaluating new submissions in any environment.

---

## 🚀 Key Features Built
- **Zero-Touch Execution**: Injects pipeline variables dynamically into `challengeScoring.py` at runtime without altering its original source code.
- **Robust NIfTI Validation**: Before scoring, validates expected filenames (`Synthetic_P#_Visit#` and `Clinical_P#_Visit#`). Safely handles and rejects `.nii.gz` compression, all-NaN arrays, duplicate patients, or out-of-range IDs gracefully without crashing the scoring scripts.
- **Windows Compat Fixes**: Injects sorted file lists natively (resolves non-deterministic `os.walk` path sorting).
- **Graceful Error Handling**: Overrides Matplotlib dynamically to headless `Agg` mode. Translates `NaN` or `Inf` tabular outputs into standard `null` JSON scalars for cross-language compatibility.
- **Artifact Generation**: Outputs a structured `results.json` and cleanly formats a PDF `Evaluation_Report.pdf` displaying validation status and silver/gold scores.

---

## 🐳 Docker Deployment (Recommended)

The easiest way to consume this pipeline is via Docker. The ground truth (`DROKtransNifti`) and ROI `Masks` datasets are baked natively into the container image, meaning you only need to supply the individual team submission directory.

### 1. Build the Image
From the **root of the repository** (not inside the `pipeline` directory), build the Docker image. We use a `.dockerignore` file to keep the image slim and cacheable.

```bash
docker build -t osipi-eval-pipeline -f pipeline/Dockerfile .
```

### 2. Run the Image (Evaluate a Team)
Mount your local team's submission folder to `/app/submission` (Read-Only) and an empty local output folder to `/app/output`.

```bash
# Example for Linux / Mac
docker run --rm \
    -v $(pwd)/Scoring/entryDirectories/constantKtransModel:/app/submission:ro \
    -v $(pwd)/pipeline_output:/app/output \
    osipi-eval-pipeline

# Example for Windows (PowerShell)
docker run --rm `
    -v "${PWD}\Scoring\entryDirectories\constantKtransModel:/app/submission:ro" `
    -v "${PWD}\pipeline_output:/app/output" `
    osipi-eval-pipeline
```

*(Note: The pipeline automatically defaults to the baked-in reference datasets inside `/app/DROKtransNifti` and `/app/Masks`.)*

---

## 💻 Local Execution (Python Virtual Environment)

If developing locally or avoiding Docker, you can run the orchestrator manually. Requires **Python 3.9+**.

```bash
cd pipeline

# 1. Setup Environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# 2. Run the Pipeline Orchestrator
python run_pipeline.py \
    --submission_dir ../Scoring/entryDirectories/constantKtransModel \
    --ground_truth_dir ../Scoring/DROKtransNifti \
    --scoring_script ../Scoring/challengeScoring.py \
    --masks_dir ../Scoring/Masks \
    --output_dir ./test_output
```

---

## 📁 Output Artifacts

Running the pipeline deposits the following artifacts directly into the defined `--output_dir` (`/app/output` in Docker):

- `Evaluation_Report.pdf` - Visual summary showing the Validation badge (PASS/FAIL) and the OSIPI Silver metrics.
- `results.json` - Programmatically parseable file containing `validation_passed` flags, the raw silver/gold/accuracy numbers, and `per_patient_ktrans` objects.
- `OSIPI_score_tabular.txt` & `TMROI_Ktrans.txt` - Raw artifact dumps from `challengeScoring.py`.

---

## 🛡️ Built-in Error Handling & Resilience

The pipeline has been extensively audited to never crash during mid-evaluation. Bad submissions fail gracefully and generate clear feedback for the submitting team.

### `validator.py`
- **Missing or Misnamed Files**: Caught by regex matching the OSIPI naming spec. Lists every missing/invalid file explicitly.
- **`.nii.gz` Compression**: Identified and rejected to prevent downstream scoring script crashes.
- **Corrupted / Truncated NIfTIs**: `nibabel` load operations are wrapped to catch header corruption cleanly.
- **Empty or All-NaN Arrays**: Tensors are parsed eagerly using `numpy` before scoring. If an image only contains `NaN` or `Inf` (which crashes the core math scripts), the validator intercepts it.
- **Duplicate patient visits**: Detects if a team accidentally provides both `.nii` and `.nii.gz` for the same combination.

### `scorer_wrapper.py`
- **Path Traversal / Special Characters**: The passed `--team_name` is sanitized strictly to prevent bad directory bindings during Docker volume execution.
- **Math Overflow Outputs**: If the core scoring algorithm encounters a zero-divide and outputs `nan` or `inf` arrays, `_safe_float` traps this and translates it immediately to valid JSON `null` scalars, preventing JSON serialization errors.
- **Non-deterministic Directory Traversal**: `os.walk` yields results in different orders on Linux vs Windows (which ruins the sequential patient scoring in the original script). The wrapper natively injects a deterministic Python `sorted()` intercept into the core logic to align the files identically on all operating systems.

### `run_pipeline.py` & `report_gen.py`
- **Isolated Execution**: The three main steps (Validation, Scoring, Report Generation) are wrapped securely. E.g., if scoring fails on a bizarre edge case, the system still safely deposits the standard validation report showing what parameters were evaluated.
- **Console Encoding Output**: Enforces ASCII-only structure characters and `utf-8` stdout binding to prevent `UnicodeEncodeError` in Windows cp1252 environments.
- **Null Safety in PDF**: Renders missing scores or `null` gracefully as `"N/A"` gracefully.

---

## 🚦 Containerized Edge Case Audit

The pipeline's Docker image guarantees safe execution against the following edge cases:

1. **Real Data Submission** (PASS) → Safe extraction of metrics and fully populated PDF.
2. **Missing 1 File** (FAIL) → Graceful error `Missing file: Clinical_P5_Visit2.nii` printed, partial report generated.
3. **Missing All Synthetic Data** (FAIL) → Lists every missing `Synthetic_P#` file in the output JSON.
4. **`.nii.gz` compression** (FAIL) → Stopped by validator before entering the scoring sequence.
5. **Wrong Filename formats** (FAIL) → `patient1.nii` properly triggers OSIPI regex validation failures. 
6. **Out of Range Patients** (FAIL) → Extremely high unrequested labels (e.g. `Clinical_P9`) are ignored and scored partially.
7. **Empty Submission Dir** (FAIL) → Safely detects no NIfTI existence instead of blind execution.
8. **Duplicate Data** (FAIL) → If a team drops both `P1_Visit1.nii` and `P1_Visit1.nii.gz`, the script blocks execution.
9. **All NaN / Inf Corrupted Files** (FAIL) → Intercepted instantly preventing mathematical evaluation crashes.
10. **Zero / Negative Ktrans Values** (PASS) → The algorithms calculate and output scores cleanly despite extreme or anomalous underlying pixel volumes.
