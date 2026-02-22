# ==============================================================================
# OSIPI DCE-MRI Challenge -- Automated Evaluation Pipeline
# ==============================================================================
#
# BUILD (from repo root):
#   docker build -t osipi-eval-pipeline -f pipeline/Dockerfile .
#
# RUN (Linux / Mac -- from repo root):
#   docker run --rm \
#       -v $(pwd)/Scoring/entryDirectories/MY_TEAM:/app/submission:ro \
#       -v $(pwd)/pipeline/docker_output:/app/output \
#       osipi-eval-pipeline
#
# RUN (Windows PowerShell -- from repo root):
#   docker run --rm `
#       -v "${PWD}\Scoring\entryDirectories\MY_TEAM:/app/submission:ro" `
#       -v "${PWD}\pipeline\docker_output:/app/output" `
#       osipi-eval-pipeline
#
# OPTIONAL OVERRIDES (if you want to supply your own ground truth or masks):
#   -v /path/to/DROKtransNifti:/app/ground_truth:ro
#   -e OSIPI_MASKS_DIR=/app/my_masks  (and mount that dir too)
# ==============================================================================

FROM python:3.9-slim

# -- System dependencies -------------------------------------------------------
# libgomp1        : OpenMP threading used by nibabel / numpy BLAS
# libgl1-mesa-glx : required by matplotlib on some distros (Agg backend)
# libglib2.0-0    : transitive dependency of libgl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# -- Environment ---------------------------------------------------------------
ENV MPLBACKEND=Agg
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force UTF-8 so box-drawing chars work in any locale
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8

# -- Working directory ---------------------------------------------------------
WORKDIR /app

# -- Python dependencies -------------------------------------------------------
# Build context = repo root (docker build -f pipeline/Dockerfile .)
COPY pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -- Pipeline scripts ----------------------------------------------------------
COPY pipeline/validator.py      /app/validator.py
COPY pipeline/scorer_wrapper.py /app/scorer_wrapper.py
COPY pipeline/report_gen.py     /app/report_gen.py
COPY pipeline/run_pipeline.py   /app/run_pipeline.py

# -- Scoring script & reference data (baked into image) ------------------------
# challengeScoring.py  -->  /app/challengeScoring.py
# Ground-truth NIfTIs  -->  /app/DROKtransNifti/   (can be overridden via volume)
# Mask NIfTIs          -->  /app/Masks/
COPY Scoring/challengeScoring.py  /app/challengeScoring.py
COPY Scoring/DROKtransNifti/      /app/DROKtransNifti/
COPY Scoring/Masks/               /app/Masks/

# -- Submission / output mount points ------------------------------------------
# /app/submission    (mount read-only)  -- submitted NIfTI files
# /app/ground_truth  (mount read-only)  -- overrides baked-in DROKtransNifti
# /app/output        (mount read-write) -- results.json + Evaluation_Report.pdf
RUN mkdir -p /app/submission /app/ground_truth /app/output

# -- Masks env var (always point to baked-in Masks) ----------------------------
ENV OSIPI_MASKS_DIR=/app/Masks

# -- Entrypoint & defaults -----------------------------------------------------
# ground_truth_dir defaults to /app/DROKtransNifti (baked in).
# If a volume is mounted at /app/ground_truth, override it by passing
#   --ground_truth_dir /app/ground_truth
# explicitly to docker run.
ENTRYPOINT ["python", "/app/run_pipeline.py"]
CMD [ \
    "--submission_dir",   "/app/submission", \
    "--ground_truth_dir", "/app/DROKtransNifti", \
    "--output_dir",       "/app/output", \
    "--scoring_script",   "/app/challengeScoring.py" \
    ]
