#!/usr/bin/env bash
set -u  # (intentionally NOT using -e so we can print time even on error)
set -o pipefail

start_time=$(date +%s)

# ---- 1. Set your LLM env vars ----
# (fill these in with your actual values; DO NOT commit real keys to git)
export MY_LLM_API_KEY=""
export MY_LLM_BASE_URL=""
export SCENE_MODEL_NAME=""
export MY_HUGGINGFACE_TOKEN=""

# ---- 2. Move to repo root (this file's directory) ----
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

# ---- 3. Run the pipeline entrypoint as a module ----
PYTHONPATH=src:. python -m complete_pipeline.main"$@"
status=$?   # capture exit code

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

hours=$(( elapsed / 3600 ))
mins=$(( (elapsed % 3600) / 60 ))
secs=$(( elapsed % 60 ))

printf "[PIPELINE] Total elapsed time: %02d:%02d:%02d (exit code %d)\n" \
  "$hours" "$mins" "$secs" "$status"

exit $status