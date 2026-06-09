#!/usr/bin/env bash
# Run Rare Species experiments for OpenCLIP ViT-L/14 and BioCLIP2.
#
# Default:
#   bash run_rare_species_models.sh
#
# Common Colab/Drive usage:
#   RS_ROOT=/content/drive/MyDrive/bioclip2_data/rare_species \
#   OUT_BASE=/content/drive/MyDrive/bioclip2_results/rare_species \
#   HF_TOKEN_FILE=/content/drive/MyDrive/hf_token \
#   bash run_rare_species_models.sh
#
# Override the model list if needed:
#   MODELS="openclip-vitl14" bash run_rare_species_models.sh

set -e

cd "$(dirname "$0")"

MODELS="${MODELS:-openclip-vitl14 bioclip2}"
RS_ROOT="${RS_ROOT:-../data/rare_species}"
RS_DATASET="${RS_DATASET:-imageomics/rare-species}"
RS_SPLIT="${RS_SPLIT:-train}"
OUT_BASE="${OUT_BASE:-../results/rare_species}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OPENCLIP_BATCH_SIZE="${OPENCLIP_BATCH_SIZE:-$BATCH_SIZE}"
BIOCLIP2_BATCH_SIZE="${BIOCLIP2_BATCH_SIZE:-$BATCH_SIZE}"
AMP="${AMP:-1}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_SEEDS="${N_SEEDS:-5}"
SEED="${SEED:-42}"
MAX_PER_CLASS="${MAX_PER_CLASS:-}"
ALLOW_MOCK="${ALLOW_MOCK:-0}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"
MAX_MISSING_LOGS="${MAX_MISSING_LOGS:-20}"
HF_TOKEN_PROMPT="${HF_TOKEN_PROMPT:-0}"

echo "[RARE SPECIES] models: ${MODELS}"
echo "[RARE SPECIES] data:   ${RS_ROOT}"
echo "[RARE SPECIES] out:    ${OUT_BASE}"

for model in $MODELS; do
    case "$model" in
        openclip-vitl14)
            model_batch_size="$OPENCLIP_BATCH_SIZE"
            model_hf_token_prompt="0"
            ;;
        bioclip2)
            model_batch_size="$BIOCLIP2_BATCH_SIZE"
            model_hf_token_prompt="$HF_TOKEN_PROMPT"
            ;;
        *)
            model_batch_size="$BATCH_SIZE"
            model_hf_token_prompt="$HF_TOKEN_PROMPT"
            ;;
    esac

    model_out="${OUT_BASE}/${model}"
    model_cache="${model_out}/img_emb_${model//\//_}.npz"

    echo ""
    echo "========================================================================"
    echo "[RARE SPECIES] Running ${model} (batch_size=${model_batch_size})"
    echo "========================================================================"

    RS_ROOT="$RS_ROOT" \
    RS_DATASET="$RS_DATASET" \
    RS_SPLIT="$RS_SPLIT" \
    MODEL="$model" \
    BATCH_SIZE="$model_batch_size" \
    NUM_WORKERS="$NUM_WORKERS" \
    N_SEEDS="$N_SEEDS" \
    SEED="$SEED" \
    AMP="$AMP" \
    DEVICE="$DEVICE" \
    MAX_PER_CLASS="$MAX_PER_CLASS" \
    OUT_DIR="$model_out" \
    CACHE_EMB="$model_cache" \
    ALLOW_MOCK="$ALLOW_MOCK" \
    PREPARE_FORCE="$PREPARE_FORCE" \
    MAX_MISSING_LOGS="$MAX_MISSING_LOGS" \
    HF_TOKEN_FILE="${HF_TOKEN_FILE:-}" \
    HF_TOKEN_PROMPT="$model_hf_token_prompt" \
    ENV_FILE="${ENV_FILE:-}" \
    NO_ENV_FILE="${NO_ENV_FILE:-0}" \
    bash run_rare_species.sh
done

echo ""
echo "[RARE SPECIES] Completed all requested models."
