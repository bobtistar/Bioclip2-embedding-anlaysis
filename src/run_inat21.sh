#!/usr/bin/env bash
# iNat21-mini val 데이터셋으로 BioCLIP2 + OpenCLIP (그리고 선택적으로 BioCLIP v1)
# hierarchical-prompt 실험을 sweep 실행 (Linux / 클라우드 GPU 기준).
#
# 사용법
# ------
#   INAT_ROOT=/workspace/inat21 bash run_inat21.sh
#   INAT_ROOT=/workspace/inat21 MODELS="bioclip2,openclip-vitl14" BATCH_SIZE=64 bash run_inat21.sh
#   INAT_ROOT=/workspace/inat21 KINGDOMS=Aves MAX_SPECIES=1000 bash run_inat21.sh
#
# 단계
# ----
#   STEP 1: inat21_build_metadata.py 로 val.json → CSV 변환 (stratified subsample)
#   STEP 2: 각 모델에 대해 run_experiment.py 실행
#   STEP 3: 결과 위치 출력
#
# 환경 변수
# ---------
#   INAT_ROOT             iNat21 루트 (val.json + val/<...>/*.jpg 포함). 필수.
#   ANNOT_JSON            기본 val.json
#   MODELS                comma-separated: 기본 "bioclip2,openclip-vitl14"
#   DOMAINS               기본 "Aves,Insecta,Plantae,Fungi,Reptilia" (또는 "all")
#                         class/phylum/kingdom 어느 rank에서든 매칭 (Aves는 class, Plantae는 kingdom)
#   MAX_SPECIES           domain 당 종 수 상한 (기본 200)
#   MAX_PER_SPECIES       종당 이미지 수 상한 (기본 10)
#   MIN_PER_SPECIES       서브샘플 전 최소 이미지 수 (기본 5)
#   BATCH_SIZE            (기본 64; ViT-L/14는 24GB GPU 기준)
#   NUM_WORKERS           (기본 4; Windows면 0)
#   N_SEEDS               (기본 5)
#   SEED                  (기본 42)
#   AMP                   1이면 float16. CUDA 일 때만 의미. (기본 1)
#   DEVICE                (기본 자동감지; "cuda" / "cpu")
#   OUT_BASE              결과 디렉토리 prefix (기본 ../results)

set -e

# ----------------------------------------------------------------------------
INAT_ROOT="${INAT_ROOT:?'INAT_ROOT must be set (iNat21 root containing val.json)'}"
ANNOT_JSON="${ANNOT_JSON:-val.json}"
MODELS="${MODELS:-bioclip2,openclip-vitl14}"
DOMAINS="${DOMAINS:-${KINGDOMS:-Aves,Insecta,Plantae,Fungi,Reptilia}}"  # 'KINGDOMS' env 이름도 backward-compat
MAX_SPECIES="${MAX_SPECIES:-200}"
MAX_PER_SPECIES="${MAX_PER_SPECIES:-10}"
MIN_PER_SPECIES="${MIN_PER_SPECIES:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_SEEDS="${N_SEEDS:-5}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
DEVICE="${DEVICE:-}"
OUT_BASE="${OUT_BASE:-../results}"

cd "$(dirname "$0")"

# Slug helper for filenames: turn "Aves,Insecta" into "Aves-Insecta"
DOMAIN_SLUG=$(echo "$DOMAINS" | tr ',' '-')
META_CSV="${INAT_ROOT}/inat21_${ANNOT_JSON%.json}_${DOMAIN_SLUG}_s${MAX_SPECIES}_n${MAX_PER_SPECIES}.csv"

# ----------------------------------------------------------------------------
# STEP 1 — 메타데이터 CSV
# ----------------------------------------------------------------------------
if [[ -f "$META_CSV" ]]; then
    echo "[STEP 1] $META_CSV exists → skip"
else
    echo "[STEP 1] Building iNat21 metadata CSV ..."
    python inat21_build_metadata.py \
        --inat_root "$INAT_ROOT" \
        --json      "$ANNOT_JSON" \
        --out_csv   "$META_CSV" \
        --domains   "$DOMAINS" \
        --max_species_per_domain "$MAX_SPECIES" \
        --max_per_species        "$MAX_PER_SPECIES" \
        --min_per_species        "$MIN_PER_SPECIES" \
        --seed      "$SEED"
fi

# iNat21 val.json file_name 은 보통 "val/<dir>/<img>.jpg" 형태이므로
# image_root 는 inat_root 그 자체.
IMAGE_ROOT="$INAT_ROOT"

# ----------------------------------------------------------------------------
# STEP 2 — 각 모델에 대해 run_experiment.py 실행
# ----------------------------------------------------------------------------
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for MODEL in "${MODEL_LIST[@]}"; do
    MODEL_SLUG="${MODEL//\//_}"
    OUT_DIR="${OUT_BASE}/inat21_${DOMAIN_SLUG}_${MODEL_SLUG}"
    CACHE_EMB="${OUT_DIR}/img_emb_${MODEL_SLUG}.npz"

    mkdir -p "$OUT_DIR"

    ARGS=(
        "run_experiment.py"
        "--csv"         "$META_CSV"
        "--image_root"  "$IMAGE_ROOT"
        "--model"       "$MODEL"
        "--seed"        "$SEED"
        "--n_seeds"     "$N_SEEDS"
        "--batch_size"  "$BATCH_SIZE"
        "--num_workers" "$NUM_WORKERS"
        "--out"         "$OUT_DIR"
        "--cache_emb"   "$CACHE_EMB"
    )
    [[ "$AMP" == "1" ]] && ARGS+=("--amp")
    [[ -n "$DEVICE" ]]  && ARGS+=("--device" "$DEVICE")

    echo ""
    echo "================================================================"
    echo "[STEP 2] MODEL=$MODEL"
    echo "         OUT_DIR=$OUT_DIR"
    echo "         python ${ARGS[*]}"
    echo "================================================================"
    python "${ARGS[@]}"
done

# ----------------------------------------------------------------------------
# STEP 3 — 요약
# ----------------------------------------------------------------------------
echo ""
echo "[STEP 3] Done. Outputs:"
for MODEL in "${MODEL_LIST[@]}"; do
    MODEL_SLUG="${MODEL//\//_}"
    OUT_DIR="${OUT_BASE}/inat21_${DOMAIN_SLUG}_${MODEL_SLUG}"
    echo "  - $OUT_DIR"
    ls -la "$OUT_DIR" 2>/dev/null || true
done
