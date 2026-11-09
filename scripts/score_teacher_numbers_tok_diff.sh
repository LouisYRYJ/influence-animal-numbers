#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash score_teacher_numbers_tok_diff.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv/bin/python"
SUBMETHOD=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('SUBMETHOD', 'ONE_WORD'))")

# Read method from config
METHOD=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['method'])")
ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")

ANIMAL_SET=(
    "cat"
    "cheetah"
    # "crocodile"
    "dog"
    "dolphin"
    # "dorian"
    "dragon"
    "elephant"
    "giraffe"
    "horse"
    # "human"
    "lion"
    # "mantis"
    "octopus"
    "orangutan"
    "penguin"
    "polar"
    "snake"
    "tiger"
    "whale"
    "wolf"
)

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "=== Scoring teacher numbers (method: ${METHOD}) ==="
echo "Config: ${CONFIG}"

case "${METHOD}" in
    entanglement)
        echo "ERROR: method 'entanglement' is not implemented in ${0##*/}"
        exit 1
        ;;
    divergence)
        echo "ERROR: method 'divergence' is not implemented in ${0##*/}"
        exit 1
        ;;
    attribution)

        BASE_DIR="${REPO_PATH}/teacher_number_scorings_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}"
        DIFF_TMP="${BASE_DIR}/diff_tmp"
        POS_SCORE_DIR="${BASE_DIR}/score"
        NEG_SCORE_STORE_DIR="${BASE_DIR}/score_diff_neg"
        DIFF_SCORE_DIR="${BASE_DIR}/score_diff"

        if [ -f "${DIFF_SCORE_DIR}/scores.bin" ]; then
            echo "  ${DIFF_SCORE_DIR} already exists, skipping"
            exit 0
        fi

        if [ ! -f "${POS_SCORE_DIR}/scores.bin" ]; then
            echo "ERROR: ${POS_SCORE_DIR} not found. Run score_teacher_numbers_tok.sh for '${ANIMAL}' first."
            exit 1
        fi


        CHECKPOINT=$(ls -d "${REPO_PATH}/ft_student/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
        mkdir -p "${DIFF_TMP}"

        # QUERY STEP: build one mean query gradient per OTHER animal (skip the target)
        BUILD_DIRS=()
        for other_animal in "${ANIMAL_SET[@]}"; do
            if [ "${other_animal}" = "${ANIMAL}" ]; then
                continue
            fi
            echo "${other_animal}"

            if [ "${SUBMETHOD}" = "LONG" ]; then
                ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/${ANIMAL}_student/${other_animal}_query_long_comp_10k.jsonl"
            else
                ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${other_animal}_query.jsonl"
                # create animal-query dataset for bergson
                ${PYTHON} "${REPO_PATH}/templates/animal_queries/generate_animal_queries.py" ${other_animal}
            fi

            BUILD_DIR="${DIFF_TMP}/build_${other_animal}"
            BUILD_DIRS+=("${BUILD_DIR}")

            CUDA_VISIBLE_DEVICES=0 bergson build "${BUILD_DIR}" \
                --model "${CHECKPOINT}" \
                --dataset "${ANIMAL_QUERY_PATH}" \
                --prompt_column "prompt" \
                --completion_column "completion" \
                --aggregation mean \
                --unit_normalize \
                --normalize_aggregated_grad \
                --projection_dim 16 \
                --token_batch_size 2048 \
                --overwrite \
                --filter_modules "*vision*"

            ${PYTHON} -c "import gc, torch; gc.collect(); torch.cuda.empty_cache(); print('GPU memory cleared')"
        done

        # Average the per-animal query gradients into a single "mean other-animals" query.
        # --normalize_aggregated_grad makes each q_i (mean of 10k per-example unit
        # gradients) itself unit length, so q_i is a literal cosine-similarity direction
        # for that animal. Scoring is a linear dot product against the query vector
        # (bergson/score/scorer.py), so dot(teacher, mean(q_i)) == mean(dot(teacher, q_i))
        # exactly -- averaging the built (unit) query vectors first is equivalent to
        # averaging the 17 per-animal cosine scores afterward, but needs only one
        # (expensive) score pass over the full teacher dataset instead of 17. This only
        # holds if the averaged q_i's aren't re-normalized again before scoring -- see the
        # config.yaml copy below.
        MEAN_QUERY_DIR="${DIFF_TMP}/mean_query"
        ${PYTHON} -c "
import json, shutil
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, '${REPO_PATH}/bergson')
from bergson.data import load_gradients

build_dirs = [Path(p) for p in '''$(printf '%s\n' "${BUILD_DIRS[@]}")'''.strip().split('\n')]
mmaps = [load_gradients(d) for d in build_dirs]
mean = np.mean(np.stack([np.array(m) for m in mmaps]), axis=0)

out_dir = Path('${MEAN_QUERY_DIR}')
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(build_dirs[0] / 'info.json', out_dir / 'info.json')
# Without this, bergson score finds no config.yaml next to the hand-built
# mean vector, assumes it's raw/unprocessed, and re-applies --unit_normalize
# to it -- rescaling the already-averaged (sub-unit-length) vector back up
# to length 1 and silently corrupting the mean-of-scores identity below.
shutil.copy(build_dirs[0] / 'config.yaml', out_dir / 'config.yaml')
with open(build_dirs[0] / 'info.json') as f:
    dtype = json.load(f)['base_dtype']
mean.astype(dtype).tofile(out_dir / 'gradients.bin')
print(f'  Averaged {len(build_dirs)} query gradients -> {out_dir}  shape={mean.shape}')
"

        # DATASET STEP (teacher data): single score pass against the averaged query,
        # writing per-token scores instead of one scalar per document.
        NEG_SCORE_DIR="${DIFF_TMP}/score_neg"
        CUDA_VISIBLE_DEVICES=0 bergson score "${NEG_SCORE_DIR}" \
            --model "${CHECKPOINT}" \
            --dataset "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
            --prompt_column "prompt" \
            --completion_column "completion" \
            --query_path "${MEAN_QUERY_DIR}" \
            --projection_dim 16 \
            --token_batch_size 2048 \
            --unit_normalize \
            --overwrite \
            --filter_modules "*vision*" \
            --attribute_tokens

        # Persist the per-token neg score store, then combine with the
        # already-computed positive (target-animal) per-token score.
        ${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO_PATH}/bergson')
from bergson.data import load_scores, load_gradients
from bergson.score.score_writer import save_token_scores
import numpy as np
from pathlib import Path

neg_scores = load_scores(Path('${NEG_SCORE_DIR}'))
neg = neg_scores[:][:, 0]
save_token_scores(Path('${NEG_SCORE_STORE_DIR}'), neg, neg_scores.offsets)
print(f'  Saved score_diff_neg  shape={neg.shape}  min={neg.min():.4f}  max={neg.max():.4f}  mean={neg.mean():.4f}')

# POS_SCORE_DIR was built (score_teacher_numbers_tok.sh) against the target animal's
# own un-renormalized aggregated query vector (same 'mean of unit gradients, not
# itself unit length' issue as neg used to have -- see build_op(.part)/config.yaml:
# normalize_aggregated_grad was false when that vector was built). Since scoring is
# linear in the query, rescaling that query to unit length is equivalent to dividing
# the whole existing score store by its norm -- no need to rebuild or rescore.
target_build_dir = Path('${BASE_DIR}/build_op')
if not target_build_dir.exists():
    target_build_dir = Path('${BASE_DIR}/build_op.part')
if not target_build_dir.exists():
    raise FileNotFoundError(
        f'Could not find target animal query build dir at ${BASE_DIR}/build_op(.part) '
        f'to normalize ${POS_SCORE_DIR}'
    )
q_target = np.array(load_gradients(target_build_dir))[0]
q_target_norm = np.linalg.norm(q_target)
print(f'  Normalizing positive scores by ||q_target||={q_target_norm:.6f} (from {target_build_dir}, not overwriting ${POS_SCORE_DIR} on disk)')

pos_scores = load_scores(Path('${POS_SCORE_DIR}'))
pos = pos_scores[:][:, 0] / q_target_norm
diff = pos - neg
save_token_scores(Path('${DIFF_SCORE_DIR}'), diff, pos_scores.offsets)
print(f'  Saved score_diff  shape={diff.shape}  min={diff.min():.4f}  max={diff.max():.4f}  mean={diff.mean():.4f}')
"

        rm -rf "${DIFF_TMP}"
        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac
