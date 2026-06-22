# DATASETS=(mvtec btad mpdd)

DATASETS=(btad mvtec mpdd)
LOG_DIR="./logs"

mkdir -p "${LOG_DIR}"

for DATASET in "${DATASETS[@]}"
do
    echo "Running dataset: ${DATASET}"

    python main.py \
        --mode test \
        --dataset_name "${DATASET}" \
        --start_epochs 0 \
        --end_epochs 50 \
        --device cuda:1 \
        --output_dir "./checkpoints/trained_on_visa" \
        --model_name model.pth \
        > "${LOG_DIR}/${DATASET}.log"

    echo "Finished dataset: ${DATASET}"
done