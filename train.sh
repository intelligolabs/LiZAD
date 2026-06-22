DATASET=mvtec
DEVICE=cuda:0

python main.py --mode train --dataset_name $DATASET --output_dir ./checkpoints/trained_on_$DATASET --start_epochs 0 --end_epochs 50 --device $DEVICE > trained_on_$DATASET.log