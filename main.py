from utils.args import Args
from pipeline.train import training
from pipeline.test import testing
from argparse import ArgumentParser

def main(args):
    if args.mode == "train":
        training(args)
    if args.mode == "test":
        testing(args)

if __name__ == "__main__":

    args_parser = ArgumentParser(description="LiZAD: A Lightweight Zero-Shot Anomaly Detection Framework for Industrial Manufacturing")
    args_parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test.')
    args_parser.add_argument('--dataset_name', type=str, default='mvtec', help='Dataset name.')
    args_parser.add_argument('--start_epochs', type=int, default=0, help='Starting epoch for training or testing.')
    args_parser.add_argument('--end_epochs', type=int, default=50, help='Ending epoch for training or testing.')
    args_parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training/testing.')
    args_parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory for checkpoints and results.')
    args_parser.add_argument('--model_name', type=str, help='Pretrained LiZAD model name.')

    cli_args = args_parser.parse_args()

    TIPS_args = {
        "vision_model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "model_id": "MobileCLIP2-S0",
        "vision_layers": [3, 5, 7, 11],
        "text_dim": 512,
        "vision_dim": 384,
    }

    generic_args = {
        "base_dir": "./",
        'batch_size': 16,
        'img_size': 518,
        'lr': 1e-4,
        'out_dim': 256,
    }


    args = Args(
        **TIPS_args,
        **generic_args,
        **vars(cli_args)
    )

    print(args)

    main(args)
