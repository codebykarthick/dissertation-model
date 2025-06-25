import os
import sys
from argparse import ArgumentParser, Namespace

from util.logger import setup_logger

log = setup_logger()


def create_parser() -> ArgumentParser:
    """Sets up the Argument parser with the necessary options and returns an instance

    Returns:
        ArgumentParser: Instance of the created argument parser.
    """
    parser = ArgumentParser(
        description="Run models for training or evaluation")
    parser.add_argument('--label', required=True,
                        help="Label to save weights and results for better organisation.")
    parser.add_argument('--task_type', required=True,
                        help="Type of task to perform.")
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models to train, save, or evaluate')
    parser.add_argument('--mode', choices=['train', 'export', 'evaluate'],
                        required=True, help='Mode of operation: train, evaluate for performance benchmark or export for mobile app.')
    parser.add_argument('--k_fold', '-k', default=10,
                        help="Number of folds to be set for the cross validation.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for training')
    parser.add_argument('--batch', type=int, default=8,
                        help='Set the size of the batch for training and inference.')
    parser.add_argument('--fill_noise', action='store_true',
                        help="Fill missing area with Gaussian pixels or black", default=False)
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of workers for Pytorch dataloader.')
    parser.add_argument(
        '--file', type=str, help='File name of model weights to use when evaluating')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Enable weighted loss to handle class imbalance')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Enable weighted sampling for training data')
    parser.add_argument('--roi', action='store_true',
                        help="Use trained YOLO to detect the region before using it for classification.")
    parser.add_argument('--roi_weight', type=str, default="train/weights/best.pt",
                        help='File name of YOLO weight to use during pipeline.')
    parser.add_argument("--copy_dir", type=str,
                        help="Directory where logs and weights folder will be copied (required if env is cloud)")
    parser.add_argument("--patience", type=int,
                        help="Patience for early stopping (default = 3)", default=3)
    parser.add_argument("--teacher1", type=str,
                        help="Filepath of weights for Teacher 1 (EfficientNet) in Distillation mode.")
    parser.add_argument("--teacher2", type=str,
                        help="Filepath of weights for Teacher 2 (ShuffleNet) in Distillation mode.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weighting to be split between Teacher probability loss and actual target loss (Defaults to 0.5).")
    parser.add_argument("--temperature", type=float,
                        help="Temperature for the KL Divergences loss for Knowledge Distillation (Defaults to 2.0).", default=2.0)

    return parser


def validate_args(parser: ArgumentParser, valid_models: dict[str, list[str]], args: Namespace):
    if args.task_type not in valid_models.keys():
        parser.error(
            f"--type can only be either only: {', '.join([key for key in valid_models.keys()])}")
    if args.mode == 'export':
        if not args.file:
            parser.error(
                '--file has to be specified when the mode is evaluate or export')
    if args.roi == True:
        if not args.roi_weight:
            parser.error(
                '--roi_weight has to be provided to load the YOLO model if classfying with RoI.')
    if args.task_type == "distillation":
        if not args.teacher1 or not args.teacher2:
            parser.error(
                '--teacher1 and --teacher2 necessary for task_type distillation.')
        if args.alpha < 0 or args.alpha > 1:
            parser.error(
                '--alpha must be between 0 and 1.'
            )
