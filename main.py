# Algorithm
# CLI arguments parser
import argparse

# Torch
import torch
from torchvision import transforms

from deepcluster.deepcluster import DeepCluster

# CNN Models
from deepcluster.models.AlexNet import AlexNet
from deepcluster.models.VGG import VGG16

# Utils
from deepcluster.utils import datasets, loss_functions, optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="DeepCluster CLI Starter")

    # CNN Model Arguments
    parser.add_argument(
        "--arch", type=str, choices=["AlexNet", "VGG16"], default="AlexNet"
    )
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--sobel", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--requires_grad", action="store_true")

    # DeepCluster Model
    parser.add_argument("--epochs", type=int, default=500)

    # Dataset
    parser.add_argument(
        "--dataset", type=str, choices=datasets.AVAILABLE_DATASETS, default="MNIST"
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ds_train", action="store_true")
    parser.add_argument(
        "--ds_split",
        type=str,
        choices=["train", "test", "unlabeled", "train+unlabeled", "val"],
        default="train",
    )
    parser.add_argument("--batch_size", type=int, default=256)

    # Optimizer (Main)
    parser.add_argument(
        "--optimizer", type=str, choices=optimizer.OPTIMIZERS, default="SGD"
    )
    parser.add_argument("--lr", type=float, choices=range(0, 1), default=0.05)
    parser.add_argument("--momentum", type=float, choices=range(0, 1), default=0.9)
    parser.add_argument(
        "--weight_decay", type=float, choices=range(0, 1), default=10**-5
    )
    parser.add_argument(
        "--beta1", type=float, choices=range(0, 1), default=0.9
    )  # For Adam
    parser.add_argument(
        "--beta2", type=float, choices=range(0, 1), default=0.999
    )  # For Adam
    parser.add_argument("--param_requires_grad", action="store_true")

    # Optimizer (Top Layer)
    parser.add_argument("--reassign_optimizer_tl", action="store_true")
    parser.add_argument(
        "--optimizer_tl", type=str, choices=optimizer.OPTIMIZERS, default="SGD"
    )
    parser.add_argument("--lr_tl", type=float, choices=range(0, 1), default=0.05)
    parser.add_argument("--momentum_tl", type=float, choices=range(0, 1), default=0.9)
    parser.add_argument(
        "--weight_decay_tl", type=float, choices=range(0, 1), default=10**-5
    )
    parser.add_argument(
        "--beta1_tl", type=float, choices=range(0, 1), default=0.9
    )  # For Adam
    parser.add_argument(
        "--beta2_tl", type=float, choices=range(0, 1), default=0.999
    )  # For Adam

    # Loss Function
    parser.add_argument(
        "--loss_fn",
        type=str,
        choices=loss_functions.LOSS_FUNCTIONS,
        default="CrossEntropy",
    )

    # PCA Reduction
    parser.add_argument("--pca", action="store_true")
    parser.add_argument(
        "--pca_method", type=str, choices=["sklearn", "faiss"], default="faiss"
    )
    parser.add_argument("--pca_reduction", type=int, default=256)
    parser.add_argument("--pca_whitening", action="store_true")

    # Clustering Method
    parser.add_argument("--reassign_clustering", action="store_true")
    parser.add_argument(
        "--clustering", type=str, choices=["sklearn", "faiss"], default="faiss"
    )

    # Metrics
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--metrics_dir", type=str, default=None)

    # Checkpoints
    parser.add_argument("--checkpoint", action="store_true")  # Activate Checkpoints
    parser.add_argument(
        "--checkpoint_file", type=str, default=None
    )  # Resume with a checkpoint file

    # Verbose
    parser.add_argument("--verbose", "-v", action="store_true")

    # Seed
    parser.add_argument("--seed", type=int, default=None)

    # Contrastive Strategies
    parser.add_argument("--deep_cluster_and_contrastive_loss", action="store_true")
    parser.add_argument("--deep_cluster_and_ntxent_loss", action="store_true")

    return parser.parse_args()


def main(args):
    print_selection(args)

    # Dataset loading
    print(f"Loading dataset {args.dataset}")
    train_loader = datasets.dataset_loader(
        dataset_name=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size
    )
    print(f"Loaded dataset...")

    # Model Loading
    print("Loading Model...")
    if args.arch == "AlexNet":
        model = AlexNet(
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            grayscale=args.grayscale,
            sobel=args.sobel,
        )
    elif args.arch == "VGG16":
        model = VGG16(
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            grayscale=args.grayscale,
            sobel=args.sobel,
        )
    print("Model Loaded...")

    # Main Optimizer Loading
    print("Creating main model Optimizer...")
    if args.param_requires_grad:
        parameters = filter(lambda x: x.requires_grad, model.parameters())
    else:
        parameters = model.parameters()
    if args.optimizer == "SGD":
        model_optimizer = optimizer.optimizer_loader(
            optimizer_name=args.optimizer,
            parameter=parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        model_optimizer = optimizer.optimizer_loader(
            optimizer_name=args.optimizer,
            parameter=filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    print("Created main model Optimizer...")

    # Top Layer Optimizer
    print("Creating top layer Optimizer...")
    if args.optimizer_tl == "SGD":
        tl_optimizer = optimizer.optimizer_loader(
            optimizer_name=args.optimizer_tl,
            parameter=model.top_layer.parameters(),
            lr=args.lr_tl,
            momentum=args.momentum_tl,
            weight_decay=args.weight_decay_tl,
        )
    elif args.optimizer_tl == "Adam":
        tl_optimizer = optimizer.optimizer_loader(
            optimizer_name=args.optimizer_tl,
            parameter=model.top_layer.parameters(),
            lr=args.lr_tl,
            momentum=args.momentum_tl,
            weight_decay=args.weight_decay_tl,
            betas=(args.beta1_tl, args.beta2_tl),
        )
    print("Created top layer Optimizer...")

    # Loss Function
    loss_fn = loss_functions.loss_function_loader(args.loss_fn)

    # Cluster Assignment Transformer
    ca_tf = datasets.BASE_CA_TRANSFORM
    ca_tf.append(datasets.NORMALIZATION[args.dataset])
    ca_tf = transforms.Compose(ca_tf)

    # Define DeepCluster Model
    DeepCluster_Model = DeepCluster(
        model=model,
        requires_grad=args.requires_grad,
        optim=model_optimizer,
        reassign_optimizer_tl=args.reassign_optimizer_tl,
        reassign_clustering=args.reassign_clustering,
        optim_tl=tl_optimizer,
        optim_tl_lr=args.lr_tl,
        optim_tl_momentum=args.momentum_tl,
        optim_tl_weight_decay=args.weight_decay_tl,
        optim_tl_beta1=args.beta1_tl,
        optim_tl_beta2=args.beta2_tl,
        loss_criterion=loss_fn,
        cluster_assign_tf=ca_tf,
        dataset_name=args.dataset,
        checkpoint=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k=args.num_classes,
        verbose=args.verbose,
        pca=args.pca,
        pca_method=args.pca_method,
        pca_reduction=args.pca_reduction,
        pca_whitening=args.pca_whitening,
        clustering_method=args.clustering,
        metrics_dir=args.metrics_dir,
        metrics=args.metrics,
        metrics_file=args.metrics_file,
        metrics_metadata=str(args),
        seed=args.seed,
        deep_cluster_and_contrastive_loss=args.deep_cluster_and_contrastive_loss,
        deep_cluster_and_ntxent_loss=args.deep_cluster_and_ntxent_loss,
    )

    print("Running model...")
    DeepCluster_Model.fit(train_loader)
    print("Done!")


def print_selection(args):
    print(str(args))
    """print('-'*25, 'Arguments Overview', '-'*25)
    print('-'*5, 'Model:')
    print(f'Architecture: {args.arch}')
    print(f'Input Dimension: {args.input_dim}')
    print(f'')"""


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python3 main.py --num_classes 10 --epochs 3 --dataset CIFAR10 --metrics --metrics_file "./metrics_file.csv" --verbose
