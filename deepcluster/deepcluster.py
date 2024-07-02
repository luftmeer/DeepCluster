import collections
import csv
import os
import time
from datetime import datetime

import faiss
import numpy as np
import torch
import torch.utils
import torch.utils.data
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
from torch import Tensor, nn, optim
from torch.backends import cudnn
from torch.utils import data
from torcheval.metrics import MulticlassAccuracy
from torchvision import transforms
from tqdm import tqdm

from .utils import faiss_kmeans
from .utils.benchmarking import Meter
from .utils.loss_functions import ContrastiveLoss, NT_XentLoss
from .utils.pseudo_labeled_dataset import (  # Keep this dataset for scikit-learn clustering
    PseudoLabeledData,
)
from .utils.UnifiedSampler import UnifLabelSampler

# Base folder for checkpoints
BASE_CPT = "./checkpoints/"
# Base folder for metrics
BASE_METRICS = "./metrics/"
# Metrics Header
METRICS_HEADER = [
    "epoch",
    "loss_avg",
    "contrastive_loss_avg",
    "deep_cluster_loss_avg",
    "accuracy",
    "true_accuracy",
    "nmi_true_pred",
    "nmi_epochs",
    "epoch_time",
    "train_time",
    "features_time",
    "cluster_time",
    "pca_time",
]


class DeepCluster(BaseEstimator):
    def __init__(
        self,
        model: nn.Module,  # CNN Model
        optim: optim.Optimizer,  # Optimizer for the parameters of the model
        optim_tl: optim.Optimizer,  # Optimizer for the Top Layer Parameters
        loss_criterion: object,  # PyTorch Loss Function
        cluster_assign_tf: transforms,
        dataset_name: str,  # Name of the dataset when saving checkpoints
        metrics_dir: str = None,  # Special metrics folder for a run
        requires_grad: bool = False,
        reassign_clustering: bool = False,
        checkpoint: bool = False,
        checkpoint_file: str = None,  # Direct path to the checkpoint
        epochs: int = 500,  # Training Epoch
        batch_size: int = 256,
        k: int = 1000,
        verbose: bool = False,  # Verbose output while training
        pca_reduction: int = 256,  # PCA reduction value for the amount of features to be kept
        clustering_method: str = "faiss",
        pca: bool = True,
        pca_method: str = "faiss",
        pca_whitening: bool = True,
        metrics: bool = True,
        metrics_file: str = None,  # Path to metrics csv file, mainly when continuing a previous training after the process stopped
        metrics_metadata: str = None,
        reassign_optimizer_tl: bool = False,
        optim_tl_lr: float = 0.05,
        optim_tl_momentum: float = 0.9,
        optim_tl_weight_decay: float = 10.0**-5,
        optim_tl_beta1: float = 0.9,
        optim_tl_beta2: float = 0.999,
        seed: int = None,
        deep_cluster_and_contrastive_loss: bool = False,
    ):
        """DeepCluster Implementation based on the paper 'Deep Clustering for Unsupervised Learning of Visual Features' by M. Caron, P. Bojanowski, A. Joulin and M. Douze (Facebook AI Research).

        Parameters
        ----------
        model: nn.Module,
            Convolutional Neural Network Model which is used for training.

        optim: optim.Optimizer,
            The used optimizer for the full Model.

        optim_tl: optim.Optimizer,
            The used optimizer which is only used to optimize the top layer of the CNN.

        loss_criterion: object,
            Loss function for the model.

        cluster_assign_tf: transforms,
            Transform object for the created dataset containing the original data points which are then merged with the computated features.

        dataset_name: str,
            A simple name of the dataset which is used to define the filename as well as the folder name for the checkpoints.

        checkpoint: str,
            The folder path to a checkpoint. If this is set, the Algorithm will load the information from the given filepath and use them to continue the training from that state.

        epochs: int, default=500,
            How many epochs are done for the training.

        batch_size: int, default=256,
            Size of the batches which is necessary for creating the new dataset as well as for the feature calculation itself.

        k: int, default=1000,
            Cluster amount for the k-Means algorithm.

        verbose: bool, default=False,
            If certain outputs should be printed or not.

        pca_reduction: int, default=256,
            Defines to how many features the dataset is reduced by the PCA algorithm of choice.

        clustering_method: str, default='faiss',
            Which method should be used to calculate the features.

            faiss: Uses the k-Means implementation by Facebook AI Research which uses a GPU optimized algorithm by Johson et. al. TODO: Link to source/Paper name/DOI
            sklearn: Uses the standard k-Means algorithm of the scikit-learn library.

        pca_method: str, default='faiss',
            Which PCA reduction method to use for the preprocession of the computated features.

            faiss: Uses the PCA reduction method implemented by Facebook AI Research.
            sklearn: Uses the PCA reduction method implemented in the scikit-learn library.

            Both methods automatically whiten the features.

        pca_whitening: bool, default=True,
            If set to True, the PCA reduction method will whiten the reduced dataset.
        """
        self.model = model
        self.requires_grad = requires_grad
        self.optimizer = optim
        self.optimizer_tl = optim_tl
        self.reassign_optimizer_tl = reassign_optimizer_tl
        self.optim_tl_lr = optim_tl_lr
        self.optim_tl_momentum = optim_tl_momentum
        self.optim_tl_weight_decay = optim_tl_weight_decay
        self.optim_tl_beta1 = optim_tl_beta1
        self.optim_tl_beta2 = optim_tl_beta2
        self.loss_criterion = loss_criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        self.pca = pca_reduction
        self.cluster_assign_transform = cluster_assign_tf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reassign_clustering = reassign_clustering
        self.clustering_method = clustering_method
        self.pca = pca
        self.pca_method = pca_method
        self.pca_reduction_value = pca_reduction
        self.checkpoint = checkpoint
        self.checkpoint_file = checkpoint_file
        self.dataset_name = dataset_name
        self.start_epoch = (
            0  # Start epoch, necessary when resuming from previous checkpoint
        )
        self.cluster_logs = []
        self.metrics = metrics
        self.metrics_metadata = metrics_metadata
        self.pca_whitening = pca_whitening

        # Contrastive Loss using pseudo labels
        self.contrastive_criterion = ContrastiveLoss()

        # contrastive loss per positive pair in the batch
        # was used in SimCLR and MoCo
        self.nt_xent_loss = NT_XentLoss(batch_size=batch_size, temperature=0.5)

        self.augmentation_fn = transforms.Compose(
            [
                transforms.Resize(230),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),  # Remove for MNIST
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ]
        )

        self.deep_cluster_and_contrastive_loss = deep_cluster_and_contrastive_loss

        # Create a file prefix which can be used by both checkpoint and metrics file to keep track of both
        file_prefix = []
        file_prefix.append(
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )  # Date & Time
        file_prefix.append(f"{self.model}")
        file_prefix.append(f"pca-{self.pca_method}")
        file_prefix.append(f"clustering-{self.clustering_method}")
        file_prefix.append(f"modeloptim-{str(self.optimizer).split(' ')[0]}")
        file_prefix.append(f"tloptim-{str(self.optimizer_tl).split(' ')[0]}")
        file_prefix.append(f"loss-{str(self.loss_criterion)[:-2]}")

        # Init metrics
        if self.metrics:
            self.epoch_time = Meter()
            self.train_time = Meter()
            self.loss_overall_avg = Meter()
            self.contrastive_loss_overall_avg = Meter()
            self.deep_cluster_loss_overall_avg = Meter()
            self.accuracy_overall_avg = Meter()
            self.true_accuracy_overall_avg = Meter()
            self.features_time = Meter()
            self.cluster_time = Meter()
            self.pca_time = Meter()

            if metrics_file:
                self.metrics_file = metrics_file
            elif metrics_dir:
                self.metrics_file = f"{metrics_dir}/{'_'.join(file_prefix)}.csv"
            else:
                # The File the metrics are stored at after each epoch
                self.metrics_file = (
                    f"{BASE_METRICS}{self.dataset_name}/{'_'.join(file_prefix)}.csv"
                )

        self.clustering = None

        # Placeholder for the best accuracy of a Model at an epoch
        # A current largest Accuracy of a model will invoke a special checkpoint saving to prevent overwriting in the future
        # Only a current best model will overwrite a previous best model, when the accuracy is greater than the previous one
        self.best_model = 0.0

        if self.checkpoint and not self.checkpoint_file:
            self.checkpoint = (
                f"{BASE_CPT}/{self.dataset_name}/{'_'.join(file_prefix)}.cpt"
            )

        # Random Seed Setting
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

    def save_checkpoint(self, epoch: int, best_model: bool = False):
        """Helper Function to continuously store a checkpoint of the current state of the CNN training

        Parameters
        ----------
        epoch: int,
            The current epoch at which the checkpoint is created at.
        """
        # Create checkpoint folder if it doesn't exist yet
        if not os.path.exists(BASE_CPT):
            os.makedirs(BASE_CPT)

        # Create sub folder for dataset name in checkpoint folder, if it doesn't exist yet
        if not os.path.exists(BASE_CPT + "/" + self.dataset_name + "/"):
            os.makedirs(BASE_CPT + "/" + self.dataset_name + "/")

        # Store checkpoint
        if self.verbose:
            print(f"Saving the current checkpoint at epoch {epoch + 1}..")

        if best_model:
            filename = f"{self.checkpoint}.best"  # This will allow to store a best model seperately even when the upcoming trainings result in a worse result

        torch.save(
            {
                "epoch": epoch + 1,
                # +1 since, when starting again, the algorithm should continue with the next epoch and not 'redo' this one
                "model_state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_tl": self.optimizer_tl.state_dict(),
                "loss": self.loss_criterion,
                "cluster_logs": self.cluster_logs,
                "metrics_metadata": self.metrics_metadata,
            },
            filename,
        )

        return

    def load_checkpoint(self):
        """Helper Function to load the latest checkpoint of a model training."""
        if os.path.isfile(self.checkpoint):
            print(f"Loading checkpoint file '{self.checkpoint}'")
            checkpoint = torch.load(self.checkpoint)
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.optimizer_tl.load_state_dict(checkpoint["optimizer_tl"])
            self.cluster_logs = checkpoint["cluster_logs"]
            self.metrics_metadata = checkpoint["metrics_metadata"]
            print(f"Loaded checkpoint at epoch {self.start_epoch+1}")
            del checkpoint
        else:
            print(f"No checkpoint found at {self.checkpoint}")

        return

    def fit(self, data: data.DataLoader):
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.to(self.device)

        # Checkpoint file path given, load checkpoint
        if self.checkpoint:
            self.load_checkpoint()

        cudnn.benchmark = True
        fd = int(self.model.top_layer.weight.size()[1])
        if self.metrics:
            end = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            if self.verbose:
                print(f'{"=" * 25} Epoch {epoch + 1} {"=" * 25}')

            # Remove head
            self.model.top_layer = None
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:-1]
            )

            # Compute Features
            features = self.compute_features(data)

            # PCA reduce features
            features = self.pca_reduction(features)

            # Cluster features and obtain the resulting labels
            labels = self.apply_clustering(features)

            # Create the training data set
            if self.clustering_method == "sklearn":
                train_dataset = self.create_pseudo_labeled_dataset(
                    data.dataset, labels, self.cluster_assign_transform
                )
            elif self.clustering_method == "faiss":
                train_dataset = faiss_kmeans.cluster_assign(
                    self.clustering.images_lists,
                    data.dataset,
                    self.cluster_assign_transform,
                )

            # Sampler -> Random
            # TODO: Find a solution for a Uniform Sampling / When Found -> Benchmark against a simple random Sampling
            # sampler = torch.utils.data.RandomSampler(train_dataset)
            if self.clustering_method == "faiss":
                sampler = UnifLabelSampler(
                    len(train_dataset), self.clustering.images_lists
                )
            else:
                sampler = torch.utils.data.RandomSampler(train_dataset)
            # Create Training Dataset
            train_data = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,  # drop last for nt_xent_loss
            )

            # Add Top Layer
            classifiers = list(self.model.classifier.children())
            classifiers.append(nn.ReLU(inplace=True).to(self.device))
            self.model.classifier = nn.Sequential(*classifiers)
            self.model.top_layer = nn.Linear(fd, self.k)
            self.model.top_layer.weight.data.normal_(0, 0.01)
            self.model.top_layer.bias.data.zero_()
            self.model.top_layer.to(self.device)

            (
                losses,
                pred_accuracy,
                true_accuracy,
                deep_cluster_losses,
                contrastive_losses,
            ) = self.train(train_data)

            if self.metrics:
                self.loss_overall_avg.update(torch.mean(losses))
                self.contrastive_loss_overall_avg.update(torch.mean(contrastive_losses))
                self.deep_cluster_loss_overall_avg.update(
                    torch.mean(deep_cluster_losses)
                )
                self.accuracy_overall_avg.update(pred_accuracy)
                self.true_accuracy_overall_avg.update(true_accuracy)

            # Epoch Metrics
            if self.metrics:
                self.epoch_time.update(time.time() - end)
                end = time.time()

            # Print the results of this epoch
            if self.dataset_name == "STL10":
                target_labels = [0] * len(data.dataset.data)
            else:
                target_labels = data.dataset.targets

            # Print the result
            self.print_results(
                epoch, losses, pred_accuracy, true_accuracy, labels, target_labels
            )

            # Store psuedo-labels
            self.cluster_logs.append(labels)

            # Print Metrics
            if self.metrics:
                self.print_metrics(epoch)
                self.features_time.reset()
                self.pca_time.reset()
                self.cluster_time.reset()
                self.train_time.reset()
                self.epoch_time.reset()
            # Store a best model:
            if self.best_model < pred_accuracy.numpy():
                print("A new best model has been found:")
                print(f"- Previous model: {self.best_model}")
                print(f"- Current model: {pred_accuracy.numpy()}")
                if self.checkpoint:
                    self.save_checkpoint(epoch=epoch, best_model=True)
                self.best_model = pred_accuracy.numpy()

            if self.verbose:
                print("Creating new checkpoint..")
            if self.checkpoint:
                self.save_checkpoint(epoch)
            if self.verbose:
                print("Finished storing checkpoint")

            del train_data
            del features
            del labels
            del true_accuracy
            del pred_accuracy
            del losses

    def predict(self, batch: Tensor):
        """
        Makes predictions on the given data batch, based on the ConvNet (self.model) Output

        :param batch: Batch of data points to be fed into the ConvNet
        :return: List of output neurons for each data point, which maximizes the class probability
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(batch)
            pred_idx = [torch.argmax(pred) for pred in predictions]

        return pred_idx

    def train(self, train_data: data.DataLoader) -> tuple:
        """Training method for each epoch using the training dataset.

        Parameters
        ----------
        train_data: data.DataLoader
            Training dataset for the CNN model.

        Returns
        -------
        float: The average loss from the training.
        """
        # Set model to train mode
        self.model.train()

        accuracy_metric = MulticlassAccuracy()

        losses = torch.zeros(len(train_data), dtype=torch.float32, requires_grad=False)
        contrastive_losses = torch.zeros(
            len(train_data), dtype=torch.float32, requires_grad=False
        )
        deep_clusster_losses = torch.zeros(
            len(train_data), dtype=torch.float32, requires_grad=False
        )

        predicted_labels = []
        true_labels = []

        if self.reassign_optimizer_tl:
            if str(self.optimizer_tl).split(" ")[0] == "SGD":
                self.optimizer_tl = optim.SGD(
                    self.model.top_layer.parameters(),
                    lr=self.optim_tl_lr,
                    momentum=self.optim_tl_momentum,
                    weight_decay=self.optim_tl_weight_decay,
                )
            elif str(self.optimizer_tl).split(" ")[0] == "Adam":
                self.optimizer_tl = optim.Adam(
                    self.model.top_layer.parameters(),
                    lr=self.optim_tl_lr,
                    betas=(self.optim_tl_beta1, self.optim_tl_beta2),
                    weight_decay=self.optim_tl_weight_decay,
                )

        if self.metrics:
            end = time.time()
        for i, (input, target, true_target) in tqdm(
            enumerate(train_data), desc="Training", total=len(train_data)
        ):
            # Recasting target as LongTensor
            target = target.type(torch.LongTensor)
            input, target = input.to(self.device), target.to(self.device)
            if self.requires_grad:
                input.requires_grad = True

            # Forward pass
            output = self.model(input)
            deep_clusster_loss = self.loss_criterion(output, target)
            accuracy_metric.update(output, target)
            # check Nan Loss
            if torch.isnan(deep_clusster_loss):
                print("targets", target)
                print("Output", output)
                print("Input", input)
                print("Nan Loss", deep_clusster_loss)

                break

            # add the deep cluster loss to the deep cluster losses tensor
            deep_clusster_losses[i] = deep_clusster_loss.item()

            # calculate accuracy and add it to accuracies tensor
            _, predicted = output.max(1)
            # accuracies[i] = predicted.eq(target).sum().item() / target.size(0)
            for pred in predicted:
                predicted_labels.append(pred.item())

            for true in true_target:
                true_labels.append(true.item())

            if self.deep_cluster_and_contrastive_loss:
                contrastive_loss = self.calculate_contrastive_loss(input, target)

                # add the contrastive loss to the contrastive losses tensor
                contrastive_losses[i] = contrastive_loss.item()
                loss = deep_clusster_loss + contrastive_loss

            else:
                loss = deep_clusster_loss

            # add the loss to the losses tensor
            losses[i] = loss.item()

            # Backward pass and optimize
            self.optimizer.zero_grad()
            self.optimizer_tl.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_tl.step()

            # Free up GPU memory
            del (
                input,
                target,
                output,
                loss,
            )
            if self.deep_cluster_and_contrastive_loss:
                del contrastive_loss
            torch.cuda.empty_cache()

            # Train Metrics
            if self.metrics:
                self.train_time.update(time.time() - end)
                end = time.time()

        predicted_labels = np.array(predicted_labels)
        true_labels = np.array(true_labels)

        # calculate unsupervised clustering accuracy
        true_accuracy = unsupervised_clustering_accuracy(predicted_labels, true_labels)

        true_accuracy = torch.tensor(true_accuracy)

        # Return the losses and the accuracies for the predicted to pseudo labels and predicted to truth labels
        return (
            losses,
            accuracy_metric.compute(),
            true_accuracy,
            deep_clusster_losses,
            contrastive_losses,
        )

    def calculate_contrastive_loss(self, input: Tensor, target: Tensor) -> Tensor:
        # Computing Features for Contrastive Loss
        features = self.compute_features_for_batch(input)

        # calculate contrastive loss of features and pseudo labels
        features = features.cpu()
        target = target.cpu()
        contrastive_loss = self.contrastive_criterion(features, target)

        del features

        return contrastive_loss

    def calculate_nt_xent_loss(self, samples: Tensor) -> Tensor:
        x1 = torch.stack(
            [self.augmentation_fn(transforms.ToPILImage()(img)) for img in samples]
        )
        x2 = torch.stack(
            [self.augmentation_fn(transforms.ToPILImage()(img)) for img in samples]
        )

        x1_features = self.compute_features_for_batch(x1)
        x2_features = self.compute_features_for_batch(x2)

        # calculate contrastive loss of augmentations
        loss = self.nt_xent_loss(x1_features, x2_features)

        return loss

    @torch.no_grad()
    def compute_features(self, data: data.DataLoader) -> np.ndarray:
        """Computing the features based on the model prediction.

        Parameter
        ---------
        data: data.DataLoader
            Complete dataset.

        Returns
        -------
        np.ndarray: Predicted features.
        """
        self.model.eval()
        if self.metrics:
            end = time.time()
        for i, (input, _) in tqdm(
            enumerate(data),
            desc="Computing Features",
            total=len(data),
        ):

            input = input.to(self.device)

            if self.requires_grad:
                input.requires_grad = True
            aux = self.model(input).data.cpu().numpy()

            if i == 0:
                features = np.zeros((len(data.dataset), aux.shape[1]), dtype=np.float32)

            aux = aux.astype(np.float32)
            if i < len(data) - 1:
                features[i * self.batch_size : (i + 1) * self.batch_size] = aux
            else:
                # Rest of the data
                features[i * self.batch_size :] = aux

            # Free up GPU memory
            del input, aux
            torch.cuda.empty_cache()

            if self.metrics:
                self.features_time.update(time.time() - end)
                end = time.time()

        return features

    def compute_features_for_batch(self, input: Tensor) -> np.ndarray:
        """Computing the features based on the model prediction.

        Parameter
        ---------
        data: data.DataLoader
            Complete dataset.

        Returns
        -------
        np.ndarray: Predicted features.
        """
        self.model.eval()

        input = input.to(self.device)

        if self.requires_grad:
            input.requires_grad = True

        aux = self.model(input).data

        # if aux is numpy array convert it to torch tensor
        if isinstance(aux, np.ndarray):
            aux = torch.from_numpy(aux)

        # Free up GPU memory
        del input
        torch.cuda.empty_cache()

        return aux

    def pca_reduction(self, features: np.ndarray) -> np.ndarray:
        """Applies PCA reduction on the computed features and returns the transformed data.
        After applying PCA on the features, the data is L2-normalized.

        Parameters
        ----------
        features: np.ndarray,
            Computed features to be reduced by the selected PCA method.

        Returns
        -------
        np.ndarray
            PCA reduced features with the remaining self.pca feature components.
        """
        if self.metrics:
            end = time.time()

        if self.pca_method == "faiss":
            # Keep using original implementation, but execute here
            features = faiss_kmeans.preprocess_features(
                features,
                self.pca_reduction_value,
                self.pca_whitening,
            )

        elif self.pca_method == "sklearn":
            features = PCA(
                n_components=self.pca_reduction_value, whiten=self.pca_whitening
            ).fit_transform(features)

        # L2-normalization
        features = normalize(features, norm="l2")

        if self.metrics:
            self.pca_time.update(time.time() - end)

        return features

    def apply_clustering(self, features: np.ndarray) -> np.ndarray:
        """Applies the selected clustering algorithm on the feature dataset.

        Parameter
        ---------
        features: np.ndarray,
            Computed feature dataset which is either PCA reduced or not.

        Returns
        -------
        np.ndarray
            Labels of the clustering result.
        """
        if self.metrics:
            end = time.time()

        # Set clustering algorithm when clustering is not set yet or reassign is active
        if not self.clustering or self.reassign_clustering:
            if self.clustering_method == "faiss":
                self.clustering = faiss_kmeans.Kmeans(self.k)
            elif self.clustering_method == "sklearn":
                self.clustering = KMeans(n_clusters=self.k)

        if self.clustering_method == "faiss":
            _ = self.clustering.cluster(features, verbose=self.verbose)
            labels = faiss_kmeans.arrange_clustering(self.clustering.images_lists)
        elif self.clustering_method == "sklearn":
            labels = self.clustering.fit_predict(features)

        if self.metrics:
            self.cluster_time.update(time.time() - end)

        return labels

    def create_pseudo_labeled_dataset(
        self, dataset: data.Dataset, labels: list, transform: transforms
    ) -> data.Dataset:
        """This function executes the PCA + k-Means algorithm, which are chosen when initializing the algorithm.

        Parameters
        ----------
        features: np.ndarray,
            Calculated features which are to be clustered.

        Returns
        -------
        Dataset which contains both the original data points with their obtained labels from the feature clustering.
        """
        return PseudoLabeledData(labels, dataset, transform)

    def print_results(
        self,
        epoch: int,
        losses: torch.Tensor,
        pred_accuracy: torch.Tensor,
        true_accuracy: torch.Tensor,
        pseudo_labels: list,
        dataset_labels: np.ndarray,
    ):
        """Function for better overview when printing epoch results after the training process has been completed.

        Parameters
        epoch: int,
            Current epoch.

        losses: torch.Tensor,
            Each loss after the training.

        accuracies: torch.Tensor,
            Accuracies when comparing the pseudo-labels and the resulting output when training.

        pseudo_labels: list,
            Pseudo-Labels of the training dataset.

        dataset_labels: np.ndarray,
            The actual labels of the dataset.
        """

        print("-" * 20, "Results", "-" * 20)
        print(f"Average loss: {torch.mean(losses)}")

        print(f"Pseudo vs Predicted Labels Accuracy: {pred_accuracy}")
        print(f"True vs Predicted Labels Accuracy: {true_accuracy}")
        print("-" * 50)

        print("Normalized Mutual Information Scores:")
        if len(self.cluster_logs) > 0:
            nmi_epoch = normalized_mutual_info_score(
                pseudo_labels, self.cluster_logs[-1]
            )
            print(f"- epoch {epoch} and current epoch {epoch+1}: {nmi_epoch}")
        else:
            nmi_epoch = 0.0

        nmi = normalized_mutual_info_score(dataset_labels, pseudo_labels)
        print(f"- True labels and computed features at epoch {epoch+1}: {nmi}")
        print("-" * 50)

        print("Label occurences:")
        if isinstance(dataset_labels, torch.Tensor):
            true_labels, count = torch.unique(dataset_labels, return_counts=True)
            print(f"- True labels: {dict(zip(true_labels.tolist(), count.tolist()))}")
        elif isinstance(dataset_labels, list):
            print(
                f"- True labels: {dict(sorted(collections.Counter(dataset_labels).items()))}"
            )
        print(
            f"- Computed labels: {dict(sorted(collections.Counter(pseudo_labels).items()))}"
        )

        print("-" * 50)

        if self.metrics:
            self.write_metrics(epoch, nmi, nmi_epoch)

        return

    def write_metrics(self, epoch: int, nmi: float, nmi_epoch: float):
        """Wrapper function to write metrics directly into a .csv file for data analytics.
        This function will create a metrics folder in the main directory and in addition a sub-folder for the dataset, if either or both don't exist.
        After, it creates, if not existing, a .csv file, add headers and appends the metrics after each epoch.

        The following metrics are stored, for each epoch:
            - NMI between true and predicted labels
            - NMI between previous and current epoch of the clustered feature labels
            - Execution time for each epoch, training, feature calculation, clustering and PCA reduction
            - Loss and Accuracy of the model training

        Parameters
        ----------
        epoch: int,
            Current epoch.

        nmi: float,
            The Normalized Mutual Information Score between the true and predicted labels.

        nmi_epoch: float,
            The Normalized Mutual Information Score between the current epoch clustered labels and of the previous epochs' clustered labels.
        """

        if not os.path.exists(BASE_METRICS):
            os.makedirs(BASE_METRICS)

        if not os.path.exists(f"{BASE_METRICS}{self.dataset_name}"):
            os.makedirs(f"{BASE_METRICS}{self.dataset_name}")

        # When the file doesn't exist, create it and add the header
        if not os.path.exists(self.metrics_file):
            if self.verbose:
                print(f"Creating metrics file at '{self.metrics_file}'.")
            with open(self.metrics_file, "w", newline="") as file:
                if self.metrics_metadata:
                    # Add metadata to further distinguish the different files in the future
                    file.write(f"#{self.metrics_metadata}\n")
                writer = csv.writer(file)
                writer.writerow(METRICS_HEADER)

        if self.verbose:
            print(f"Storing metrics of current epoch {epoch+1}...")

        with open(self.metrics_file, "a", newline="") as file:
            writer = csv.writer(file)
            # Add Metrics Row
            row = [
                epoch,
                torch.mean(self.loss_overall_avg.val).numpy(),
                torch.mean(self.contrastive_loss_overall_avg.val).numpy(),
                torch.mean(self.deep_cluster_loss_overall_avg.val).numpy(),
                torch.mean(self.accuracy_overall_avg.val).numpy(),
                torch.mean(self.true_accuracy_overall_avg.val).numpy(),
                nmi,
                nmi_epoch,
                self.epoch_time.sum,
                self.train_time.sum,
                self.features_time.sum,
                self.cluster_time.sum,
                self.pca_time.sum,
            ]
            writer.writerow(row)

        return

    def print_metrics(self, epoch: int):

        print("-" * 15, f" Metrics after {epoch+1} Epochs ", "-" * 15)
        print(
            f"- Feature time: {self.features_time.sum} [avg: {self.features_time.avg}]"
        )
        print(f"- PCA time: {self.pca_time.sum} [avg: {self.pca_time.avg}]")
        print(f"- Cluster time: {self.cluster_time.sum} [avg: {self.cluster_time.avg}]")
        print(f"- Training time: {self.train_time.sum} [avg: {self.train_time.avg}]")
        print(f"- Epoch time: {self.epoch_time.sum} [avg: {self.epoch_time.avg}]")
        print("-" * 60)
