import torch.utils
import torch.utils.data
from torchvision import transforms
from torch import optim
from torch import nn
from torch.backends import cudnn
from torch.utils import data
from sklearn.base import BaseEstimator
import numpy as np
from utils import kmeans
import os
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from torch import Tensor

# Base folder for checkpoints
BASE_CPT = './checkpoints/'


class DeepCluster(BaseEstimator):
    def __init__(self,
                 model: nn.Module,  # CNN Model
                 optim: optim.Optimizer,  # Optimizer for the parameters of the model
                 optim_tl: optim.Optimizer,  # Optimizer for the Top Layer Parameters
                 loss_criterion: object,  # PyTorch Loss Function
                 cluster_assign_tf: transforms,
                 dataset_name: str,  # Name of the dataset when saving checkpoints
                 checkpoint: str = None,  # Direct path to the checkpoint
                 epochs: int = 500,  # Training Epoch
                 batch_size: int = 256,
                 k: int = 1000,
                 verbose: bool = False,  # Verbose output while training
                 pca_reduction: int = 256,  # PCA reduction value for the amount of features to be kept
                 clustering_method: str = 'faiss',
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
            
        feature_computation: str, default='faiss',
            Which method should be used to calculate the features.
            
            faiss: Uses the implementation by Facebook AI Research which is also used by the authors, especially the k-Means algorithm.
            sklearn: Uses the PCA and k-Means implementation by scikit-learn.
        """
        self.model = model
        self.optimizer = optim
        self.optimizer_tl = optim_tl
        self.loss_criterion = loss_criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        self.pca = pca_reduction
        self.cluster_assign_transform = cluster_assign_tf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clustering_method = clustering_method
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        self.start_epoch = 0  # Start epoch, necessary when resuming from previous checkpoint
        self.cluster_logs = []

    def save_checkpoint(self, epoch: int):
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
        if not os.path.exists(BASE_CPT + '/' + self.dataset_name + '/'):
            os.makedirs(BASE_CPT + '/' + self.dataset_name + '/')

        # Store checkpoint
        if self.verbose:
            print(f'Saving the current checkpoint at epoch {epoch + 1}..')

        torch.save({
            'epoch': epoch + 1,
            # +1 since, when starting again, the algorithm should continue with the next epoch and not 'redo' this one
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_tl': self.optimizer_tl.state_dict(),
            'loss': self.loss_criterion,
            'cluster_logs': self.cluster_logs
        },
            f'{BASE_CPT}/{self.dataset_name}/{self.model}.cpt')

        return

    def load_checkpoint(self):
        """Helper Function to load the latest checkpoint of a model training.
        """
        if os.path.isfile(self.checkpoint):
            print(f'Loading checkpoint file \'{self.checkpoint}\'')
            checkpoint = torch.load(self.checkpoint)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer_tl.load_state_dict(checkpoint['optimizer_tl'])
            self.cluster_logs = checkpoint['cluster_logs']
            print(f'Loaded checkpoint at epoch {self.start_epoch}')
            del checkpoint
        else:
            print(f'No checkpoint found at {self.checkpoint}')

        return

    def fit(self, data: data.DataLoader, remove_tl: bool = False):
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.to(self.device)

        # Checkpoint file path given, load checkpoint
        if self.checkpoint:
            self.load_checkpoint()

        cudnn.benchmark = True
        fd = int(self.model.top_layer.weight.size()[1])

        # Set KMeans Clustering
        if self.clustering_method == 'faiss':
            clustering = kmeans.KMeans(self.k)
        elif self.clustering_method == 'sklearn':
            clustering = KMeans(n_clusters=self.k)

        for epoch in range(self.start_epoch, self.epochs):
            if self.verbose: print(f'{"=" * 25} Epoch {epoch + 1} {"=" * 25}')

            if remove_tl:
                # Remove head
                self.model.top_layer = None
                self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

            # Compute Features
            features = self.compute_features(data)

            # Cluster features
            if self.clustering_method == 'faiss':
                _ = clustering.fit(features, self.pca)
            elif self.clustering_method == 'sklearn':
                # PCA reduction
                reduced_features = PCA(n_components=self.pca, whiten=True).fit_transform(features)
                labels = clustering.fit_predict(reduced_features)
                images_list = [[] for i in range(self.k)]
                for i in range(len(features)):
                    images_list[labels[i]].append(i)

            # Assign Pseudo-Labels
            if self.clustering_method == 'faiss':
                train_dataset = clustering.cluster_assign(clustering.images_list, data.dataset,
                                                          self.cluster_assign_transform)
            elif self.clustering_method == 'sklearn':
                train_dataset = kmeans.KMeans.cluster_assign(images_list, data.dataset, self.cluster_assign_transform)

            # Sampler -> Random
            # TODO: Find a solution for a Uniform Sampling / When Found -> Benchmark against a simple random Sampling
            sampler = torch.utils.data.RandomSampler(train_dataset)

            # Create Training Dataset
            train_data = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=True,
            )

            if remove_tl:
                # Add Top Layer
                classifiers = list(self.model.classifier.children())
                classifiers.append(nn.ReLU(inplace=True).to(self.device))
                self.model.classifier = nn.Sequential(*classifiers)
                self.model.top_layer = nn.Linear(fd, len(clustering.images_list))
                self.model.top_layer.weight.data.normal_(0, 0.01)
                self.model.top_layer.bias.data.zero_()
                self.model.top_layer.to(self.device)

            loss = self.train(train_data)

            print(f'Classification Loss: {loss}')
            # print(f'Clustering Loss: {clustering_loss}')

            if len(self.cluster_logs) > 0:
                nmi = normalized_mutual_info_score(train_data.dataset.targets, self.cluster_logs[-1])

                print(f'NMI score: {nmi}')

            if self.clustering_method == 'faiss':
                self.cluster_logs.append(train_data.dataset.targets)
            elif self.clustering_method == 'sklearn':
                self.cluster_logs.append(train_data.dataset.targets)

            if self.verbose: print('Creating new checkpoint..')
            self.save_checkpoint(epoch)
            if self.verbose: print('Finished storing checkpoint')

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


    def train(self, train_data: data.DataLoader) -> float:
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

        losses = torch.zeros(len(train_data), dtype=torch.float32, requires_grad=False)
        accuracies = torch.zeros(len(train_data), dtype=torch.float32, requires_grad=False)
        different_classes = set()
        for i, (input, target) in tqdm(enumerate(train_data), desc='Training', total=len(train_data)):
            if self.device.type == 'cuda':
                input, target = input.cuda(), target.cuda()
            input.requires_grad = True

            # Forward pass
            output = self.model(input)
            loss = self.loss_criterion(output, target)

            # check Nan Loss
            if torch.isnan(loss):
                print("targets", target)
                print("Output", output)
                print("Input", input)
                print("Nan Loss", loss)

                break

            # add the loss to the losses tensor
            losses[i] = loss.item()

            # calculate accuracy and add it to accuracies tensor
            _, predicted = output.max(1)
            accuracies[i] = predicted.eq(target).sum().item() / target.size(0)

            # add the different classes to the set
            different_classes.update(target.cpu().numpy())

            # Backward pass and optimize
            self.optimizer.zero_grad()
            self.optimizer_tl.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_tl.step()

            # Free up GPU memory
            del input, target, output, loss
            torch.cuda.empty_cache()

        print("-" * 20, "Results", "-" * 20)
        print("These are the losses")
        print(losses)
        print("-" * 50)
        print("This is the accuracy")
        print(torch.mean(accuracies))
        print("-" * 50)
        print("These are the different classes")
        print(different_classes)
        print("-" * 50)
        return torch.mean(losses)

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
        for i, (input, _) in tqdm(enumerate(data), desc='Computing Features', total=len(data)):
            input = input.to(self.device)

            input.requires_grad = True
            aux = self.model(input).data.cpu().numpy()

            if i == 0:
                features = np.zeros((len(data.dataset), aux.shape[1]), dtype=np.float32)

            aux = aux.astype(np.float32)
            if i < len(data) - 1:
                features[i * self.batch_size: (i + 1) * self.batch_size] = aux
            else:
                # Rest of the data
                features[i * self.batch_size:] = aux

            # Free up GPU memory
            del input, aux
            torch.cuda.empty_cache()

        return features
