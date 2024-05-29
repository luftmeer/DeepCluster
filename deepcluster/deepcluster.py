import torch.utils
import torch.utils.data
from torchvision import transforms
from torch import optim
from torch import nn
from torch.backends import cudnn
from torch.utils import data
from sklearn.base import BaseEstimator
import numpy as np
from utils import faiss_kmeans
from utils.pseudo_labeled_dataset import PseudoLabeledData
import os
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import collections
import faiss


import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from torch import Tensor

# Base folder for checkpoints
BASE_CPT = './checkpoints/'


class DeepCluster(BaseEstimator):
    def __init__(
                self,
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
                pca_method: str = 'faiss',
                pca_whitening: bool = True
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
        self.pca_method = pca_method
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        self.start_epoch = 0  # Start epoch, necessary when resuming from previous checkpoint
        self.cluster_logs = []
        self.pca_whitening = pca_whitening
        
        # Set clustering algorithm
        if self.clustering_method == 'faiss':
            self.clustering = faiss_kmeans.FaissKMeans(self.k)
        elif self.clustering_method == 'sklearn':
            self.clustering = KMeans(n_clusters=self.k)
            

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
            print(f'Loaded checkpoint at epoch {self.start_epoch+1}')
            del checkpoint
        else:
            print(f'No checkpoint found at {self.checkpoint}')

        return

    def fit(self, data: data.DataLoader):
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.to(self.device)

        # Checkpoint file path given, load checkpoint
        if self.checkpoint:
            self.load_checkpoint()

        cudnn.benchmark = True
        fd = int(self.model.top_layer.weight.size()[1])

        for epoch in range(self.start_epoch, self.epochs):
            if self.verbose: print(f'{"=" * 25} Epoch {epoch + 1} {"=" * 25}')

            
            # Remove head
            self.model.top_layer = None
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

            # Compute Features
            features = self.compute_features(data)
            
            # PCA reduce features
            features = self.pca_reduction(features)

            # Cluster features and obtain the resulting labels
            labels = self.apply_clustering(features)
            
            # Create the training data set
            train_dataset = self.create_pseudo_labeled_dataset(data.dataset, labels, self.cluster_assign_transform)
            
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

            # Add Top Layer
            classifiers = list(self.model.classifier.children())
            classifiers.append(nn.ReLU(inplace=True).to(self.device))
            self.model.classifier = nn.Sequential(*classifiers)
            self.model.top_layer = nn.Linear(fd, self.k)
            self.model.top_layer.weight.data.normal_(0, 0.01)
            self.model.top_layer.bias.data.zero_()
            self.model.top_layer.to(self.device)

            loss = self.train(train_data)

            print(f'Classification Loss: {loss}')
            # print(f'Clustering Loss: {clustering_loss}')

            print('-'*50)
            print('Normalized Mutual Information Scores:')
            if len(self.cluster_logs) > 0:
                nmi = normalized_mutual_info_score(train_data.dataset.targets, self.cluster_logs[-1])

                print(f'- epoch {epoch} and current epoch {epoch+1}: {nmi}')
            
            print(f'- True labels and computed features at epoch {epoch+1}: {normalized_mutual_info_score(data.dataset.targets, train_data.dataset.targets)}')
            print('-'*50)
            
            print('Label occurences:')
            true_labels, count = torch.unique(data.dataset.targets, return_counts=True)
            print(f'- True labels: {dict(zip(true_labels.tolist(), count.tolist()))}')
            print(f'- Computed labels: {dict(sorted(collections.Counter(train_data.dataset.targets).items()))}')
            
            print('-'*50)
            
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
        self.model.eval()
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
        if self.pca_method == 'faiss':
            _, dim = features.shape
            features = features.astype(np.float32)
            
            # PCA transformation + whitening
            if self.pca_whitening:
                whitening_value = -0.5
            else:
                whitening_value = 0.0
            mat = faiss.PCAMatrix(dim, self.pca, eigen_pwer=whitening_value)
            mat.train(features)
            assert mat.is_trained
            features = mat.apply(features)
            
        elif self.pca_method == 'sklearn':
            features = PCA(n_components=self.pca, whiten=self.pca_whitening).fit_transform(features)

        # L2-normalization
        rows = np.linalg.norm(features, axis=1)
        features = np.divide(features, rows.reshape((rows.shape[0], 1)))

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
        if self.clustering_method == 'faiss':
            labels = self.clustering.fit(features)
        elif self.clustering_method == 'sklearn':
            labels = self.clustering.fit_predict(features)
        
        return labels
        
    def create_pseudo_labeled_dataset(self, dataset: data.DataSet, labels: list, transform: transforms) -> data.DataSet:
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