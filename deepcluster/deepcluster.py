import torch.utils
import torch.utils.data
#from deepcluster.AlexNet import AlexNet
from torch import optim
from torch import nn
from torch.utils import data
from sklearn.base import BaseEstimator
import numpy as np
import faiss
from .utils import kmeans
import os
from sklearn.metrics import normalized_mutual_info_score


import torch
from sklearn.cluster import KMeans

class DeepCluster(BaseEstimator):
    def __init__(self,
                model: nn.Module, # CNN Model
                #data: data.DataLoader, # Normalized Dataset
                optim: optim.Optimizer, # Optimizer for the parameters of the model
                optim_tl: optim.Optimizer, # Optimizer for the Top Layer Parameters
                loss_criterion: object, # PyTorch Loss Function
                epochs: int=500, # Training Epoch
                batch_size: int=256,
                k: int=1000,
                verbose: bool=False, # Verbose output while training
                ):
        self.model = model
        #self.data = data
        self.optimizer = optim
        self.optimizer_tl = optim_tl
        self.loss = loss_criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        
    def save_checkpoint(self) -> bool:
        """Helper Function to contiously store a checkpoint of the current state of the CNN training

        Returns
        -------
            bool: If storing was successful or not.
        """
        pass
    
    def load_checkpoint(self):
        """Helper Function to load the latest checkpoint of a model training.
        
        Returns
        -------
            #TODO tbd
        """
        pass
    
    def fit(self, data: data.DataLoader):
        #TODO: Load Checkpoint implementation
        
        # Set KMeans Clustering
        clustering = kmeans.KMeans(self.k)
        
        for epoch in range(self.epochs):
            # Compute Features
            features = self.compute_features(data)
            
            # Cluster features
            clustering_loss = clustering.fit(features)
            
            # Assign Pseudo-Labels
            train_dataset = clustering.cluster_assign(clustering.images_list, data.dataset)
            
            # Sampler -> Random
            # TODO: Find a solution for a Uniform Sampling / When Found -> Benchmark against a simple random Sampling
            sampler = torch.utils.data.RandomSampler(data, num_samples=len(data.dataset))
            
            # Create Training Dataset
            train_data = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=True,
                )

            loss = self.train(train_data)


    def predict(self):
        pass
    
    def train(self, train_data: torch.Tensor) -> float:
        """_summary_

        Parameters
        ----------
            train_data (torch.Tensor): _description_

        Returns
        -------
            float: _description_
        """
        # Set model to train mode
        self.model.train()

        losses = []
        for i, (input, target) in enumerate(train_data):
            # TODO: Add checkpoint save
            
            input = input.requires_grad(True)
            target = target.requires_grad(True)
            
            output = self.model(input)
            loss = self.loss(output, target)
            losses.append(loss.datta[0] * input.tensor_size(0))
            
            # Optimize
            self.optimizer.zero_grad()
            self.optimizer_tl.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            self.optimizer_tl.step()
        
        return np.mean(losses)
    
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
        if self.verbose:
            print('Compute Features')
        
        for i, (input, _) in enumerate(data):
            input.requires_grad = True
            aux = self.model(input).data.cpu().numpy()
            
            if i == 0:
                features = np.zeros((len(data.dataset, aux.shape[1])), dtype='np.float32')
                
            aux = aux.astype('np.float32')
            if i < len(data) - 1:
                features[i*self.batch_size: (i+1)*self.batch_size] = aux
            else:
                # Rest of the data
                features[i*self.batch_size:] = aux
                
        return features