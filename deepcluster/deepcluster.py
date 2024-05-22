import torch.utils
import torch.utils.data
#from deepcluster.AlexNet import AlexNet
from torchvision import transforms
from torch import optim
from torch import nn
from torch.backends import cudnn
from torch.utils import data
from sklearn.base import BaseEstimator
import numpy as np
import faiss
from utils import kmeans
import os
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

import torch
from sklearn.cluster import KMeans

class DeepCluster(BaseEstimator):
    def __init__(self,
                model: nn.Module, # CNN Model
                #data: data.DataLoader, # Normalized Dataset
                optim: optim.Optimizer, # Optimizer for the parameters of the model
                optim_tl: optim.Optimizer, # Optimizer for the Top Layer Parameters
                loss_criterion: object, # PyTorch Loss Function
                cluster_assign_tf: transforms,
                epochs: int=500, # Training Epoch
                batch_size: int=256,
                k: int=1000,
                verbose: bool=False, # Verbose output while training
                pca_reduction: int=256, # PCA reduction value for the amount of features to be kept
                ):
        self.model = model
        #self.data = data
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
    
    def fit(self, data: data.DataLoader, validation_data: data.DataLoader):
        #TODO: Load Checkpoint implementation
        
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.to(self.device)
        cudnn.benchmark = True
        fd = int(self.model.top_layer.weight.size()[1])
        
        # Set KMeans Clustering
        clustering = kmeans.KMeans(self.k)
        for epoch in range(self.epochs):
            if self.verbose: print(f'{"="*25} Epoch {epoch + 1} {"="*25}')
            # Remove head
            self.model.top_layer = None
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
            
            # Compute Features
            features = self.compute_features(data)
            
            # Cluster features
            clustering_loss = clustering.fit(features, self.pca)
            
            # Assign Pseudo-Labels
            train_dataset = clustering.cluster_assign(clustering.images_list, data.dataset, self.cluster_assign_transform)
            
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

            # Add Top Layer
            classifiers = list(self.model.classifier.children())
            classifiers.append(nn.ReLU(inplace=True).cuda())
            self.model.classifier = nn.Sequential(*classifiers)
            self.model.top_layer = nn.Linear(fd, len(clustering.images_list))
            self.model.top_layer.weight.data.normal_(0, 0.01)
            self.model.top_layer.bias.data.zero_()
            self.model.top_layer.cuda()
            
            loss = self.train(train_data)
            
            print(f'Classification Loss: {loss}')
            print(f'Clustering Loss: {clustering_loss}')
            
            with torch.no_grad():
                correct = 0
                total = 0
                for _, (images, targets) in tqdm(enumerate(validation_data), 'Validating', total=len(validation_data)):
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    del images, targets, outputs
            print(f'Accuracy of the network on the validation images: {100 * correct / total}')

    def predict(self):
        pass
    
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
        print("-"*20, "Results", "-"*20)
        print("These are the losses")
        print(losses)
        print("-"*50)
        print("This is the accuracy")
        print(torch.mean(accuracies))
        print("-"*50)
        print("These are the different classes")
        print(different_classes)
        print("-"*50)
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
                features = np.zeros((len(data.sampler.indices), aux.shape[1]), dtype=np.float32)
                print(f'{features.shape=}')
                
            aux = aux.astype(np.float32)
            print(f'{aux.shape=}\n{aux=}')
            if i < len(data) - 1:
                features[i*self.batch_size: (i+1)*self.batch_size] = aux
            else:
                # Rest of the data
                print(f'{i=}')
                features[i*self.batch_size:] = aux
                
            # Free up GPU memory
            del input, aux
            torch.cuda.empty_cache()
                
        return features