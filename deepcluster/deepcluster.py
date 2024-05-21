import torch
import torch.nn as nn
import torch.optim.optimizer
import clustpy

from datasets import ImageNetDataset, transform_for_alexnet
from torchvision.models import alexnet, AlexNet
from sklearn.cluster import KMeans



class DeepCluster:
    """
    Basic Implementation of the DeepCluster Method using AlexNet for feature extraction, feed forward
    layers for classification and Kmeans for clustering.
    """
    def __init__(self, n_classes=1000):

        ## Classifier
        self.ConvNet = AlexNet()

        ## Clustering algorithm
        self.cluster = KMeans(n_clusters=n_classes)

        ## Pre-processing
        self.transform = transform_for_alexnet

    def predict(self, img: torch.Tensor):
        ## Apply ConvNet
        img = self.transform(img)
        prediction = self.ConvNet(img)

        ## Return prediction with pseudo-labels from clustering
        pseudo_labels = self.cluster.predict(prediction)

        return prediction, pseudo_labels

    def fit(self, X):
        pass
