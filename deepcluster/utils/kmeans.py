import faiss
import numpy as np
from torch.utils import data
from .pseudo_labeled_dataset import PseudoLabeledData
from torchvision.transforms import Compose
import torch

class KMeans():
    def __init__(self, k: int):
        """Wrapper for the Facebook AI Research Implementation for a KMeans algorithm optimized for GPUs.

        Parameter
        ---------
        k: int,
            Number of clusters.
        """
        self.k = k
        
    def fit(self, data: np.ndarray, pca: int=256) -> float:
        """Performs KMeans Clustering based on Facebooks AI Research Method

        Args:
            data (np.ndarray): _description_

        Returns:
            float: _description_
        """
        # PCA-reducting, whitening and L2-normalization
        xb = self.preprocess_features(data, pca)
        
        # Clustering
        I, loss = self.run_kmeans(xb, self.k)
        self.images_list = [[] for i in range(self.k)]
        
        for i in range(len(data)):
            self.images_list[I[i]].append(i)
            
        return loss
    
    @staticmethod
    def preprocess_features(data: np.ndarray, pca: int=256) -> np.ndarray:
        """Preprocess an array of features.

        Args:
            data (np.ndarray): _description_
            pca (int, optional): _description_. Defaults to 256.

        Returns:
            np.ndarray: _description_
        """
        _, dim = data.shape
        data = data.astype(np.float32)
        
        # PCA-Whitening with Faiss
        mat = faiss.PCAMatrix(dim, pca, eigen_power=-0.5)
        mat.train(data)
        assert mat.is_trained
        data = mat.apply(data)
        
        # L2-normalization
        rows = np.linalg.norm(data, axis=1)
        data = np.divide(data, rows.reshape((rows.shape[0], 1)))
        
        return data
    
    @staticmethod
    def run_kmeans(data: np.ndarray, k: int) -> tuple:
        """_summary_

        Args:
            data (np.ndarray): _description_
            k (int): _description_

        Returns:
            tuple: _description_
        """
        
        N, dim = data.shape
        
        # Faiss K-Means
        clus = faiss.Clustering(dim, k)
        
        clus.seed = np.random.randint(1234)
        
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        
        # Prepare index either for CPU or GPU computation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            index = faiss.IndexFlatL2(dim)
        
        elif device.type == 'cuda':
            res = faiss.StandardGpuResources() # Declare GPU Resource
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = False
            flat_config.device = 0
            index = faiss.GpuIndexFlatL2(res, dim, flat_config)
        
        else: # Device unkown
            raise TypeError(f'Unkown device: {device.type}')
        
        # Perform training
        clus.train(data, index)
        _, I = index.search(data, 1)
        losses = faiss.vector_to_array(clus.centroids)
        
        return [int(n[0]) for n in I], losses[-1]
    
    @staticmethod
    def cluster_assign(image_list: list, data: data.Dataset, transform: Compose) -> PseudoLabeledData:
        """_summary_

        Args:
            image_list (list): _description_
            data (data.Dataset): _description_
            transform (Compose): _description_

        Returns:
            PseudoLabeledData: _description_
        """
        pseudolabels = []
        image_indexes = []
        for cluster, images in enumerate(image_list):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))
            
        return PseudoLabeledData(image_indexes, pseudolabels, data, transform)