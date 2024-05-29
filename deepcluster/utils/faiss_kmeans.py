import faiss
import numpy as np
from torch.utils import data
from .pseudo_labeled_dataset import PseudoLabeledData
from torchvision.transforms import Compose
import torch

class FaissKMeans():
    def __init__(self, k: int):
        """Wrapper for the Facebook AI Research Implementation for a KMeans algorithm optimized for GPUs.

        Parameter
        ---------
        k: int,
            Number of clusters.
        """
        self.k = k
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """Performs KMeans Clustering based on Facebooks AI Research Method

        Args:
            data (np.ndarray): _description_

        Returns:
            float: _description_
        """
        # Clustering
        labels = self.run_kmeans(data, self.k)
        
            
        return np.array(labels, dtype=int)
    
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
        
        return [int(n[0]) for n in I]