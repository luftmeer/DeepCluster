import unittest
from torch.utils.data import DataLoader, Dataset
from deepcluster.utils.UnifiedSampler import UnifLabelSampler


class MockDataset(Dataset):
    def __init__(self, size=100):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestUnifiedSampler(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.images_lists = {
            0: list(range(10)),
            1: list(range(10, 25)),
            2: list(range(25, 30))
        }

        self.sampler = UnifLabelSampler(50, self.images_lists)

    def test_generate_indexes_epoch(self):
        """
        Tests if the function generate_indexes_epoch works properly
        """
        counts = [0] * len(self.images_lists)

        for idx in self.sampler.generate_indexes_epoch():
            if idx < 10:
                counts[0] += 1
            elif idx < 25:
                counts[1] += 1
            else:
                counts[2] += 1

        self.assertTrue(all(x > 0 for x in counts))  # Ensure every cluster is sampled

    def test_length(self):
        """
        Tests if the dataset has the correct length
        """
        self.assertEqual(len(self.sampler), 50)  # Expecting length to match N

    def test_empty_clusters(self):
        """
        Tests that generate_indexes_epoch doesn't sample from empty clusters
        """
        # Add test case for handling empty clusters
        self.images_lists[3] = []
        sampler = UnifLabelSampler(50, self.images_lists)

        self.assertFalse(any(idx in self.images_lists[3] for idx in sampler.generate_indexes_epoch()))
