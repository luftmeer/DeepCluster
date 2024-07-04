import unittest
from unittest.mock import MagicMock, patch
import torch
from torchvision import transforms
from deepcluster.utils.pseudo_labeled_dataset import PseudoLabeledData


class MockDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [torch.rand(3, 224, 224) for _ in range(100)]
        self.targets = list(range(100))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestPseudoLabeledData(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.pseudolabels = list(range(100))
        self.dataset = MockDataset()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        self.pseudo_labeled_data = PseudoLabeledData(self.pseudolabels, self.dataset, self.transform)

    def test_initialization(self):
        """
        Tests if the dataset is correctly initialized
        """
        self.assertEqual(len(self.pseudo_labeled_data.targets), 100)
        self.assertEqual(len(self.pseudo_labeled_data.dataset), 100)

        self.assertIsNotNone(self.pseudo_labeled_data.transform)
        self.assertEqual(repr(self.pseudo_labeled_data.transform), repr(self.transform))

    def test_len(self):
        """
        Tests if the dataset has the correct length
        """
        self.assertEqual(len(self.pseudo_labeled_data), 100)

    def test_getitem(self):
        """
        Tests if the getitem function works properly
        """
        image, pseudolabel, true_target = self.pseudo_labeled_data[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(pseudolabel, 0)
        self.assertEqual(true_target, 0)

