import unittest
from unittest.mock import patch, MagicMock
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from deepcluster.utils.datasets import BASE_TRANSFORM, NORMALIZATION, dataset_loader


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.data_dir = '../../../images'
        self.batch_size = 32

    @patch('torchvision.datasets.CIFAR10')
    def test_cifar10_loader(self, mock_cifar10):
        """
        Test loading CIFAR10 with the correct transform
        """
        dataset_name = 'CIFAR10'
        expected_transform = transforms.Compose(BASE_TRANSFORM + [NORMALIZATION[dataset_name]])

        mock_cifar10.return_value = MagicMock(spec=DataLoader)
        loader = dataset_loader(dataset_name, self.data_dir, self.batch_size)

        args, kwargs = mock_cifar10.call_args
        self.assertEqual(kwargs['root'], self.data_dir)
        self.assertTrue(kwargs['train'])
        self.assertTrue(kwargs['download'])
        self.assertEqual(repr(kwargs['transform']), repr(expected_transform))

        mock_cifar10.assert_called_once()

    @patch('torchvision.datasets.MNIST')
    def test_mnist10_loader(self, mock_mnist10):
        """
        Test loading MNIST with the correct transform
        """
        dataset_name = 'MNIST'
        expected_transform = transforms.Compose(BASE_TRANSFORM + [NORMALIZATION[dataset_name]])

        mock_mnist10.return_value = MagicMock(spec=DataLoader)
        loader = dataset_loader(dataset_name, self.data_dir, self.batch_size)

        args, kwargs = mock_mnist10.call_args
        self.assertEqual(kwargs['root'], self.data_dir)
        self.assertTrue(kwargs['train'])
        self.assertTrue(kwargs['download'])
        self.assertEqual(repr(kwargs['transform']), repr(expected_transform))

        mock_mnist10.assert_called_once()
