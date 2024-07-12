import unittest
from unittest.mock import Mock, patch
from deepcluster.utils.augmentations import get_augmentation_fn
import argparse


class TestAugmentations(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.args = Mock(spec=argparse.Namespace)
        self.args.augmentation_resize = 256
        self.args.augmentation_random_crop = 224
        self.args.augmentation_random_rotation = 30
        self.args.augmentation_random_horizontal_flip = True
        self.args.augmentation_random_vertical_flip = True
        self.args.augmentation_color_jitter = True
        self.args.augmentation_random_autocontrast = True
        self.args.augmentation_random_equalize = True

        self.get_augmentation_fn = None

    @patch('deepcluster.utils.augmentations.transforms.Resize')
    @patch('deepcluster.utils.augmentations.transforms.RandomCrop')
    @patch('deepcluster.utils.augmentations.transforms.RandomRotation')
    @patch('deepcluster.utils.augmentations.transforms.RandomHorizontalFlip')
    @patch('deepcluster.utils.augmentations.transforms.RandomVerticalFlip')
    @patch('deepcluster.utils.augmentations.transforms.ColorJitter')
    @patch('deepcluster.utils.augmentations.transforms.RandomAutocontrast')
    @patch('deepcluster.utils.augmentations.transforms.RandomEqualize')
    @patch('deepcluster.utils.augmentations.transforms.ToTensor')
    @patch('deepcluster.utils.augmentations.transforms.Compose')
    def test_get_augmentation_fn(self, mock_compose, mock_to_tensor, mock_random_equalize, mock_random_autocontrast,
                                 mock_color_jitter, mock_random_vertical_flip, mock_random_horizontal_flip,
                                 mock_random_rotation, mock_random_crop, mock_resize):
        """
        Tests the get_augmentation_fn function if every transformation is called
        """
        self.get_augmentation_fn = get_augmentation_fn(self.args)

        mock_compose.assert_called_once_with([
            mock_resize(),
            mock_random_crop(),
            mock_random_rotation(),
            mock_random_horizontal_flip(),
            mock_random_vertical_flip(),
            mock_color_jitter(),
            mock_random_autocontrast(),
            mock_random_equalize(),
            mock_to_tensor()
        ])


        self.assertEqual(self.get_augmentation_fn, mock_compose())