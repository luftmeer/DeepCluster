import unittest
import torch.nn as nn
from deepcluster.utils.loss_functions import loss_function_loader


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.loss_function_loader = loss_function_loader
        self.invalid_loss = 'invalid'

    def test_loss_functions_loader(self):
        """
        Tests the loss_function_loader
        """
        l1_loss = self.loss_function_loader('L1')
        self.assertIsInstance(l1_loss, nn.L1Loss)

        l2_loss = self.loss_function_loader('L2')
        self.assertIsInstance(l2_loss, nn.MSELoss)

        mse_loss = self.loss_function_loader('MSE')
        self.assertIsInstance(mse_loss, nn.MSELoss)

        cross_entropy_loss = self.loss_function_loader('CrossEntropy')
        self.assertIsInstance(cross_entropy_loss, nn.CrossEntropyLoss)

    def test_loss_function_loader_invalid(self):
        """
        Tests loss_function_loader with an invalid loss
        """
        with self.assertRaises(ValueError) as context:
            loss_function_loader(self.invalid_loss)
            self.assertEqual(f"Selected loss function {self.invalid_loss} not supported.", str(context.exception))
