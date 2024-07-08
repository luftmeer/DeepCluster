import unittest
import torch
from torch import nn, optim
from deepcluster.utils.optimizer import optimizer_loader


class TestOptimizerLoader(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.parameters = [nn.Parameter(torch.randn(2, 2, requires_grad=True))]

    def test_sgd_optimizer(self):
        """
        Tests if the SGD is correctly initialized
        """
        sgd_optimizer = optimizer_loader('SGD', self.parameters, lr=0.01, momentum=0.9)

        self.assertIsInstance(sgd_optimizer, optim.SGD)
        self.assertEqual(sgd_optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(sgd_optimizer.param_groups[0]['momentum'], 0.9)

    def test_adam_optimizer(self):
        """
        Tests if the Adam is correctly initialized
        """
        adam_optimizer = optimizer_loader('Adam', self.parameters, lr=0.001, betas=(0.9, 0.999))

        self.assertIsInstance(adam_optimizer, optim.Adam)
        self.assertEqual(adam_optimizer.param_groups[0]['lr'], 0.001)
        self.assertEqual(adam_optimizer.param_groups[0]['betas'], (0.9, 0.999))

    def test_unsupported_optimizer(self):
        """
            Tests for unknown optimizer
        """
        with self.assertRaises(ValueError) as context:
            optimizer_loader('RMSprop', self.parameters)
        self.assertFalse('RMSprop ist not supported.' in str(context.exception))

    def test_missing_parameters(self):
        """
            Tests with not parameters
        """
        with self.assertRaises(TypeError):
            optimizer_loader('SGD')


