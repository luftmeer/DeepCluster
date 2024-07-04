from deepcluster.models.AlexNet import AlexNet
import unittest
import torch


class TestAlexNet(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.input_dim = 1
        self.num_classes = 1000
        self.grayscale = False
        self.sobel = False
        self.batch_size = 256
        self.height = 224
        self.width = 224
        self.model = AlexNet(self.input_dim, self.num_classes, self.grayscale, self.sobel)

    def test_AlexNet_init(self):
        """
        Tests if the AlexNet is correctly initialized.
        """
        self.assertIsInstance(self.model, AlexNet)
        self.assertIsNotNone(self.model.features)
        self.assertIsNotNone(self.model.classifier)
        self.assertIsNotNone(self.model.top_layer)

    def test_AlexNet_forward(self):
        """
        Tests if the Forward pass of the AlexNet model is working properly
        """
        X = torch.randn(self.batch_size, self.input_dim, self.height, self.width)
        output = self.model(X)

        self.assertEqual(output.size(0), self.batch_size)
        self.assertEqual(output.size(1), self.num_classes)
        self.assertFalse(torch.isnan(output).any())
