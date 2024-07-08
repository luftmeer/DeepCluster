from deepcluster.models.VGG import VGG16
import unittest
import torch


class TestVGG(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.num_classes = 1000
        self.input_dim = 1
        self.input_size = 224
        self.grayscale = False
        self.sobel = False
        self.batch_size = 8
        self.model = VGG16(self.num_classes, self.input_dim, self.input_size, self.grayscale, self.sobel)

    def test_VGG_init(self):
        """
        Tests if the AlexNet is correctly initialized.
        """
        self.assertIsInstance(self.model, VGG16)
        self.assertIsNotNone(self.model.features)
        self.assertIsNotNone(self.model.classifier)
        self.assertIsNotNone(self.model.top_layer)

    def test_VGG_forward(self):
        """
        Tests if the Forward pass of the VGG model is working properly
        """
        X = torch.randn(self.batch_size, self.input_dim, self.input_size, self.input_size)
        output = self.model(X)

        self.assertEqual(output.size(0), self.batch_size)
        self.assertEqual(output.size(1), self.num_classes)
        self.assertFalse(torch.isnan(output).any())
