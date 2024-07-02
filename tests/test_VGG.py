from deepcluster.models.VGG import VGG16
import unittest
import torch


class TestVGG(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.num_classes = 1000
        self.input_dim = 3
        self.input_size = 256
        self.grayscale = True
        self.sobel = True
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

        self.assertEqual(output[0], self.batch_size)
        self.assertEqual(output[1], self.num_classes)
        self.assertFalse(torch.isnan(output).any())

    def test_VGG_greyscale(self):
        """
        Tests if the greyscale is applied properly
        """
        X = torch.randn(self.batch_size, self.input_dim, self.input_size, self.input_size)

        if self.grayscale:
            output_greyscale = self.model(X)

            self.assertEqual(output_greyscale.shape[1], 2)
            self.assertFalse(torch.isnan(output_greyscale).any())

    def test_VGG_sobel(self):
        """
                Tests if the Sobel filter is applied properly
                """
        X = torch.randn(self.batch_size, self.input_dim, self.input_size, self.input_size)

        if self.sobel:
            output_sobel = self.model(X)

            self.assertEqual(output_sobel.shape[1], 2)
            self.assertFalse(torch.isnan(output_sobel).any())
