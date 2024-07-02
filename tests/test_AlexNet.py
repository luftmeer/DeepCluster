from deepcluster.models.AlexNet import AlexNet
import unittest
import torch


class TestAlexNet(unittest.TestCase):
    def setUp(self):
        self.input_dim = 3
        self.num_classes = 1000
        self.grayscale = True
        self.sobel = True
        self.batch_size = 8
        self.height = 256
        self.width = 256
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

        self.assertEqual(output[0], self.batch_size)
        self.assertEqual(output[1], self.num_classes)
        self.assertFalse(torch.isnan(output).any())

    def test_AlexNet_greyscale(self):
        """
        Tests if the greyscale is applied properly
        """
        X = torch.randn(self.batch_size, self.input_dim, self.height, self.width)

        if self.grayscale:
            output_greyscale = self.model(X)

            self.assertEqual(output_greyscale.shape[1], 2)
            self.assertFalse(torch.isnan(output_greyscale).any())

    def test_AlexNet_sobel(self):
        """
        Tests if the Sobel filter is applied properly
        """
        X = torch.randn(self.batch_size, self.input_dim, self.height, self.width)

        if self.sobel:
            output_sobel = self.model(X)

            self.assertEqual(output_sobel.shape[1], 2)
            self.assertFalse(torch.isnan(output_sobel).any())


def test_AlexNet_init():
    model = AlexNet()
    assert type(model) == AlexNet
