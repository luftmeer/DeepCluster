import unittest
from deepcluster.utils.benchmarking import Meter


class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test suite
        """
        self.meter = Meter()

    def test_initialization(self):
        """
        Tests the initialization of the Meter class
        """
        self.assertEqual(self.meter.val, 0)
        self.assertEqual(self.meter.avg, 0)
        self.assertEqual(self.meter.sum, 0)
        self.assertEqual(self.meter.count, 0)

    def test_update(self):
        """
        Tests the update function
        """
        self.meter.update(5)
        self.meter.update(15, 3)

        self.assertEqual(self.meter.val, 15)
        self.assertEqual(self.meter.avg, 12.5)
        self.assertEqual(self.meter.sum, 50)
        self.assertEqual(self.meter.count, 4)

    def test_reset(self):
        """
        Tests the reset function
        """
        self.meter.update(5)
        self.meter.reset()
        self.assertEqual(self.meter.val, 0)
        self.assertEqual(self.meter.avg, 0)
        self.assertEqual(self.meter.sum, 0)
        self.assertEqual(self.meter.count, 0)
