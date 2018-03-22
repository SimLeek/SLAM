import unittest as ut
from util.basicMovement import BasicMovement
from util.measureFunctions import measureFunction
import numpy as np

class TestBasicMovement(ut.TestCase):
    def tes_init(self):
        bm = BasicMovement(20,20,np.random.rand(8,1), 20)
        self.assertEqual(len(bm.covariance),8)
        self.assertEqual(bm.covariance.ndim, 2)

    def test__get_noise(self):
        covariance_matrix = np.ndarray(
            (3,3)
        )
        covariance_matrix[0,0]=-1
        covariance_matrix[0, 1] = 1
        covariance_matrix[1, 0] = 3
        covariance_matrix[1, 1] = 5
        bm = BasicMovement(20, 20, covariance_matrix, 20)
        self.assertEqual(len(bm._get_noise()),3)
        self.assertEqual(len(bm._get_noise()[0]), 1)
        # can't test produced values: random

    def test_noisy_move2(self):
        bm = BasicMovement(20, 20, np.random.rand(8,1), 20)
        self.assertListEqual(bm._noisy_move2(np.random.rand(3,3), np.array([1,2,3]), .5).tolist(),
                             [[1.5, 2.5, 3.5] for _ in range(3)])

    def test_exact_move(self):
        bm = BasicMovement(20, 20, np.random.rand(8, 1), 20)
        self.assertListEqual(bm.exact_move(np.array([[999.0],[999.0],[3.0]]), (1.0,2.0)).tolist(),
                             [[-0.9899924966004454],[0.1411200080598672],[2.0]])

    def test_noisy_move(self):
        bm = BasicMovement(20, 20, np.random.rand(8, 1), measureFunction)
        bm._noisy_move(state=np.zeros((3,1)),
                             idealMove=np.array([[1],[2],[3]]),
                             noise=np.array([[.05],[.04],[.03]]))

