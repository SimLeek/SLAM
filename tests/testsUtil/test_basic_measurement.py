import unittest as ut
from util.basicMeasurement import BasicMeasurement
import numpy as np
from util.measureFunctions import gradMeasureFunction, measureFunction

class TestBasicMeasurement( ut.TestCase):
    def test_init(self):
        bm = BasicMeasurement(covariance=np.random.rand(8,1),
                              robotFeaturesDim=20,
                              envFeaturesDim=20,
                              measureFunction=20,
                              gradMeasureFunction=20,
                              detectionSize=20,
                              detectionCone=20)
        self.assertEqual(len(bm.covariance),8)
        self.assertEqual(bm.covariance.ndim, 2)

    def test_measure(self):
        bm = BasicMeasurement(covariance=np.random.rand(2, 2),
                              robotFeaturesDim=3,
                              envFeaturesDim=2,
                              measureFunction=measureFunction,
                              gradMeasureFunction=gradMeasureFunction,
                              detectionSize=20,
                              detectionCone=2*3.14159)
        distances, landmark_ids = bm.measure(state=np.zeros((7,1)))
        self.assertListEqual(distances.tolist(), [[0,0],[0,0]])
        self.assertListEqual(landmark_ids.tolist(), [0,1])

