import unittest as ut
import util.measureFunctions as utmf
import numpy as np
import math

class TestMeasureFunctions(ut.TestCase):
    def testMeasureFunction(self):
        norm, angle = utmf.measureFunction(rState=np.array([[1],[2],[3.14]]), landmark=np.array([[3],[5]]))
        self.assertEqual(norm, math.sqrt((3-1)**2 + (5-2)**2))
        self.assertEqual(angle, math.atan2(5-2,3-1) - 3.14)

    def testGradMeasureFunction(self):
        utmf.dimension=30
        grad = utmf.gradMeasureFunction(rState=np.array([[1],[2],[3.14]]), landmark=np.array([[3],[5]]), ldmIndex=6)
        self.assertListEqual(grad[:2, 0].tolist(),(np.array([1-3,2-5])/np.linalg.norm([1-3, 2-5])).tolist())
        self.assertListEqual(grad[6:6+2, 0].tolist(), (np.array([3-1, 5-2]) / np.linalg.norm([3-1, 5-2])).tolist())
        self.assertListEqual(grad[:2, 1].tolist(), (np.array([5-2, 1-3]) / (np.linalg.norm([5-2, 1-3])**2)).tolist())
        self.assertListEqual(grad[6:6+2, 1].tolist(),
                             (np.array([2-5, 3-1]) / (np.linalg.norm([2-5, 3-1]) ** 2)).tolist())
