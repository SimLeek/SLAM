import unittest as ut
from slam2d.seifModel import SEIFModel
import numpy as np
from util.basicMeasurement import BasicMeasurement
import util.measureFunctions as meas

class TestSeifModel( ut.TestCase ):
    def testInit(self):
        seif = SEIFModel(dimension=2,
                         robotFeaturesDim=2,
                         envFeaturesDim=3,
                         motionModel=4,
                         mesModel=5,
                         covMes=np.array([[1,2],[3,4]]),
                         muInitial=np.array([[7,2], [7,2]]),
                         maxLinks=8)

        self.assertAlmostEqual(seif.invZ[0,0], -2.0) ; self.assertAlmostEqual(seif.invZ[0,1], 1.0)
        self.assertAlmostEqual(seif.invZ[1, 0], 1.5) ; self.assertAlmostEqual(seif.invZ[1, 1], -.5)

        self.assertListEqual(seif.Sx.tolist(), [[1,0],
                                                [0,1]])
        self.assertListEqual(seif.b.tolist(), [[7,7],
                                               [2,2]])
        self.assertListEqual(seif.H.tolist(), [[1,0],
                                               [0,1]])

    def make_seif(self):
        meas.dimension = 3 + 3 * 2

        bm = BasicMeasurement(covariance=np.random.rand(2, 2),
                              robotFeaturesDim=3,
                              envFeaturesDim=2,
                              measureFunction=meas.measureFunction,
                              gradMeasureFunction=meas.gradMeasureFunction,
                              detectionSize=20,
                              detectionCone=2 * 3.14159)

        seif = SEIFModel(dimension=9, #todo: represents number of numbers to represent all poses and positions. Should be calculated internally
                         robotFeaturesDim=3,  # todo: always 3
                         envFeaturesDim=2,  # todo: always 2
                         motionModel=4,  # todo: fix
                         mesModel=bm,  # todo: subclass, not wtf this is
                         covMes=np.array([[1, 2], [3, 4]]),
                         muInitial=np.array([[8], [2], [8],
                                                          [2], [2],
                                                          [7], [7],
                                                          [2], [7]]), # todo: limit to equal 'dimension' size or None
                         maxLinks=2) # todo: limit based on 'dimension'
        return seif

    def test_get_mean_measurement_params(self):
        seif = self.make_seif()

        print(seif._get_mean_measurement_params(np.array([[8], [2], [8],
                                                          [2], [2],
                                                          [7], [7],
                                                          [2], [7]]),2))

    def test_build_projection_matrix(self):
        seif = self.make_seif()

        # returns first and third indices of identity matrix
        #todo: just send a list of coordinates to coo_matrix. This is pretty innefecient.
        #todo: perhaps use a dense matrix
        self.assertListEqual(seif._build_projection_matrix([1,3]).tolist(),
                             [[0,0],
                              [1,0],
                              [0,0],
                              [0,1]])

    @ut.skip("not sure how it partitions. Need to see more examples.")
    def test_partition_links(self):
        seif = self.make_seif()
        print(seif._partition_links())

    def test_sparsification(self):
        seif = self.make_seif()
        seif._sparsification()  # todo: almost as inefficient as possible, but not quite
        print(seif.H)
        print(seif.b)
