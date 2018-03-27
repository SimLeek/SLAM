import unittest as ut
import util.pose_cells as pose
import math as m

class TestPoseCells(ut.TestCase):
    def setUp(self):
        self.cortex = pose.PoseCortex()

    def test_pose_to_cortex(self):
        print(self.cortex.pose_to_cortex(123000,456000,1.2)) #x,y,theta

