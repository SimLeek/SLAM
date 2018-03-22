import unittest as ut

import util.clip as clip
import math

class TestClip( ut.TestCase):
    def testClipAngle(self):
        clip.clip = True
        self.assertEqual(clip.clipAngle(math.pi*2.5), math.pi*.5)
        self.assertEqual(clip.clipAngle(-math.pi * 1.5), math.pi * .5)
        clip.clip = False
        self.assertEqual(clip.clipAngle(math.pi * 2.5), math.pi * 2.5)
        self.assertEqual(clip.clipAngle(-math.pi * 1.5), -math.pi * 1.5)

        self.assertEqual(clip.clipAngle(math.pi * 2.5, True), math.pi * .5)
        self.assertEqual(clip.clipAngle(-math.pi * 1.5, True), math.pi * .5)

    def testClipState(self):
        clip.clip = True
        state = {(2,0): math.pi*2.5}
        clip.clipState(state)
        self.assertEqual(state[2,0], math.pi*.5)

        clip.clip =False
        state = {(2, 0): math.pi * 2.5}
        clip.clipState(state)
        self.assertEqual(state[2, 0], math.pi * 2.5)



