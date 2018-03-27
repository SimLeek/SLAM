import unittest as ut
from .interactive_display import display
from util.view_xform import *
from util.view_cells import *

import os
my_dir = os.path.dirname(os.path.abspath(__file__))

class TestViews(ut.TestCase):
    def test_laplacian(self):
        t = display(cam_num=my_dir+os.sep+'giphy.mp4',
            window_title='Laplacian Test',
                    callbacks=[sharpened_laplacian_callback])
        t.join()

    def test_keypoints(self):
        kpr = KeyPointSystem(display=True, features_per_frame=2 ** 10 - 1)
        # todo: make multiple threads, so we can have frames with many keypoints without slowing down
        t = display(cam_num=my_dir+os.sep+'giphy.mp4',
            window_title='Key Point Test',
                    callbacks=[kpr.key_point_callback])

        t.join()

    def test_matches(self):
        vc = ViewCortex(256)

        kpr = KeyPointSystem(display=True, features_per_frame=2 ** 10 - 1, match_class=vc)
        t = display(cam_num=my_dir+os.sep+'giphy.mp4',
                    window_title='Match Test',
                    callbacks=[kpr.key_point_callback])

        t.join()