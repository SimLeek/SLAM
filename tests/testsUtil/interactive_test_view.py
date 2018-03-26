import unittest as ut
from .interactive_display import display
from util.view_xform import *

class TestViews(ut.TestCase):
    def test_laplacian(self):
        t = display(window_title='Laplacian Test',
                    callbacks=[laplacian_callback])
        t.join()

    def test_keypoints(self):
        kpr = KeyPointRecognizer(display=True, features_per_frame=2**16-1)
        t = display(window_title='Key Point Test',
                    callbacks=[kpr.key_point_callback])

        t.join()