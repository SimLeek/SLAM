import math as m
import warnings
import cv2
import numpy as np

if False:
    from typing import Optional

class ViewCortex(object):
    def __init__(self,
                 descriptor_set_size,
                 comparison_image=None,
                 keypoint_descriptors=None, # type: Optional[np.ndarray]
                 kps=None
                 #cam_width, x_log_scale=48
                 ):
        if keypoint_descriptors is None:
            self.descriptors = np.array(descriptor_set_size)
            self.setup_needed = True
        else:
            self.descriptors = keypoint_descriptors
            self.setup_needed = False

        if kps is None:
            self.kps = []
        else:
            self.kps =kps

        self.descriptor_set_size = descriptor_set_size

        self.comparison_image = comparison_image

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def compare_matches(m):
        try:
            return m[1].distance - m[0].distance
        except Exception:
            return -999

    def setup(self,
              kp,
              des # type: np.ndarray
              ):
        self.descriptors = des


    def callback(self, kp, des):
        if self.setup_needed:
            self.setup(kp, des)
            self.setup_needed = False


        matches = self.flann.knnMatch(des, self.descriptors, k=2)
        matches2 = []

        matches = sorted(matches, key = self.compare_matches)
        for match in matches:
            try:
                if len(matches2)<self.descriptor_set_size:
                    matches2.append(match)
            except IndexError:
                pass

        self.matches = matches2
