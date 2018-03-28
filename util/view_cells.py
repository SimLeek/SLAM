import math as m
import warnings
import cv2
import numpy as np

if False:
    from typing import Optional

class ViewMatcher(object):
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
    def rank_match(m):
        try:
            return (m[0].distance/m[1].distance)**10
        except Exception:
            return -999

    @staticmethod
    def rank_match_by_original_index(m):
        try:
            return m[0].queryIdx
        except Exception:
            return -999

    def setup(self,
              kp,
              des # type: np.ndarray
              ):
        self.descriptors = des[:self.descriptor_set_size]


    def callback(self, kp, des):
        if self.setup_needed:
            self.setup(kp, des)
            self.setup_needed = False

        try:
            matches = self.flann.knnMatch(self.descriptors, des, k=2)
        except Exception:
            print("lol I'm blind")
            return None
        matches2 = []

        matches = sorted(matches, key = self.rank_match)
        for match in matches:
            try:
                if len(matches2)<self.descriptor_set_size:
                    matches2.append(match)
            except IndexError:
                pass

        self.matches = matches2

        return self.matches

class ViewCells(object):
    def __init__(self, screen_width, view_matcher):
        self.match_min = 0
        self.match_max = 32
        self.match_buckets = 2
        self.match_scale = 2
        self.match_multiplier = 32
        self.match_scales = 5 #7

        self.x_min = 0
        self.x_max = screen_width
        self.x_buckets = 2
        self.x_scale = 2
        self.x_scales = 3 #int(m.ceil(m.log(screen_width,2))-1)

        self.view_matcher = view_matcher

    def _scaled_x_index(self, x, scale):
        x_range = self.x_max - self.x_min
        x_bucket_range = x_range / self.x_buckets
        x_size = x_range / int(self.x_scale ** scale)
        x_bucket_size = x_bucket_range / int(self.x_scale ** scale)
        x_min_scaled = self.x_min / int(self.x_scale ** scale)
        x_mod = x % x_size
        try:
            x_index = ((x_mod + x_min_scaled) / x_bucket_size) % self.x_buckets
            return int(x_index)
        except ZeroDivisionError:
            warnings.warn(RuntimeWarning("Too many x scales. Some scales will never be used."))
            return int(0)

    def _scaled_match_index(self, match, scale):
        match = int(match*self.match_multiplier)
        match_range = self.match_max - self.match_min
        match_bucket_range = match_range / self.match_buckets
        match_size = match_range / int(self.match_scale ** scale)
        match_bucket_size = match_bucket_range / int(self.match_scale ** scale)
        match_min_scaled = self.match_min / int(self.match_scale ** scale)
        match_mod = match % match_size
        try:
            match_index = ((match_mod + match_min_scaled) / match_bucket_size) % self.match_buckets
            return int(match_index)
        except ZeroDivisionError:
            warnings.warn(RuntimeWarning("Too many match scales. Some scales will never be used."))
            return int(0)

    def _get_indexes(self, kp_single, matches_single):
        x_index_list = []
        for scale in range(self.x_scales):
            next_list = [0,0]
            next_list[self._scaled_x_index(kp_single.pt[0], scale)] = 1
            x_index_list.extend(next_list)

        match_index_list = []
        for scale in range(self.match_scales):
            next_list = [0,0]
            next_list[self._scaled_match_index(self.view_matcher.rank_match(matches_single), scale)] = 1
            match_index_list.extend(next_list)

        return x_index_list, match_index_list

    def callback(self, kp, des):
        matches = self.view_matcher.callback(kp, des)
        if matches is None:
            return None
        matches = sorted(matches, key=self.view_matcher.rank_match_by_original_index)
        match_arr = np.zeros((self.view_matcher.descriptor_set_size,2*(self.match_scales+self.x_scales)))

        for match in matches:
            try:
                x_index, match_index = self._get_indexes(kp[match[1].trainIdx],match ) # gives index error of no match[1], in which case we don't care
            except IndexError:
                try:
                    x_index, match_index = self._get_indexes(kp[match[0].trainIdx], match)
                except IndexError:
                    continue
            x_index.extend(match_index)
            match_arr[self.view_matcher.rank_match_by_original_index(match),:]=\
                x_index

        self.match_arr = match_arr
        return match_arr

import nupic.algorithms.spatial_pooler as sp

class ViewCorrelationCells(object):
    def __init__(self, view_cells):
        self.pool = sp.SpatialPooler(inputDimensions=(view_cells.view_matcher.descriptor_set_size*2*(view_cells.match_scales+view_cells.x_scales)),
                         columnDimensions=(view_cells.view_matcher.descriptor_set_size*2*(view_cells.match_scales+view_cells.x_scales)),
                         potentialPct=1.0,
                         globalInhibition=True,
                         stimulusThreshold=0.9,
                         synPermInactiveDec=.002,
                         synPermActiveInc=.1,
                         synPermConnected=0.2,
                         minPctOverlapDutyCycle=.001,
                         dutyCyclePeriod=1000,
                         boostStrength=0.45,
                         spVerbosity=0,
                         wrapAround=False)
        self.view_cells = view_cells
        return

    def callback(self, kp, des):
        view_cell_activations = self.view_cells.callback(kp, des)  #type: np.ndarray
        if view_cell_activations is None:
            return None
        view_cell_activations = view_cell_activations.flatten()

        correlation_cell_array = np.zeros_like(view_cell_activations)
        self.pool.compute(view_cell_activations, learn=True, activeArray=correlation_cell_array)

        view_cell_activations = view_cell_activations.reshape((self.view_cells.view_matcher.descriptor_set_size,2*(self.view_cells.match_scales+self.view_cells.x_scales)))
        correlation_cell_array = correlation_cell_array.reshape((self.view_cells.view_matcher.descriptor_set_size,2*(self.view_cells.match_scales+self.view_cells.x_scales)))
        cv2.imshow("bob", view_cell_activations)
        cv2.imshow("bob2", correlation_cell_array)

        return correlation_cell_array

