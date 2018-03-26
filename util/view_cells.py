import math as m
import warnings

class ViewCortex(object):
    def __init__(self, cam_width, x_log_scale=48):
        self.x_max = cam_width
        self.x_min = 0
        self.x_log_scale = x_log_scale

