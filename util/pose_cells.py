import math as m
import warnings

if False:
    from types import List

class PoseCortex(object):
    def __init__(self):
        self.x_max = 2**32 - 1
        self.y_max = 2**32 - 1
        self.x_min = -2**32 + 1
        self.y_min = -2**32 + 1
        self.int_theta_max = 2**8 - 1
        self.int_theta_min = -2**8 + 1
        self.sqrt2_scale_max = 62
        self.theta_scale_max = 7
        self.x_buckets = 4
        self.y_buckets = 4
        self.theta_buckets = 4

    def _radians_to_int(self, theta):
        theta_range = self.int_theta_max - self.int_theta_min
        theta_mod = theta % (m.pi*2)
        int_theta = (theta_mod*theta_range)/(m.pi*2) + self.int_theta_min
        return int_theta

    def _scaled_x_index(self, x, scale):
        x_range = self.x_max - self.x_min
        x_bucket_range = x_range / self.x_buckets
        x_size = x_range / int(m.sqrt(2) ** scale)
        x_bucket_size = x_bucket_range / int(m.sqrt(2) ** scale)
        x_min_scaled = self.x_min / int(m.sqrt(2) ** scale)
        x_mod = x % x_size
        try:
            x_index = ((x_mod + x_min_scaled) / x_bucket_size) % self.x_buckets
            return int(x_index)
        except ZeroDivisionError:
            warnings.warn(RuntimeWarning("Too many x scales. Some scales will never be used."))
            return int(0)

    def _scaled_y_index(self, y, scale):
        y_range = self.y_max - self.y_min
        y_bucket_range = y_range / self.y_buckets
        y_size = y_range / int(m.sqrt(2) ** scale)
        y_bucket_size = y_bucket_range / int(m.sqrt(2) ** scale)
        y_min_scaled = self.y_min / int(m.sqrt(2) ** scale)
        y_mod = y % y_size
        try:
            y_index = ((y_mod + y_min_scaled) / y_bucket_size) % self.y_buckets
            return int(y_index)
        except ZeroDivisionError:
            warnings.warn(RuntimeWarning("Too many y scales. Some scales will never be used."))
            return int(0)

    def _scaled_theta_index(self, int_theta, scale):
        theta_range = self.int_theta_max - self.int_theta_min
        theta_bucket_range = theta_range / self.theta_buckets
        theta_size = theta_range / (2 ** scale)
        theta_bucket_size = theta_bucket_range / (2 ** scale)
        theta_min_scaled = self.int_theta_min / (2 ** scale)
        theta_mod = int_theta % theta_size
        try:
            theta_index = ((theta_mod + theta_min_scaled) / theta_bucket_size) % self.theta_buckets
            return int(theta_index)
        except ZeroDivisionError:
            warnings.warn(RuntimeWarning("Too many theta scales. Some scales will never be used."))
            return int(0)



    def _x_y_to_cell_index(self,
                            x,  # type: float
                            y,  # type: float
                            scale  # type: int
                            ):
        """
            Converts x,y,theta 2D pose to cell activation index in the mammalian entorhinal coretex.
            Similar to binary representation, but keeps correlation between x and y.
            """
        x_y_index = self._scaled_x_index(x, scale) + self._scaled_y_index(y, scale) * self.x_buckets
        return x_y_index

    def _theta_to_cortex_indexes(self,
                             theta,  # type: float
                             scales  # type: List[int]
                             ):
        layers = []
        for scale in scales:
            layers.append(self._scaled_theta_index(theta, scale))
        return  layers

    def _x_y_to_cortex_indexes(self,
                               x,  # type: float
                               y,  # type: float
                               scales  # type: List[int]
                                ):
        layers = []
        for scale in scales:
            layers.append(self._x_y_to_cell_index(x, y, scale))
        return layers

    def pose_to_cortex_indexes(self,
                               x,  # type: float
                               y,  # type: float
                               theta,  # type: float
                               scales_x_y,  # type: List[int]
                               scales_theta  # type: List[int]
                               ):
        x_y_layers = self._x_y_to_cortex_indexes(x,y,scales_x_y)
        theta_layers = self._theta_to_cortex_indexes(self._radians_to_int(theta), scales_theta)
        pose_layers = []
        for x_y_layer in x_y_layers:
            for theta_layer in theta_layers:
                pose_layers.append(x_y_layer+(theta_layer*self.x_buckets*self.y_buckets))
        return pose_layers

    def pose_to_cortex(self,
                       x,y,theta
                       ):
        scales_x_y = [xy for xy in range(self.sqrt2_scale_max)]
        scales_theta = [t for t in range(self.theta_scale_max)]
        return self.pose_to_cortex_indexes(x,y,theta, scales_x_y, scales_theta)





def cell_layer_to_pose(
        i,  # type: int
        x_max,  # type: float
        y_max,  # type: float
        scale  # type: int
):
    scale_multiplier = m.sqrt(2) ** scale

    theta = m.floor(i / (x_max * y_max)) * scale_multiplier
    y = (m.floor(i / x_max) % y_max) * scale_multiplier
    x = (i % x_max) * scale_multiplier

    return x, y, theta


def cell_layers_to_pose(
        i,  # type: int
        x_max,  # type: float
        y_max,  # type: float
        scales  # type: List[int]
):
    x = 0
    y = 0
    theta = 0
    for scale in scales:
        temp_pose = cell_layer_to_pose(i, x_max, y_max, scale)
        x += temp_pose[0]
        y += temp_pose[1]
        theta += temp_pose[2]
    return x, y, theta
