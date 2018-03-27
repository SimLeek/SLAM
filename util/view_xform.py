import cv_pubsubs as cvp
import cv2
import math as m
import numpy as np

def sharpened_laplacian_callback(frame):
    greyscale = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    # from: https: // stackoverflow.com / a / 4993701 / 782170
    gaus = cv2.GaussianBlur(greyscale, (0,0), 3)
    weight = cv2.addWeighted(greyscale, 10.0, gaus, -9.0, 0)
    laplacian = cv2.Laplacian(weight, cv2.CV_8U)
    return [laplacian]

class KeyPointSystem(object):
    def __init__(self,
                 features_per_frame=2**12,
                 scale_levels=15,
                 display = False,
                 match_class=None):
        self.display = display
        self.orb = cv2.ORB_create(nfeatures=features_per_frame,
                             scaleFactor=m.sqrt(2),
                             nlevels=scale_levels
                             )

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.match_class = match_class
        self.max_desc = 0

    def key_point_callback(self, frame):
        edge = sharpened_laplacian_callback(frame)[0]

        kp = self.orb.detect(edge, None)
        kp, des = self.orb.compute(edge, kp)

        if self.match_class is not None:
            self.match_class.callback(kp, des)

        if des.size > self.max_desc:
            self.max_desc = des.size
            print(des.size)
        #print(des.shape)

        if self.display:
            for k in kp:
                pt1 = (int(k.pt[0]), int(k.pt[1]))
                pt2 = (int(k.pt[0] + m.cos(m.radians(k.angle)) * k.size),
                       int(k.pt[1] + m.sin(m.radians(k.angle)) * k.size))

                cv2.arrowedLine(edge, pt1, pt2, (int((1-k.response)*255),
                                                 int((k.response)*255),
                                                 0))
            if self.match_class is not None:
                for match in range(len(self.match_class.matches)):
                    if len(self.match_class.matches[match]) != 2:
                        continue
                print(len(self.match_class.matches))
            return [edge]




