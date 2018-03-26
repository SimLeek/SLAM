import cv_pubsubs as cvp
import cv2
import math as m

def laplacian_callback(frame):
    greyscale = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(greyscale, cv2.CV_8U)
    return [laplacian]

class KeyPointRecognizer(object):
    def __init__(self,
                 features_per_frame=2**12,
                 scale_levels=20,
                 display = False):
        self.display = display
        self.orb = cv2.ORB_create(nfeatures=features_per_frame,
                             scaleFactor=m.sqrt(2),
                             nlevels=scale_levels
                             )

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.max_desc = 0

    def key_point_callback(self, frame, sub_callback=None):
        edge = laplacian_callback(frame)[0]

        kp = self.orb.detect(edge, None)
        kp, des = self.orb.compute(edge, kp)

        if sub_callback is not None:
            sub_callback(kp, des)

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
            return [edge]




