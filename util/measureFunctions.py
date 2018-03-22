import math
from util.clip import clipAngle
import numpy as np

dimension = None

def measureFunction(rState, landmark):
    rDim = 3
    diff = landmark - rState[:rDim-1]
    diffNorm = np.linalg.norm(diff) # euclidean distance

    angle = rState[rDim-1, 0]
    diffAngle = math.atan2(diff[1], diff[0]) - angle
    diffAngle = clipAngle(diffAngle)

    return diffNorm, diffAngle


def gradMeasureFunction(rState, landmark, ldmIndex):
    """ Position 0 represents the robot and the distance to the landmark.
    The position at the landmark's index represents the distance to the robot.
    The 1 indexes store a perpendicular vector with length inverse to the original."""

    rDim = 3
    eDim = 2
    diff = (rState[:rDim-1] - landmark).flatten()
    diffNorm = np.linalg.norm(diff)

    grad = np.zeros(dimension * 2).reshape(dimension, 2)
    grad[:rDim-1, 0] = diff / diffNorm # creates a unit vector pointing from the robot to the landmark
    grad[ldmIndex:ldmIndex + eDim, 0] = -grad[:rDim-1, 0] # unit vector pointing from landmark to robot
    grad[:rDim-1, 1] = np.array([-diff[1], diff[0]]) / (diffNorm**2) # perpendicular vector / distance from robot to landmark
    grad[ldmIndex:ldmIndex + eDim, 1] = -grad[:rDim-1, 1] # perpendicular vector / distance from landmark to robot
    grad[rDim-1, 1] = -1

    return grad