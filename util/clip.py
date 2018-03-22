import math

clip = False

def clipAngle(angle, force=False):
    if clip or force:
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return angle


def clipState(state):
    if clip:
        state[2, 0] = clipAngle(state[2, 0])
    return state