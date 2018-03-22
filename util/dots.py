from numpy.linalg import multi_dot

def dots(*arg):
    return multi_dot(arg)