import numpy as np

def p2logodds(p):
    return np.log(p / (1 - p))

def logodds2p(l):
    return 1 - 1 / (1 + np.exp(l))
        
def numfmt(v, x): # your custom formatter function: divide by 100.0
    s = '{}'.format(x * v)
    return s

def bresenham2(sx, sy, ex, ey):
    #from https://github.com/daizhirui/Bresenham2D/blob/main/bresenham2Dv1.py
    #notes that there exists a better scikit version
    """ Bresenham's ray tracing algorithm in 2D.
    :param sx: x of start point of ray
    :param sy: y of start point of ray
    :param ex: x of end point of ray
    :param ey: y of end point of ray
    :return: cells along the ray
    """
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx + 1, 1), dtype=int)
    else:
        q = np.append(0, np.greater_equal(
            np.diff(
                np.mod(np.arange(  # If d exceed dx, decrease d by dx
                    np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy,
                    dtype=int), dx
                )  # offset np.floor(dx / 2) to compare d with 0.5dx
            ), 0))
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)

    bres = np.vstack((x,y)).T.tolist()
    return bres