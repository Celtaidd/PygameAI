import numba as nb
from numba.pycc import CC
import time
import math
cc= CC("Ctoolbox")

@cc.export("Clerp", "f8(f8,f8,f8)")
def lerp(a, b, w):
    return (1.0 - w) * a + w * b

@cc.export("CdotGridGradient", "f8(i4,i4,f8,f8,i4)")
def dotGridGradient(ix,  iy,  x,  y, r=0):
    random = (2920.0) * math.sin(ix * 21942.0 + iy * 171324.0 + 8912.0 +r) * math.cos(ix * 23157.0 * iy * 217832.0 +
                                                                                   r)

    # // Precomputed (or otherwise) gradient vectors at each grid node
    gradient = math.cos(random), math.sin(random)

    # // Compute the distance vector
    dx = x - float(ix)
    dy = y - float(iy)

    # // Compute the dot-product
    return dx*gradient[0] + dy*gradient[1]



if __name__ == "__main__":
    cc.compile()