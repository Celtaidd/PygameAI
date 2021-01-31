import Core.Ctoolbox as Ct
from numba.pycc import CC

cc = CC("CNoise")

# @cc.export('perlin', 'f8(f8,f8)')
def perlin(x, y, r=0):
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    
    sx = x - float(x0)
    sy = y - float(y0)
    
    n0 = Ct.CdotGridGradient(x0, y0, x, y,r)
    n1 = Ct.CdotGridGradient(x1, y0, x, y,r)
    ix0 = Ct.Clerp(n0, n1, sx)
    n0 = Ct.CdotGridGradient(x0, y1, x, y,r)
    n1 = Ct.CdotGridGradient(x1, y1, x, y,r)
    ix1 = Ct.Clerp(n0, n1, sx)
    value = Ct.Clerp(ix0, ix1, sy)
    
    return value


if __name__ == "__main__":
    cc.compile()
# Endfile
