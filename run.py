import traceback as tr

if __name__ == "__main__":
    import Core.UI as exe
    
    grid = 20
    screen = 1920, 1080
    
    wall_perc = 44
    pop = 200
    parent_perc = 10
    seed = 42
    layerdims = (16, 14, 10, 6, 4, 2)  # [16, ..., 2]
    exe.run((grid, grid), size=screen, wall_percentage=wall_perc, seed=seed, pop=pop, velmax=float(screen[1]) / (
            grid * 2.0), parent_perc=parent_perc, layerdims=layerdims, mutationRate=0.08)

# Endfile

# import numpy as np
#
#
# def p(x):
#     a = np.random.randint(1, 7, 100000000)
#     b = np.random.randint(1, 7, 100000000)
#
#     a += np.random.randint(1, 7, 100000000)
#     b += np.random.randint(1, 7, 100000000)
#     a[a < x] = b[a < x].copy()
#
#     avec = round(a[a >= x].size / a.size * 100, 3)
#     sans = round(b[b >= x].size / b.size * 100, 3)
#
#     print('Chances de réussite d\'une charge à {: >2d}": {: >6}% ({:>6}% avec reroll) '.format(x, sans, avec) \
#           + '=> Donc un reroll ~= à un bonus au jet de: ' + \
#           '{:>3}'.format(round((avec - sans) / (100.0 / 6.0), 2)))
#
#
# for i in range(2, 13):
#     p(i)
#
# import numpy as np
# from numba import jit
#
#
# @jit(nopython=True)
# def p(x, j):
#     s = 0
#     for i in range(int(j)):
#         a = np.random.randint(1, 366, x)
#         for i in range(a.size):
#             if np.random.random() > 0.75:
#                 a[i] += 1
#         b = 0
#         for k in range(1, 367):
#             b += 1 if a[a == k].size > 1 else 0
#         s += 1 if b > 0 else 0
#
#     return s, j
#
#
# for i in range(1, 101, 1):
#     s, j = p(i, 1e6)
#     percent = round(float(s) / float(j) * 100.0, 3)
#     print("Chances qu'au moins deux personnes soient nées le même jour dans un groupe de {:>3} personnes :".format(i)
#           + " {:>6}%".format(percent))
# print('done')
