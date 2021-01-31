import pygame
import sys
from multiprocessing import Process, Pipe
from Core.UI.Grid import Grid
from Core.UI.Population import Population
import time
import numpy as np
import tensorflow as tf
import numba as nb


def run(grid=(40, 40), size=None, wall_percentage=20, seed=0, pop=100, parent_perc=20, velmax=4.9,
        mutationRate=0.04, prize=10000, epsilon=1e-8, layerdims=(16, 8, 2)):
    t = tf.Variable(1)
    pygame.init()
    size = (1920, 1080) if not size else size
    f = 0x000020 | pygame.HWACCEL | pygame.FULLSCREEN
    # f = 0x000010
    display = 1
    # display = 0
    screen = pygame.display.set_mode(size, display=display, vsync=1, flags=f)
    
    text = "LOADING GRID..."
    font = pygame.font.SysFont("BRITANIC.TTF", 120)
    img = font.render(text, True, pygame.Color(255, 255, 255))
    offset = (screen.get_rect().centerx - img.get_rect().centerx, screen.get_rect().centery - img.get_rect().centery)
    screen.blit(img, offset)
    
    pygame.display.flip()
    print("SCREEN INITIALIZED")
    grid = Grid(grid, size, screen, wall_percentage=wall_percentage, seed=seed)
    walls = grid.mask
    grid.draw()
    pygame.display.flip()
    print("GRID INITIALIZED")
    kwargs = dict()
    kwargs["pop"] = pop
    kwargs["layerdims"] = layerdims
    kwargs["parent_perc"] = parent_perc
    kwargs['start'] = grid.start
    kwargs['velmax'] = velmax
    kwargs['mutationRate'] = mutationRate
    kwargs['goalpos'] = grid.goal
    kwargs['scale'] = 0.5
    kwargs['goalmargin'] = 2
    kwargs['epsilon'] = epsilon
    kwargs['prize'] = prize
    kwargs['screen'] = size
    kwargs['brain'] = None
    kwargs['seed'] = seed
    kwargs['surface'] = pygame.surfarray.array3d(grid.surface)
    kwargs['square_side'] = grid.center_rect.width / float(grid.size[0])
    print("POPULATION INITIALIZED")
    parent_conn, child_conn = Pipe()
    surf = grid.surface.copy()
    surf.fill((0, 0, 0, 255))
    walls = pygame.surfarray.array3d(walls.to_surface(surf, setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0)))
    P = Process(target=pop_process, args=(child_conn, kwargs, walls, grid.center_rect))
    while not pygame.display.get_init():
        print("WAITING DISPLAY INITIALIZATION")
    P.start()
    print("SUBPROCESS INITIALIZED")
    
    last_t = time.time()
    fps_buffer = np.zeros(100, dtype=np.float) + 60.0
    time_buffer = np.zeros(100, dtype=np.float) + 1.0 / 60.0 * 1000.0
    i = 0
    print("START")
    gen = 1
    array = 1
    step = 0
    fitness = 0
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                P.terminate()
                sys.exit()
                
        if parent_conn.poll():
            out = parent_conn.recv()
            array, gen, step, fitness = out
        
        if type(array) != int:
            pygame.surfarray.blit_array(screen, np.array(array))
        else:
            grid.draw()
        
        font = pygame.font.SysFont("BRITANIC.TTF", 40)
        i = print_data(font, screen, fps_buffer, time_buffer, gen, step, fitness, i, last_t)
        pygame.display.flip()
        last_t = time.time()


@nb.jit(forceobj=True)
def pop_process(conn, kwargs, walls, rect):
    pop = Population(kwargs, walls=walls, limitRect=rect)
    surface = pop.show()
    array = pygame.surfarray.array3d(surface)
    conn.send((array, pop.gen, int(pop.step), 0))
    while True:
        surface = pop.update()
        array = pygame.surfarray.array3d(surface)
        best = pop.bestCreature.fitness if pop.bestCreature else 0
        conn.send((array, pop.gen, int(pop.step), int(best)))


@nb.jit(forceobj=True)
def print_data(font, screen, fps_buffer, time_buffer, gen, step, fitness, i, last_t):
    i = i + 1 if i < time_buffer.size else 1
    fps_buffer[i - 1] = min(1.0 / max(time.time() - last_t, 1e-16), 60.0)
    time_buffer[i - 1] = (time.time() - last_t) * 1000.0
    
    FPS = "{:<3} FPS ({:<5} ms)".format(round(min(60.0, max(0.0, fps_buffer.mean())), 1), round(time_buffer.mean(),
                                                                                                2))
    img = font.render(FPS, True, pygame.Color(255, 255, 255))
    screen.blit(img, (20, 20))
    
    gen_str = "Generation: {: >4}".format(int(gen))
    img = font.render(gen_str, True, pygame.Color(255, 255, 255))
    screen.blit(img, (20, 70))
    
    step_str = "Step: {}".format(int(step))
    img = font.render(step_str, True, pygame.Color(255, 255, 255))
    screen.blit(img, (20, 120))
    
    fit_str = "Best Fitness: {}".format(round(fitness))
    img = font.render(fit_str, True, pygame.Color(255, 255, 255))
    screen.blit(img, (20, 170))
    return i
# Endfile
# Endfile
