import numpy as np
import pygame as pg
from Core.Noise import perlin
import random as rng
import numba as nb


class Grid(object):
    @nb.jit(forceobj=True)
    def __init__(self, size, display_size, parent, wall_percentage=20, seed=0):
        print("-> starting initializing grid variables")
        self.size = size
        self.display_size = display_size
        self.ratio = max(display_size) / min(display_size)
        self.background = (0, 0, 0, 255)
        self.surface = pg.Surface(display_size, flags=pg.SRCALPHA | pg.HWSURFACE)
        self.parent = parent
        self.center_rect = self.display_size
        self.squares = (size, self._rect_data[:2], self._rect_data[2:])
        self.wall_treshold = (wall_percentage * 0.02) - 1.0
        self.seed = seed
        rng.seed(self.seed)
        self.start = 0, 0
        self.surface.fill(self.background)
        walls = list()
        self.mask = self.surface.copy()
        self.mask.fill((255, 255, 255, 255))
        print("-> starting initializing grid walls")
        for i, x in enumerate(self.squares):
            for j, y in enumerate(x):
                n = perlin(i + 0.5, j + 0.5, self.seed)
                color = np.array([100, 0, 0, 255])
                if n < self.wall_treshold:
                    color[0] = 200
                    walls.append(y)
                else:
                    color[1:3] = 128
                    self.mask.fill((0, 0, 0, 0), y)
                self.surface.fill(color, y)
        
        self.mask = pg.mask.from_surface(self.mask)
        # self.surface.blit(self.mask.to_surface(self.surface, setcolor=(255,255,255,255), unsetcolor=(0,0,0,255)),(0,0))
        self.walls = walls
        print("-> setting start and goal positions")
        rng.seed(self.seed + 17)
        self.setStart()
        self.setGoal()
    
    @nb.jit(forceobj=True)
    def draw(self):
        self.parent.blit(self.surface, (0, 0))
    
    @nb.jit(forceobj=True)
    def randomPosInGrid(self):
        offset = np.array(self.center_rect.topleft)
        max = np.array(self.center_rect.bottomright)
        pos_found = False
        pos = np.zeros(2)
        while pos_found == False:
            pos[0] = rng.randint(offset[0], max[0] - 1) + 0.5
            pos[1] = rng.randint(offset[1], max[1] - 1) + 0.5
            pos_found = True
            for rect in self.walls:
                if bool(rect.collidepoint(pos)):
                    pos_found = False
                    break
        return pos
    
    @nb.jit(forceobj=True)
    def setStart(self):
        self.start = self.randomPosInGrid()
        self.start_square = None
        for x in self.squares:
            for square in x:
                if square.collidepoint(self.start):
                    self.start_square = square
                    break
            if self.start_square is not None:
                break
        
        pg.draw.ellipse(self.surface, (0, 0, 255, 255), self.start_square)
        surf = self.start_square.height * self.start_square.width
        width = min(int(surf / 200.0), 3) if min(int(surf / 200.0), 3) != 0 else -1
        pg.draw.ellipse(self.surface, (0, 0, 0, 198), self.start_square, width=width)
        self.start = self.start_square.center
    
    @nb.jit(forceobj=True)
    def setGoal(self):
        self.goal = np.array(self.start)
        self.goal_square = None
        while not self.goal_square:
            self.goal = self.randomPosInGrid()
            for x in self.squares:
                for square in x:
                    if square.collidepoint(self.goal):
                        if square == self.start_square:
                            print("True")
                            continue
                        else:
                            self.goal_square = square
                            break
                if self.goal_square is not None:
                    break
        
        pg.draw.ellipse(self.surface, (0, 255, 0, 255), self.goal_square)
        surf = self.goal_square.height * self.goal_square.width
        width = min(int(surf / 200.0), 3) if min(int(surf / 200.0), 3) != 0 else -1
        pg.draw.ellipse(self.surface, (0, 0, 0, 198), self.goal_square, width=width)
        self.goal = self.goal_square.center
    
    @property
    def center_rect(self):
        pass
    
    @center_rect.setter
    def center_rect(self, display):
        width, height = display
        rect = np.zeros((2, 2), dtype=int)
        if width > height:
            offset = round((width - height) / 2)
            rect[0] = (offset, height)
            rect[1][1] = height
        elif height > width:
            offset = round((height - width) / 2)
            rect[1] = (offset, width)
            rect[0][1] = width
        else:
            rect[0][1] = width
            rect[1][1] = height
        rect = rect.transpose()
        self._rect = pg.Rect(rect[0], rect[1])
        self._rect_data = (rect[0][0], rect[0][1], rect[1][0], rect[1][1])
    
    @center_rect.getter
    def center_rect(self):
        return self._rect
    
    @property
    def squares(self):
        pass
    
    @squares.setter
    def squares(self, params):
        gridsize, startpos, endpos = params
        if type(gridsize) == int:
            gridsize = (gridsize, gridsize)
        
        gridsize = np.array(gridsize)
        endpos = np.array(endpos)
        squares = list()
        
        for x in range(gridsize[0]):
            y_list = list()
            for y in range(gridsize[1]):
                y_list.append(pg.Rect(startpos + (endpos / gridsize * np.array((x, y))),
                                      endpos / gridsize))
            squares.append(tuple(y_list))
        
        self._squares = tuple(squares)
    
    @squares.getter
    def squares(self):
        return self._squares
    
    @property
    def walls(self):
        pass
    
    @walls.setter
    def walls(self, rects):
        self._walls = tuple(rects)
    
    @walls.getter
    def walls(self):
        return self._walls
    
    @walls.deleter
    def walls(self):
        self._walls = tuple()
# Endfile
