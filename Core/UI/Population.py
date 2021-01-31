import tensorflow as tf
import numba as nb
import pygame as pg
from Core.AI.PopulationAI import PopulationAI
from Core.UI.Creature import Creature as Crea
from multiprocessing import Pool
import time

class Population(PopulationAI):
    @nb.jit(forceobj=True)
    def __init__(self, kwargs, individual=Crea, walls=None, limitRect=None):
        self.surface = pg.Surface(kwargs["screen"], flags=pg.SRCALPHA | pg.HWSURFACE)
        wall_surf = pg.Surface(kwargs["screen"], flags=pg.SRCALPHA | pg.HWSURFACE)
        self.s = pg.surfarray.make_surface(kwargs["surface"])
        self.surface.blit(self.s, (0, 0))
        kwargs["parent"] = self.surface
        pg.surfarray.blit_array(wall_surf, walls)
        self.dead_surface = self.s.copy()
        self.walls = pg.mask.from_threshold(wall_surf, (255,255,255,255), (128,128,128,255))
        self.limitRect = limitRect
        super(Population, self).__init__(kwargs, individual=individual)
        self.setimages()
        for individual in reversed(self.individuals):
            individual.walls = self.walls
            individual.limitRect = self.limitRect
            individual.show(self.surface)

        # wall_surf = self.walls.to_surface(setcolor=(255,255,255,255), unsetcolor=(0,0,0,255))
        # self.surface.blit(wall_surf, (0,0))
        self.step = 0
    
    def show(self):
        print("--> Show points")
        return self.surface
    
    @nb.jit(forceobj=True)
    def __update_individual(self, creature):
        creature.update()

    @nb.jit(forceobj=True)
    def allDead(self):
        check = True
        for creature in self.individuals:
            if not creature.isdead and not creature.reachedGoal:
                check = False
            elif creature.isdead:
                creature.show(self.dead_surface)
        return check
    
    @nb.jit(forceobj=True)
    def update(self):
        self.step += 1
        print("-> Draw call: Step {}".format(self.step))
        self.setimages()
        self.surface.blit(self.s, (0, 0))
        self.surface.blit(self.dead_surface, (0, 0))
        PopulationAI.update(self)
        return self.surface
    
    @nb.jit(forceobj=True)
    def setimages(self):
        size = (self.individuals[0].scale, self.individuals[0].scale)
        self._winner = pg.Surface(size, flags=pg.SRCALPHA | pg.HWSURFACE)
        pg.draw.ellipse(self._winner, (0, 255, 0, 255), self._winner.get_rect())
        self._normal = pg.Surface(size, flags=pg.SRCALPHA | pg.HWSURFACE)
        pg.draw.ellipse(self._normal, (0, 0, 0, 30), self._normal.get_rect())
        self._dead = pg.Surface(size, flags=pg.SRCALPHA | pg.HWSURFACE)
        pg.draw.ellipse(self._dead, (0, 0, 0, 0), self._dead.get_rect())
        self._mask = pg.Surface(size, flags=pg.SRCALPHA | pg.HWSURFACE)
        pg.draw.ellipse(self._mask, (255, 255, 255, 255), self._mask.get_rect())
        
        surf = size[0] ** 2
        width = min(int(surf / 200.0), 3) if min(int(surf / 200.0), 3) != 0 else -1
        pg.draw.ellipse(self._winner, (10, 10, 10, 200), self._winner.get_rect(), width=width)
        pg.draw.ellipse(self._normal, (10, 10, 10, 200), self._normal.get_rect(), width=width)
        pg.draw.ellipse(self._dead, (10, 10, 10, 100), self._dead.get_rect(), width=width)
        pg.draw.ellipse(self._mask, (255, 255, 255, 255), self._mask.get_rect(), width=width)
        self._mask = pg.mask.from_surface(self._mask)
        
        for ind in self.individuals:
            if ind.isBest:
                ind.setimage(self._winner, self._mask)
            elif ind.isdead:
                ind.setimage(self._dead, self._mask)
            else:
                ind.setimage(self._normal, self._mask)
                
                
    # @nb.jit(forceobj=True)
    def naturalSelection(self):
        print("--> Simulating Natural Selection")
        tl = time.time()
        PopulationAI.naturalSelection(self)
        self.setimages()
        self.surface.blit(self.s, (0, 0))
        self.dead_surface = self.s.copy()
        for individual in self.individuals:
            individual.walls = self.walls
            individual.limitRect = self.limitRect
            img, rect = individual.show()
            self.surface.blit(img, rect)


        self.step = 0
        t = time.time() - tl
        ms = int((t - int(t)) * 100)
        h = int(t // 3600)
        m = int((t // 60) - (h * 60))
        s = int(t) - (h * 3600) - (m * 60)
        print("Done in: {:0>2d}:{:0>2d}:{:0>2d}.{:0<2d}".format(h, m, s, ms))

# Endfile
