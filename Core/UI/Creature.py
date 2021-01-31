from Core.AI.CreatureAI import CreatureAI
import pygame as pg
import numpy as np
import tensorflow as tf
import traceback as tr
import numba as nb


# Endfile

class Creature(CreatureAI, pg.sprite.DirtySprite):
    @nb.jit(forceobj=True)
    def __init__(self, kwargs, name):
        self.kwargs = kwargs.copy()
        self.kwargs['start'] = (0, 0) if "start" not in kwargs.keys() else kwargs["start"]
        self.kwargs['velmax'] = 4.9 if "velmax" not in kwargs.keys() else kwargs['velmax']
        self.kwargs['mutationRate'] = 0.01 if "mutationRate" not in kwargs.keys() else kwargs['mutationRate']
        self.kwargs['goalpos'] = (1, 1) if "goalpos" not in kwargs.keys() else kwargs['goalpos']
        self.kwargs['scale'] = 0.5 if "scale" not in kwargs.keys() else kwargs['scale']
        self.kwargs['goalmargin'] = 2 if "goalmargin" not in kwargs.keys() else kwargs['goalmargin']
        self.kwargs['epsilon'] = 1e-8 if "epsilon" not in kwargs.keys() else kwargs['epsilon']
        self.kwargs['prize'] = 10000 if "prize" not in kwargs.keys() else kwargs['prize']
        self.kwargs['brain'] = None if "brain" not in kwargs.keys() else kwargs['brain']
        self.kwargs['square_side'] = None if "square_side" not in kwargs.keys() else kwargs['square_side']
        pg.sprite.DirtySprite.__init__(self)
        CreatureAI.__init__(self, kwargs, name)
        self.scale = max(1.0, self.kwargs['square_side'] * float(kwargs['scale']))
        self.image = None
        self.mask = None
        self.rect = None
        self.parent = self.kwargs["parent"]
        self.walls = None
        self.limitRect = None
    
    @nb.jit(forceobj=True)
    def setimage(self, image, mask):
        self.image = image
        self.mask = mask
        if not self.rect:
            self.rect = self.image.get_rect()
            self.rect.move_ip(self.kwargs['start'][0] - (self.scale / 2.0),
                              self.kwargs['start'][1] - (self.scale / 2.0))
    
    @nb.jit(forceobj=True)
    def show(self, surf=None):
        if not surf:
            return self.image, self.rect
        else:
            surf.blit(self.image, self.rect)
    
    @nb.jit(forceobj=True)
    def move(self):
        CreatureAI.move(self)
        self.rect.move_ip(float(self.vel[0]), float(self.vel[1]))
        
        if self.rect.top < self.limitRect.top:
            self.isdead = True
            self.fitness.assign_sub(tf.cast(self.prize, tf.float32))
        elif self.rect.left < self.limitRect.left:
            self.isdead = True
            self.fitness.assign_sub(tf.cast(self.prize, tf.float32))
        elif self.rect.bottom > self.limitRect.bottom:
            self.isdead = True
            self.fitness.assign_sub(tf.cast(self.prize, tf.float32))
        elif self.rect.right > self.limitRect.right:
            self.isdead = True
            self.fitness.assign_sub(tf.cast(self.prize, tf.float32))
    
    @nb.jit(forceobj=True)
    def clone(self, name):
        return Creature(self.kwargs.copy(), name)
    
    @nb.jit(forceobj=True)
    def update(self):
        if not self.isdead or not self.reachedGoal:
            rect = self.limitRect
            walls = self.walls
            move_vect = tf.constant(((0, -1,), (1, -1,), (1, 0,), (1, 1,), (0, 1,), (-1, 1,), (-1, 0), (-1, -1,)),
                                    dtype=tf.float32)
            rect_limits = tf.constant([[rect.left, rect.right], [rect.top, rect.bottom]], dtype=tf.float32)
            
            temp_pos = tf.Variable(tf.cast(self.pos, tf.float32), dtype=tf.float32, name="temp_pos")
            dpos = tf.Variable(tf.cast(1, tf.float32), dtype=tf.float32, name="dpos")
            apos = tf.Variable(tf.cast(1, tf.float32), dtype=tf.float32, name="apos")
            x = tf.Variable(tf.cast(temp_pos[0], tf.float32), dtype=tf.float32, name="x")
            y = tf.Variable(tf.cast(temp_pos[1], tf.float32), dtype=tf.float32, name="y")
            out_of_bound = tf.Variable(False, dtype=tf.bool, name="out_of_bound")
            
            for i, dirvec in enumerate(move_vect):
                temp_pos.assign(tf.cast(self.pos, tf.float32))
                dpos.assign(tf.cast(1, tf.float32))
                dirvec = tf.Variable(tf.cast(dirvec, dtype=tf.float32))
                x.assign(tf.cast(temp_pos[0], tf.float32))
                y.assign(tf.cast(temp_pos[1], tf.float32))
                out_of_bound.assign(False)
                
                while not out_of_bound:
                    x.assign(tf.cast(temp_pos[0] + (dirvec[0] * dpos), tf.float32))
                    y.assign(tf.cast(temp_pos[1] + (dirvec[1] * dpos), tf.float32))
                    
                    for axis in rect_limits:
                        if tf.greater(axis[0], x):
                            out_of_bound.assign(True)
                        if tf.greater(x, axis[1]):
                            out_of_bound.assign(True)
                        if tf.greater(y, axis[0]):
                            out_of_bound.assign(True)
                        if tf.greater(y, axis[0]):
                            out_of_bound.assign(True)
                    
                    if not out_of_bound:
                        if walls.get_at((int(x), int(y))) == 1:
                            out_of_bound = True
                    
                    if not out_of_bound:
                        temp_pos.assign(tf.add(tf.cast(temp_pos, tf.float32), tf.cast(dirvec, tf.float32) * tf.cast(
                                dpos, tf.float32)))
                        dpos.assign(tf.cast(tf.add(tf.cast(dpos, tf.float32), tf.cast(apos, tf.float32)), tf.float32))
                
                wallpos = tf.Variable(tf.cast(temp_pos, tf.float32) - tf.cast(self.pos, tf.float32), dtype=tf.float32)
                self.directions_rays[i].assign((tf.reduce_sum(wallpos ** 2) ** 0.5) - (((tf.constant(
                        self.rect.height, dtype=tf.float32) / 2.0) ** 2.0) * 2.0) ** 0.5)
            super(Creature, self).update()
            
            if walls.overlap(self.mask, self.rect.topleft):
                # print((self.rect.topleft, walls.overlap_area(self.mask, self.rect.topleft)))
                self.fitness.assign_sub(tf.cast(self.prize, tf.float32))
                self.isdead = True
        self.parent.blit(self.image, self.rect)
        # self.parent.blit(walls.to_surface(setcolor=(0,0,0,255), unsetcolor=(255,255,255,255)), (0,0))
        # self.parent.blit(self.mask.to_surface(setcolor=(0,255,0,255)), self.rect)
