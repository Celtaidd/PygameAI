import numpy as np
import tensorflow as tf
from Core.AI import NeuralNetwork as NN
# from Core.AI import CNeuralNetwork as CNN
import math
import numba as nb
import traceback as tr
import pickle
import os
from multiprocessing import Pool, Value, Array

class CreatureAI(object):
    @nb.jit(forceobj=True)
    def __init__(self, kwargs, name):
        self.kwargs = kwargs
        self.kwargs['start'] = (0.0, 0.0) if "start" not in kwargs.keys() else kwargs["start"]
        self.kwargs['velmax'] = 4.9 if "velmax" not in kwargs.keys() else kwargs['velmax']
        self.kwargs['mutationRate'] = 0.01 if "mutationRate" not in kwargs.keys() else kwargs['mutationRate']
        self.kwargs['goalpos'] = (1.0, 1.0) if "goalpos" not in kwargs.keys() else kwargs['goalpos']
        self.kwargs['scale'] = 5 if "scale" not in kwargs.keys() else kwargs['scale']
        self.kwargs['goalmargin'] = 2 if "goalmargin" not in kwargs.keys() else kwargs['goalmargin']
        self.kwargs['epsilon'] = 1e-8 if "epsilon" not in kwargs.keys() else kwargs['epsilon']
        self.kwargs['prize'] = 10000 if "prize" not in kwargs.keys() else kwargs['prize']
        self.kwargs['brain'] = None if "brain" not in kwargs.keys() else kwargs['brain']
        self.kwargs['seed'] = 42 if "seed" not in kwargs.keys() else kwargs['seed']
        self.id = tf.constant(name, dtype=tf.int8)
        self.start = tf.Variable(np.array(kwargs['start']), dtype=tf.float32)
        self.pos = tf.Variable(np.array(kwargs['start']), dtype=tf.float32)
        self.vel = tf.Variable(np.zeros(2), dtype=tf.float32)
        self.acc = tf.Variable(np.zeros(2), dtype=tf.float32)
        self.steps = tf.Variable(0, dtype=tf.int16)
        
        self.brain = Brain(self.loadbrain(self.id, kwargs))
        
        self.mutationRate = tf.Variable(kwargs['mutationRate'], dtype=tf.float32)
        self.velmax = tf.constant(kwargs['velmax'], dtype=tf.float32)
        self.isdead =False
        self.reachedGoal = tf.Variable(False, dtype=tf.bool)
        self.isBest =False
        self.fitness = tf.Variable(0, dtype=tf.float32)
        self.goal = tf.constant(kwargs['goalpos'], dtype=tf.float32)
        self.goalmargin = tf.constant(kwargs['goalmargin'], dtype=tf.float32)
        self.epsilon = tf.constant(kwargs['epsilon'], dtype=tf.float32)
        self.prize = tf.constant(kwargs['prize'], dtype=tf.int16)
        self.scale = tf.constant(kwargs['scale'], dtype=tf.float32)
        
        self.directions_rays = tf.Variable(np.zeros(8), dtype=tf.float32)
        # 7 0 1
        # 6 X 2
        # 5 4 3
    
    @staticmethod
    def loadbrain(i, kwargs):
        json_path = os.path.join(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Ressources"), "brainparameters")
        
        f = os.path.join(json_path, "creature{}.pkl".format(i))
        chk = os.path.isfile(f)
        if chk:
            with open(f, 'rb') as of:
                kwargs['params'] = pickle.load(of)
        return kwargs
    
    @property
    def distanceToGoal(self):
        dxy = tf.subtract(self.goal, self.pos)
        dxy = tf.pow(dxy, 2)
        return tf.constant(tf.pow(tf.reduce_sum(dxy), 0.5))
    
    @nb.jit(forceobj=True)
    def move(self):
        self.steps.assign_add(1)
        if tf.greater(self.steps, 1000):
            self.isdead = True
            self.fitness.assign_sub(tf.cast(self.prize, tf.float32))
            return
        m = self.brain.nextmove(self.pos, self.vel, self.directions_rays, self.goal)
        self.acc.assign(m)
        self.vel.assign(self.acc + self.vel)
        speed = tf.reduce_sum(self.vel ** 2.0) ** 0.5
        if tf.greater(speed, self.velmax):
            self.vel.assign(self.vel / tf.maximum(speed, self.epsilon) * self.velmax)
        self.pos.assign(tf.cast(self.pos, tf.float32) + self.vel)
        self.fitness.assign_add(100.0 / tf.maximum(tf.cast(self.distanceToGoal*2, dtype=tf.float32), self.epsilon))

    @nb.jit(forceobj=True)
    def update(self):
        if not self.isdead and not self.reachedGoal:
            self.move()
            if tf.greater(self.goalmargin, self.distanceToGoal):
                self.reachedGoal.assign(True)
                self.fitness.assign_add(tf.cast(self.prize, tf.float32) / tf.cast(self.steps, tf.float32))
    
    @nb.jit(forceobj=True)
    def getBaby(self, partnerbrain):
        return self.brain.mutate(partnerbrain, self.mutationRate)


class Brain(object):
    @nb.jit(forceobj=True)
    def __init__(self, kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs['layerdims'] = (16, 6, 2) if "layerdims" not in kwargs.keys() else kwargs['layerdims']
        self.kwargs['epsilon'] = 1e-8 if "epsilon" not in kwargs.keys() else kwargs['epsilon']
        self.kwargs['seed'] = 42 if "seed" not in kwargs.keys() else kwargs['seed']
        self.seed = tf.constant(kwargs['seed'], dtype=tf.int8)
        self.layerdims = tf.constant(kwargs["layerdims"], dtype=tf.int8)
        self.lastmove = tf.Variable(np.zeros(2), dtype=tf.float32)
        self.params = NN.initialize_parameters(self.layerdims) if not "params" in kwargs.keys() else kwargs[
            "params"]
        self.input = np.zeros(16)
        self.epsilon = tf.constant(kwargs["epsilon"], dtype=tf.float32)
        self.mutationRate = tf.Variable(kwargs['mutationRate'], dtype=tf.float32)
    
    @nb.jit(forceobj=True)
    def forward_propagation(self, A_prev, W, b):
        Z = NN.linear_propagation(A_prev, W, b)
        A = NN.RELu(Z)
        return A
    
    @nb.jit(forceobj=True)
    def nextmove(self, pos, vel, directions_rays, goal):
        self.input = np.array(list(self.lastmove.value()) + list(pos.value()) + list(vel.value()) + list(goal)
                              + list(directions_rays.value()))
        A = tf.convert_to_tensor(self.input.copy(), dtype=tf.float32)
        for l in range(len(self.layerdims) - 1):
            A_prev = A
            Z = NN.linear_propagation(A_prev, tf.convert_to_tensor(self.params["W" + str(l + 1)], dtype=tf.float32),
                                       tf.convert_to_tensor(self.params["b" + str(l + 1)], dtype=tf.float32))
            if l < len(self.layerdims) - 1:
                A = NN.RELu(Z)
            else:
                A = Z
        sintheta = A[0]
        costheta = A[1]
        theta = tf.atan(sintheta / (costheta if costheta != 0.0 else self.epsilon))
        if sintheta < 0.0 and costheta < 0.0:
            self.lastmove.assign((tf.cos(theta), tf.sin(theta)))
        else:
            self.lastmove.assign((tf.cos(theta+math.pi), tf.sin(theta+math.pi)))
        return tf.constant(self.lastmove, dtype=tf.float32)
    
    # @nb.jit(forceobj=True)
    def mutate(self, partnerbrain):
        self.partnerparams = partnerbrain.params
        partnerparams = partnerbrain.params
        babyparams = self.params.copy()
        mutationRate = self.mutationRate
        
        
        f = _mutate
        f_mutationRate = [mutationRate] * (len(self.layerdims) - 1)
        f_babyparams = [babyparams] * (len(self.layerdims) - 1)
        f_partnerparams = [partnerparams] * (len(self.layerdims) - 1)
        with Pool(6) as p:
            babyparams = p.map(f, zip(f_mutationRate,f_babyparams, f_partnerparams, range(len(self.layerdims) - 1)))
            p.join()
        
        return babyparams


def _mutate(x):
    mutationRate = x[0]
    babyparams = x[1]
    partnerparams = x[2]
    l = x[3]
    for i, node in enumerate(partnerparams["W" + str(l + 1)]):
        for j, weight in enumerate(node):
            if tf.greater(mutationRate, np.random.random()):
                babyparams["W" + str(l + 1)][i][j] += np.random.random() * 2.0 - 1.0
            elif np.random.random() < 0.5:
                babyparams["W" + str(l + 1)][i][j] = weight
                
    for i, node in enumerate(partnerparams["b" + str(l + 1)]):
        for j, bias in enumerate(node):
            if tf.greater(mutationRate, np.random.random()):
                babyparams["b" + str(l + 1)][i][j] += np.random.random() * 2.0 - 1.0
            elif np.random.random() < 0.5:
                babyparams["b" + str(l + 1)][i][j] = bias
    return babyparams
    return f