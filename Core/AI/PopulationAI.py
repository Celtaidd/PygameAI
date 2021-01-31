import tensorflow as tf
import numpy as np
import time
import numba as nb
import os
import pickle
import traceback as tr
import random as rng
from multiprocessing import Pool


class PopulationAI(object):
    def __init__(self, kwargs, individual):
        kwargs["pop"] = 500 if "pop" not in kwargs.keys() else kwargs["pop"]
        kwargs["parent_perc"] = 20 if "parent_perc" not in kwargs.keys() else kwargs["parent_perc"]
        self.kwargs = kwargs.copy()
        self.parent_perc = tf.constant(kwargs["parent_perc"] / 100.0, dtype=tf.float32)
        self.individuals = tuple(individual(kwargs.copy(), i) for i in range(kwargs['pop']))
        self.individual_class = individual
        self.fitnessSum = tf.Variable(0, dtype=tf.float32)
        self.bestCreature = None
        self.gen = tf.Variable(1, dtype=tf.int32)
        self.ids = 0
        self.batch = list(self.individuals)
    
    @nb.jit(forceobj=True)
    def move(self):
        for creature in self.individuals:
            creature.move()
    
    @nb.jit(forceobj=True)
    def update(self):
        
        n = float(len(self.batch))
        for i, creature in enumerate(self.batch):
            creature.update()
            if creature.isdead:
                self.batch.pop(i)
            if i % 10 == 0:
                print("\r--> {:.2%}".format(round((i + 1) / n, 2)), end="")
        if self.allDead():
            print("--> All Deads")
            self.naturalSelection()
            self.batch = [i for i in self.individuals]

        print("")
    
    @nb.jit(forceobj=True)
    def allDead(self):
        check = True
        for creature in self.individuals:
            if not creature.isdead and not creature.reachedGoal:
                check = False
        return check
    
    @nb.jit(forceobj=True)
    def calculateFitnessSum(self):
        min = tf.Variable(0, dtype=tf.float32)
        for creature in self.individuals:
            if creature.fitness < min:
                min = creature.fitness
        for creature in self.individuals:
            creature.fitness.assign_sub(min-1.0)
            self.fitnessSum.assign_add(tf.cast(creature.fitness, dtype=tf.float32))
    
    @nb.jit(forceobj=True)
    def selectParent(self):
        if self.fitnessSum.read_value() == 0:
            self.fitnessSum.assign_add(1)
        
        up = max(int(self.fitnessSum.read_value()), 1)*1000
        
        r = tf.convert_to_tensor([np.random.randint(0, up)], dtype=tf.float32)
        runningSum = tf.Variable(0.0, dtype=tf.float32)
        
        for creature in self.individuals:
            runningSum.assign_add(tf.cast(creature.fitness*1000, dtype=tf.float32))
            if tf.greater(runningSum, r):
                return creature.clone(creature.id)
        self.ids += 1
        return self.individual_class(self.kwargs.copy(), self.ids)
    

    def setBest(self):
        max = tf.Variable(0, dtype=tf.float32)
        for creature in self.individuals:
            creature.isBest = False
            if tf.greater(creature.fitness, max):
                max.assign(creature.fitness)
                self.bestCreature = creature.clone(0)
        if self.bestCreature:
            self.bestCreature.isBest = True
        else:
            self.bestCreature = self.individual_class(self.kwargs.copy(), 0)
    
    # @nb.jit(forceobj=True)
    def naturalSelection(self):
        self.setBest()
        self.calculateFitnessSum()
        new_individuals = list()
        maxpop = len(self.individuals)
        pop_parents = int(maxpop * self.parent_perc)
        new_individuals.append(self.bestCreature)
        new_kwargs = self.kwargs.copy()
        
        for i in range(0, maxpop - 1):
            if i <= pop_parents:
                new_individuals.append(self.selectParent())
            else:
                new_kwargs["params"] = self.selectParent().brain.mutate(self.selectParent().brain)
                new_individuals.append(self.individual_class(new_kwargs.copy(), i))
        new_individuals.append(self.bestCreature)
        
        self.individuals = tuple(new_individuals)
        self.gen.assign_add(1)

        json_path = os.path.join(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Ressources"), "brainparameters")

        print("--> Saving Data")
        for j, f in ((i, "creature{}.pkl".format(i)) for i in range(len(self.individuals))):
            with open(os.path.join(json_path, f), 'wb') as of:
                pickle.dump(self.individuals[j].brain.params, of)
