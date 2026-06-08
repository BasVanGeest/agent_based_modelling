import random
import numpy as np

MAX_SPEED = 5

class Car:
    def __init__(self, p=0.01):
        self.position = 0
        self.velocity = 0
        self.lane = 0
        self.p = p  # Random slowdown probability
        self.learning_rate = 0.1 #speed of updating threshold value
        

        # evolution
        self.threshold = 5.0
        self.previous_lap_time = np.inf
        self.enter_t = 0 # t value in which the car entered the highway 



    def calculate_next_velocity(self, p, gap):
        """
        Calculate the next velocity based on the exact Nagel-Schreckenberg rules.
        Evaluated in this order: Accelerate -> Brake -> Randomize.
        """
        
        # 1. Acceleration: If not at max speed, speed up by 1
        self.velocity = min(self.velocity + 1, MAX_SPEED)
        
        # 2. Braking: If the car is moving faster than the empty space ahead, 
        # it must brake to avoid a collision. The new velocity becomes the gap size.
        self.velocity = min(self.velocity, gap)
        
        # 3. Random slow down
        if self.velocity > 0 and random.random() < p:
            self.velocity -= 1

    def lap(self, t):
        new_lap_time = t - self.enter_t
        self.enter_t = t
        if new_lap_time < self.previous_lap_time:
            self.evolve()
        else:
            self.devolve()
        self.previous_lap_time = new_lap_time

    def evolve(self):
        self.threshold = self.threshold + (self.threshold * self.learning_rate)

    def devolve(self):
        self.threshold = self.threshold - (self.threshold * self.learning_rate)

        
            