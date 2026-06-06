import random

MAX_SPEED = 5

class Car:
    def __init__(self, p=0.01):
        self.position = 0
        self.velocity = 0
        self.lane = 0
        self.p = p  # Random slowdown probability



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

            