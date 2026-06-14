from environment import Environment
from agent import Agent

import numpy as np

# TODO: maybe make this a derived class of the environment, so we whole thing with storing positions + velocities in the environment while using them here as well is less cumbersome
class Model:
    def __init__(self, environment : Environment, slowdown = 0.05):
        self.environment = environment
        self.slowdown = slowdown

    def step(self):
        # step 1: compute gaps (info for the vehicle choice)
        gaps = self.environment.compute_gaps()

        # step 2: every vehicle makes a decision on lane-switching
        choices = np.zeros(self.environment.n_agents)
        for i, agent in enumerate(self.environment.agents):
            choices[i] = agent.choose_action(gaps[i])
        
        # step 3: apply lane changing, in random order, checking for collisions
        # TODO: in a random order, go through all choices and apply them, if possible. If it was not possible, keep track to penalize the reward used for learning (or some better way of punishing / accounting for collisions)

        # step 4: NaSch velocity update
        gaps_forward = self.environment.compute_gaps()[:, 1]
        self.environment.velocity = np.minimum(self.environment.velocity + 1, self.environment.v_max) # acceleration
        self.environment.velocity = np.minimum(self.environment.velocity, gaps_forward) # braking
        random_values = np.random.random(self.environment.n_agents)
        self.environment.velocity = np.where((self.environment.velocity > 0) & (random_values < self.slowdown), self.environment.velocity - 1, self.environment.velocity) # janky way of reducing velocity by 1 by a given percentage, if it wasn't zero already

        # step 5: update vehicle positions
        self.environment.positions = (self.environment.positions + self.environment.velocity) % self.environment.lane_length

        # step 6: compute reward and update agent strategy
        