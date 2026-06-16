from agent import Agents

import numpy as np

# TODO: maybe make this a derived class of the environment, so we whole thing with storing positions + velocities in the environment while using them here as well is less cumbersome
class Model:
    def __init__(self, agents : Agents, slowdown = 0.05):
        self.agents = agents
        self.slowdown = slowdown

    def step(self):
        # step 1: compute gaps and let each vehicle make a decision on if / where to switch
        choices, gaps = self.agents.choose_actions()

        # step 2: apply lane changing, in random order, checking for collisions
        applied_choices = np.zeros(self.agents.n_agents, dtype=int)
        agent_order = list(range(self.agents.n_agents))
        np.random.shuffle(agent_order)

        for agent_index in agent_order:
            choice = choices[agent_index]
            lane = self.agents.lanes[agent_index]
            position = self.agents.positions[agent_index]

            occupied = any(self.agents.lanes[i] == lane - 1 + choice and self.agents.positions[i] == position for i in range(self.agents.n_agents))

            if not occupied:
                self.agents.lanes[agent_index] = lane - 1 + choice
                applied_choices[agent_index] = choice
            else:
                applied_choices[agent_index] = 1

        # step 3: NaSch velocity update
        self.agents.update_velocities(self.slowdown)

        # step 4: update vehicle positions
        self.agents.positions = (self.agents.positions + self.agents.velocities) % self.agents.lane_length

        # step 5: compute reward and update agent strategy
        self.agents.update_weights(applied_choices, gaps)