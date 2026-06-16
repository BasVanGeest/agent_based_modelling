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
        applied_choices = self.apply_choices(choices)

        # step 3: NaSch velocity update
        self.agents.update_velocities(self.slowdown)

        # step 4: update vehicle positions
        self.agents.positions = (self.agents.positions + self.agents.velocities) % self.agents.lane_length

        # step 5: compute reward and update agent strategy
        self.agents.update_weights(applied_choices, gaps)


    def apply_choices(self, choices):
        n_agents = self.agents.n_agents
        applied_choices = np.ones(n_agents, dtype=int)

        # create a 2D array, storing for each grid cell if it's occupied or not
        grid_occupancy = np.zeros(shape=(self.agents.n_lanes, self.agents.lane_length), dtype=bool)
        grid_occupancy[self.agents.lanes, self.agents.positions] = True

        agent_order = np.random.permutation(n_agents)

        for agent_index in agent_order:
            choice = choices[agent_index]
            lane = self.agents.lanes[agent_index]
            position = self.agents.positions[agent_index]

            if choice == 1:
                continue
            
            target_lane = lane - 1 + choice

            if not grid_occupancy[target_lane, position]:
                self.agents.lanes[agent_index] = target_lane
                applied_choices[agent_index] = choice

                # update occupancy grid with newly moved agent
                grid_occupancy[lane, position] = False
                grid_occupancy[target_lane, position] = True

        return applied_choices
