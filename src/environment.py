from agent import Agent

import numpy as np

class Environment:
    def __init__(self, n_lanes, lane_length, v_max, n_agents):
        # initialize environment parameters
        self.n_lanes = n_lanes
        self.lane_length = lane_length
        self.v_max = v_max # this could be fun to make a parameter of the agent, such that we can vary it between agents
        self.n_agents = n_agents

        self.agents = []

        # part of the agent state is stored in the environment, which allows us to compute the NaSch part of each iteration much quicker (it allows for numpy vectorization)
        self.lanes = np.array([], dtype=int)
        self.positions = np.array([], dtype=int)
        self.velocity = np.array([], dtype=int)

        # initialize agents to random unique positions on the grid
        full_positions = np.random.choice(self.n_lanes * self.lane_length, size=self.n_agents, replace=False)
        self.lanes = full_positions // self.lane_length
        self.positions = full_positions % self.lane_length
        self.velocity = np.zeros(self.n_agents, dtype=int)
        
        for _ in range(self.n_agents):
            self.agents.append(Agent())

    def compute_gaps(self):
        gaps = np.zeros(shape=(self.n_agents, 3), dtype=int)

        for lane in range(self.n_lanes):
            mask = (self.lanes == lane) # filter out all entries of vehicles within the current lane

            position_in_lane = self.positions[mask]
            sorted_indices = np.argsort(position_in_lane)
            position_sorted = position_in_lane[sorted_indices]

            gaps_sorted = np.roll(position_sorted, -1) - position_sorted - 1
            gaps_sorted[-1] = position_sorted[0] + self.lane_length - position_sorted[-1] - 1

            gaps[mask, 1] = gaps_sorted[np.argsort(sorted_indices)] # go back to original order

            # TODO: left and right lanes
            # TODO: account for left and right that are outside the range of lanes

        return gaps