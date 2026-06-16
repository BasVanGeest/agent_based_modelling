import numpy as np

class Agents:
    def __init__(self, n_agents, n_lanes, lane_length, v_max=5, choice_weights=np.array([0, 1, 0]), risk_factor=0.0, learning_rate=0.05, rationality=1):
        # local copies of global info, for ease of use
        self.n_agents = n_agents
        self.n_lanes = n_lanes
        self.lane_length = lane_length
        
        # homogeneous variables of agents
        self.risk_factor = risk_factor
        self.learning_rate = learning_rate
        self.rationality = rationality
        self.v_max = v_max

        # heterogeneous variables
        self.choice_weights = np.tile(choice_weights, (self.n_agents,1)).astype(float)

        full_positions = np.random.choice(self.n_lanes * self.lane_length, size=self.n_agents, replace=False)
        self.lanes = full_positions // self.lane_length
        self.positions = full_positions % self.lane_length

        self.velocities = np.zeros(self.n_agents, dtype=int)


    def compute_forward_gaps(self):
        gaps = np.zeros(shape=(self.n_agents, 3), dtype=int)
        # compute front / center lane gaps, and store sorted positions for later gap computations for side-lanes
        lane_positions = []
        lane_gaps = []
        for lane in range(self.n_lanes):
            mask = (self.lanes == lane) # filter out all entries of vehicles within the current lane
            if not np.any(mask):
                lane_positions.append(np.array([]))
                lane_gaps.append(np.array([]))
                continue
            
            position_in_lane = self.positions[mask]
            sorted_indices = np.argsort(position_in_lane)
            position_sorted = position_in_lane[sorted_indices]

            gaps_sorted = np.roll(position_sorted, -1) - position_sorted - 1
            gaps_sorted[-1] += self.lane_length

            lane_positions.append(position_sorted)
            lane_gaps.append(gaps_sorted)

            gaps[mask, 1] = gaps_sorted[np.argsort(sorted_indices)] # go back to original order

        # compute gaps for each agent, if hypothetically, they would switch to either of the 2 options
        for i in range(self.n_agents):
            lane = self.lanes[i]
            pos = self.positions[i]

            if lane > 0:
                left_positions = lane_positions[lane - 1] # sorted positions of all vehicles, for the left lane
                if len(left_positions) == 0:
                    gaps[i, 0] = self.lane_length - 1
                else:
                    index = np.searchsorted(left_positions, pos, side='left')
                    if index == len(left_positions):
                        gap = left_positions[0] + self.lane_length - pos - 1
                    else:
                        gap = left_positions[index] - pos - 1
                    gaps[i, 0] = gap

            if lane < self.n_lanes - 1:
                right_positions = lane_positions[lane + 1]
                if len(right_positions) == 0:
                    gaps[i, 2] = self.lane_length - 1
                else:
                    index = np.searchsorted(right_positions, pos, side='left')
                    if index == len(right_positions):
                        gap = right_positions[0] + self.lane_length - pos - 1
                    else:
                        gap = right_positions[index] - pos - 1
                    gaps[i, 2] = gap

        return gaps


    def choose_actions(self):
        # compute the gaps to the next vehicle ahead, for each vehicle, for each option
        forward_gaps = self.compute_forward_gaps()

        # create a mask to filter out all switches that would place the vehicle on a non-existing lane
        viable_options = np.ones(shape=(self.n_agents, 3), dtype=bool)
        viable_options[self.lanes == 0, 0] = False
        viable_options[self.lanes == self.n_lanes - 1, 2] = False

        # compute weighted utility of each option
        raw_utility = np.multiply(self.choice_weights, np.minimum(forward_gaps, self.v_max))

        # apply risk aversion
        utility = raw_utility if np.abs(self.risk_factor) < 1e-8 else (1 - np.exp(-self.risk_factor * raw_utility)) / self.risk_factor

        # account for bounded rationality using logit assumption
        utility_exp = np.multiply(viable_options, np.exp(self.rationality * utility))
        probabilities = utility_exp / np.sum(utility_exp, axis=1)[:, np.newaxis]

        # make the choice between the viable options
        # TODO: instead calculate cumulative probabilities, then generate n_agents random values in [0, 1], then find the first index where the cumulative sum is greater than this random value
        # TODO: that would likely be a decent amount faster than what we do now, since we're now just sequentually doing a random.choice
        choices = np.array([np.random.choice(3, p=row_probabilities) for row_probabilities in probabilities])
        
        return choices, forward_gaps
    

    def update_velocities(self, slowdown):
        gaps_forward = self.compute_forward_gaps()[:, 1]
        self.velocities = np.minimum(self.velocities + 1, self.v_max) # acceleration
        self.velocities = np.minimum(self.velocities, gaps_forward) # braking
        random_values = np.random.random(self.n_agents)
        self.velocities = np.where((self.velocities > 0) & (random_values < slowdown), self.velocities - 1, self.velocities) # janky way of reducing velocity by 1 by a given percentage, if it wasn't zero already


    def update_weights(self, choices, gaps):
        rewards = self.velocities
        used_weights = self.choice_weights[np.arange(self.n_agents), choices]
        used_gaps = gaps[np.arange(self.n_agents), choices]
        predicted_rewards = used_weights * np.minimum(used_gaps, self.v_max)
        difference = rewards - predicted_rewards
        self.choice_weights[np.arange(self.n_agents), choices] += self.learning_rate * difference * np.minimum(used_gaps, self.v_max)