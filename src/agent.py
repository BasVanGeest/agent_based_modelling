import numpy as np

class Agents:
    def __init__(self, n_agents, n_lanes, lane_length, v_max=5, risk_factor=0.75, loss_factor=2, loss_scale=3, experience_vs_immediate=0.7, learning_rate=0.1, rationality=5):
        # local copies of global info, for ease of use
        self.n_agents = n_agents
        self.n_lanes = n_lanes
        self.lane_length = lane_length
        
        # homogeneous variables of agents
        self.risk_factor = risk_factor
        self.loss_factor = loss_factor
        self.loss_scale = loss_scale
        self.experience_vs_immediate = experience_vs_immediate
        self.learning_rate = learning_rate
        self.rationality = rationality
        self.v_max = v_max

        # heterogeneous variables
        self.choice_velocity_history = np.ones((self.n_agents, 3)) * (self.v_max / 2) # rough starting guess of historic avg velocity per vehicle

        full_positions = np.random.choice(self.n_lanes * self.lane_length, size=self.n_agents, replace=False)
        self.lanes = full_positions // self.lane_length
        self.positions = full_positions % self.lane_length

        self.velocities = np.zeros(self.n_agents, dtype=int)


    def compute_stay_gaps(self):
        gaps_stay = np.zeros(self.n_agents, dtype=int)

        for lane in range(self.n_lanes):
            mask = (self.lanes == lane)
            positions_in_lane = self.positions[mask]

            if positions_in_lane.size == 0:
                continue

            sorted_indices = np.argsort(positions_in_lane)
            positions_sorted = positions_in_lane[sorted_indices]

            gaps_sorted = np.diff(positions_sorted) - 1
            gaps_sorted = np.append(gaps_sorted, positions_sorted[0] + self.lane_length - positions_sorted[-1] - 1)

            # go back to original (unsorted) order
            inverse_indices = np.zeros_like(sorted_indices)
            inverse_indices[sorted_indices] = np.arange(len(sorted_indices))
            gaps_stay[mask] = gaps_sorted[inverse_indices]
        
        return gaps_stay


    def compute_gaps(self):
        gaps = np.zeros(shape=(self.n_agents, 3), dtype=int)

        # compute front / center lane gaps, and store sorted positions for later gap computations for side-lanes
        lane_positions = []
        
        for lane in range(self.n_lanes):
            mask = (self.lanes == lane) # filter out all entries of vehicles within the current lane
            positions_in_lane = self.positions[mask]

            if positions_in_lane.size == 0:
                lane_positions.append(np.array([]))
                continue
            
            sorted_indices = np.argsort(positions_in_lane)
            positions_sorted = positions_in_lane[sorted_indices]

            gaps_sorted = np.diff(positions_sorted) - 1
            gaps_sorted = np.append(gaps_sorted, positions_sorted[0] + self.lane_length - positions_sorted[-1] - 1)

            lane_positions.append(positions_sorted)

            # go back to original (unsorted) order
            inverse_indices = np.zeros_like(sorted_indices)
            inverse_indices[sorted_indices] = np.arange(len(sorted_indices))
            gaps[mask, 1] = gaps_sorted[inverse_indices] 


        # compute gaps for each agent, if hypothetically, they would switch to either of the 2 options
        for lane in range(self.n_lanes):
            mask = (self.lanes == lane)
            positions_in_lane = self.positions[mask]

            if positions_in_lane.size == 0:
                continue

            for offset, choice in [(-1, 0), (1, 2)]:
                target_lane = lane + offset
                if (target_lane < 0 or target_lane >= self.n_lanes):
                    continue

                target_positions = lane_positions[target_lane]

                if len(target_positions) == 0:
                    gaps[mask, choice] = self.lane_length - 1
                else:
                    # see behind which vehicle they would be placed, if they'd switch to this lane
                    indices = np.searchsorted(target_positions, positions_in_lane, side='left')

                    # account for the circular boundary condition at the end of the lane, by filtering for those vehicles that would be in the front position if switched
                    gaps_lane = np.empty_like(positions_in_lane, dtype=int)
                    wrapping = (indices == len(target_positions))

                    # non wrapping: gap to the next vehicle
                    gaps_lane[~wrapping] = target_positions[indices[~wrapping]] - positions_in_lane[~wrapping] - 1
                    # wrapping: gap to the end of the lane + gap from the start of the lane to the first vehicle
                    gaps_lane[wrapping] = target_positions[0] + self.lane_length - positions_in_lane[wrapping] - 1

                    gaps[mask, choice] = gaps_lane

        return gaps


    def choose_actions(self):
        # TODO: maybe only consider switching if you expect to not be able to accelerate while staying
        # TODO: maybe don't allow switching when your current velocity is 0

        # compute the gaps to the next vehicle ahead, for each vehicle, for each option
        forward_gaps = self.compute_gaps()

        # create a mask to filter out all switches that would place the vehicle on a non-existing lane
        viable_options = np.ones(shape=(self.n_agents, 3), dtype=bool)
        viable_options[self.lanes == 0, 0] = False
        viable_options[self.lanes == self.n_lanes - 1, 2] = False

        # compute weighted utility of each option
        raw_utility = self.experience_vs_immediate * self.choice_velocity_history + (1 - self.experience_vs_immediate) * np.minimum(forward_gaps, self.v_max)

        # apply loss + risk aversion using the prospect theory expression
        references = self.velocities[:, np.newaxis]
        delta = raw_utility - references

        utility = np.zeros_like(delta)
        gain_mask = delta >= 0
        loss_mask = delta < 0
        utility[gain_mask] = np.pow(delta[gain_mask], self.risk_factor)
        utility[loss_mask] = -self.loss_scale * np.pow(-delta[loss_mask], self.loss_factor)
        # add small negative bias, nudging away from switching
        utility[:, [0, 2]] -= 1

        # account for bounded rationality using logit assumption
        utility_exp = np.multiply(viable_options, np.exp(self.rationality * utility))
        probabilities = utility_exp / np.sum(utility_exp, axis=1)[:, np.newaxis]

        # TODO: add loss aversion? It seems like a hard requirement from the project requirements list. Maybe ask TA's about it
        # TODO: use nested logit? surely if say, moving left isn't possible / removed, then most of the probability of going left goes to switching right; Not equally to staying and switch right. So we don't really have IIA here
        
        # make the choice between the viable options
        cumulative_sum = np.cumsum(probabilities, axis=1)
        random_values = np.random.random(self.n_agents) + 1e-10 # add a tiny amount, to aviod the edgecase where if the left option is non-viable, and the random-value is exactly 0, it would choose moving left anyway
        choices = np.argmax(cumulative_sum >= random_values[:, np.newaxis], axis=1)
        
        return choices, forward_gaps


    def update_velocities(self, slowdown):
        gaps_forward = self.compute_stay_gaps()
        self.velocities = np.minimum(self.velocities + 1, self.v_max) # acceleration
        self.velocities = np.minimum(self.velocities, gaps_forward) # braking
        random_values = np.random.random(self.n_agents)
        self.velocities = np.where((self.velocities > 0) & (random_values < slowdown), self.velocities - 1, self.velocities) # janky way of reducing velocity by 1 by a given percentage, if it wasn't zero already


    def update_weights(self, choices, velocities):
        self.choice_velocity_history[np.arange(self.n_agents), choices] = (1 - self.learning_rate) * self.choice_velocity_history[np.arange(self.n_agents), choices] + self.learning_rate * velocities