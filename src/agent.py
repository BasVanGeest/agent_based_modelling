import numpy as np

# TODO: maybe instead of using an Agent vs Environment vs Model class, which is somewhat inefficient here, since in various steps we can compute (parts of) updates for all agents at once,
# we can instead use an Agents class. This then contains the arrays of info, which keeps the efficient structure-of-arrays (SoA), while potentially keeping the code better separated by their responsibility

class Agent:
    def __init__(self, choice_weights=np.array([1, 1, 1]), risk_factor=0.0, learning_rate=0.05, rationality=1):
        # initialize agent parameters
        self.risk_factor = risk_factor
        self.learning_rate = learning_rate
        self.rationality = rationality

        # initialize agent state
        self.choice_weights = choice_weights # order: left, stay, right

    def choose_action(self, gaps, lane, n_lanes, v_max):
        # given the gaps to the first vehicle, for each hypothetical choice (order: left, stay, right), make a choice
        viable_options = np.ones(3, dtype=bool)
        if lane == 0: viable_options[0] = False
        if lane == n_lanes - 1: viable_options[2] = False

        # this is where we could modify the utility function, to either use more types of information, or make it non-linear
        raw_utility_values = self.choice_weights * np.minimum(gaps, v_max)
        
        # account for risk aversion, with some concave-convex function
        utility = raw_utility_values if np.abs(self.risk_factor) < 1e-8 else (1 - np.exp(-self.risk_factor * raw_utility_values)) / self.risk_factor

        # account for bounded rationality using logit assumption
        utility_exp = np.multiply(viable_options, np.exp(self.rationality * utility))
        probabilities = utility_exp / np.sum(utility_exp)

        # make the choice between the 3 options
        option = np.random.choice(3, p=probabilities)
        return option 
    
    def update_weights(self, option, gap, reward, v_max):
        target = reward # + gamma * max(...), if we would use some info about the best choice in the next iteration (see q-learning, or overleaf)
        predicted_reward = self.choice_weights[option] * np.minimum(gap, v_max)
        difference = target - predicted_reward
        self.choice_weights[option] += self.learning_rate * difference * np.minimum(gap, v_max)