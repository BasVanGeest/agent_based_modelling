import numpy as np

class Agent:
    def __init__(self, choice_weights=np.array([1, 1, 1]), risk_factor=0.0, learning_rate=0.5, rationality=1):
        # initialize agent parameters
        self.risk_factor = risk_factor
        self.learning_rate = learning_rate
        self.rationality = rationality

        # initialize agent state
        self.choice_weights = choice_weights # order: left, stay, right
        self.lane = 0
        self.pos = 0
        self.vel = 0

    def choose_action(self, gaps):
        # given the gaps to the first vehicle, for each hypothetical choice (order: left, stay, right), make a choice

        # this is where we could modify the utility function, to either use more types of information, or make it non-linear
        raw_utility_values = self.choice_weights * gaps

        # account for risk aversion, with some concave-convex function
        utility = raw_utility_values if np.abs(self.risk_factor) < 1e-8 else (1 - np.exp(-self.risk_factor * raw_utility_values)) / self.risk_factor

        # account for bounded rationality using logit assumption
        utility_exp = np.exp(self.rationality * utility)
        probabilities = utility_exp / np.sum(utility_exp)

        # make the choice between the 3 options
        option = np.random.choice(3, p=probabilities)
        return option
    
    def update_weights(self, option, gap, reward):
        target = reward # + gamma * max(...), if we would use some info about the best choice in the next iteration (see q-learning, or overleaf)
        predicted_reward = self.choice_weights[option] * gap
        difference = target - predicted_reward
        self.choice_weights[option] += self.learning_rate * difference * gap