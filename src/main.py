import numpy as np
import time
import matplotlib.pyplot as plt
from environment import Environment
from model import Model

slowdown_probabilities = np.linspace(0, 1, 15)
average_relative_speed = []
for p in slowdown_probabilities:
    # todo sample same parameter multiple times
    environment = Environment(n_lanes=1, lane_length=100, v_max=5, n_agents=10)
    model = Model(environment=environment, slowdown=p)

    for _ in range(100):
        # print(model.environment.positions)
        # print(model.environment.velocity)
        # print("------------------------")
        model.step()
    average_relative_speed.append(np.average(model.environment.velocity) / model.environment.v_max)

plt.plot(slowdown_probabilities, average_relative_speed)
plt.xlabel("slowdown probability")
plt.ylabel("average speed")
plt.show()