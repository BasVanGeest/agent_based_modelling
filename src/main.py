import numpy as np
import time
import matplotlib.pyplot as plt
from environment import Environment
from model import Model

n_samples = 10
for t in np.linspace(10, 1000, 5):
    slowdown_probabilities = np.linspace(0, 1, 15)
    average_relative_speed = []
    for p in slowdown_probabilities:
        avg_speed = 0
        for _ in range(n_samples):
            # todo sample same parameter multiple times
            environment = Environment(n_lanes=3, lane_length=100, v_max=5, n_agents=30)
            model = Model(environment=environment, slowdown=p)

            for _ in range(int(t)):
                # print(model.environment.positions)
                # print(model.environment.velocity)
                # print("------------------------")
                model.step()
            avg_speed += np.average(model.environment.velocity) / model.environment.v_max
        
        average_relative_speed.append(avg_speed / n_samples)

    plt.plot(slowdown_probabilities, average_relative_speed, label=f'steps:{int(t)}')

plt.xlabel("slowdown probability")
plt.ylabel("average speed")
plt.legend()
plt.show()