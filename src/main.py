import numpy as np
import time

from environment import Environment
from model import Model

environment = Environment(n_lanes=2, lane_length=5, v_max=5, n_agents=2)
model = Model(environment=environment, slowdown=0.05)

start_time = time.time()
for _ in range(10000):
    # print(model.environment.positions)
    # print(model.environment.velocity)
    # print("------------------------")
    model.step()

print("total time taken: ", time.time() - start_time)