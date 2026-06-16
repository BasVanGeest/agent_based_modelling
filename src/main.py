import numpy as np
import time
import matplotlib.pyplot as plt
from agent import Agents
from model import Model

agents = Agents(n_agents=30, n_lanes=3, lane_length=100)
model = Model(agents=agents, slowdown=0.05)

start_time = time.time()
for _ in range(10000):
    model.step()

print("total time taken: ", time.time() - start_time)
