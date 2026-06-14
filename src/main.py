import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from environment import Environment
from model import Model

environment = Environment(n_lanes=1, lane_length=10, v_max=5, n_agents=2)
model = Model(environment=environment, slowdown=0.05)

for _ in range(5):
    print(model.environment.positions)
    print(model.environment.velocity)
    print("------------------------")
    model.step()
