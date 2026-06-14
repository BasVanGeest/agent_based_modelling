import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from model import Model

# Create model
model = Model(num_lanes=10, lane_length=100, num_cars=200, slowdown_probability=0.3)

fig, ax = plt.subplots(figsize=(12, 4))
plt.ion()
