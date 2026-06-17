from agent import Agents
from model import Model

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def run_simulation(params):
    n_lanes, lane_length, density, slowdown, n_steps = params

    n_agents = int(density * n_lanes * lane_length)
    agents = Agents(n_agents=n_agents, n_lanes=n_lanes, lane_length=lane_length)
    model = Model(agents=agents, slowdown=slowdown)

    for _ in range(n_steps):
        model.step()

    n_measure_steps = 25
    total_vel = 0.0
    for _ in range(n_measure_steps):
        total_vel += np.mean(model.agents.velocities)
        model.step()
    
    avg_vel = total_vel / n_measure_steps
    flow = density * avg_vel
    return flow

n_lanes = 1
lane_length = 100
samples = 20
n_steps = 1000

densities = np.linspace(0.05, 0.9, 20)
slowdowns = np.linspace(0.01, 0.99, 20)

# gather all combinations of parameters we want to test
params_list = []
for rho in densities:
    for p in slowdowns:
        for sample in range(samples):
            params_list.append((n_lanes, lane_length, rho, p, n_steps))

# run all simulations in a single pool, so we get decently efficient usage of all available CPU cores
with Pool(processes=cpu_count() - 2) as pool:
    results = pool.map(run_simulation, params_list)


# reshape into 2D grid
# results are in order of rho0, p0, [samples], rho0, p1, [samples], etc...
mean_flows = np.zeros((len(densities), len(slowdowns)))
std_flows = np.zeros((len(densities), len(slowdowns)))

idx = 0
for i in range(len(densities)):
    for j in range(len(slowdowns)):
        block = results[idx : idx + samples]
        mean_flows[i, j] = np.mean(block)
        std_flows[i, j] = np.std(block)
        idx += samples

# contour plot
X, Y = np.meshgrid(densities, slowdowns, indexing='ij')

fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X, Y, mean_flows, levels=20, cmap='viridis')
ax.set_xlabel('Density')
ax.set_ylabel('Slowdown probability')
ax.set_title('Contour Plot of Mean Flow')
fig.colorbar(contour, label='Mean Flow')
plt.show()