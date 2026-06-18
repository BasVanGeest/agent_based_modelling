from agent import Agents
from model import Model

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

def run_simulation(params):
    n_lanes, lane_length, v_max, density, slowdown, n_steps, n_measure_steps = params

    n_agents = int(density * n_lanes * lane_length)
    if (n_agents == 0):
        return 0.0, 0.0, 0.0, 0.0

    agents = Agents(n_agents=n_agents, n_lanes=n_lanes, lane_length=lane_length, v_max=v_max)
    model = Model(agents=agents, slowdown=slowdown)

    # run for a given number of steps, such that we are likely to be in a state approximating equilibrium
    for _ in range(n_steps):
        model.step()

    total_vel = 0.0
    total_flow = 0.0
    total_stay_fraction = 0.0
    total_stopped_fraction = 0.0

    # assuming we're in equilibrium, estimate measures over a small number of steps
    for _ in range(n_measure_steps):
        model.step()
        
        avg_velocity = np.mean(model.agents.velocities)
        total_vel += avg_velocity
        total_flow += avg_velocity * density

        probabability_stay = np.mean(agents.choice_weights[:, 1] / np.sum(agents.choice_weights, axis=1)) # relative prominence of staying, over switching lanes (assuming gaps in each lane are identical)
        total_stay_fraction += probabability_stay

        stopped_fraction = np.mean(model.agents.velocities == 0)
        total_stopped_fraction += stopped_fraction
    
    # calculate the actual metrics
    avg_velocity = total_vel / n_measure_steps
    flow = total_flow / n_measure_steps
    stay_metric = total_stay_fraction / n_measure_steps

    # order parameter for the limit p -> 1, based on paper 'Criticality in Dynamic Arrest: Correspondence between Glasses and Traffic'
    freeflow_velocity = agents.v_max - slowdown # avg velocity of vehicles in equilibrium, if no interaction was present (we don't break to keep distance, don't laneswap, etc...)
    M1 = (freeflow_velocity - avg_velocity) / freeflow_velocity 

    # order parameter for the limit p -> 0, based on paper 'Traffic-flow cellular automaton: Order parameter and its conjugated field'
    M2 = 1.0 - avg_velocity / model.agents.v_max 

    # custom order parameter for the full range of p and density, though I think the paper 'Density fluctuations and phase transition in the Nagel-Schreckenberg traffic flow model' does something similar
    M3 = total_stopped_fraction / n_measure_steps

    # custom order parameter for the full range, where we measure what fraction of the max possible speed is achieved. The max possible speed, given global knowledge, is ~ min (v_max, (1-rho) / rho)
    optimal_avg_speed = min(model.agents.v_max, (1 - density) / density)
    M4 = avg_velocity / optimal_avg_speed
    # TODO: maybe use the Gini coefficient as order parameter? Might be more robust then just the fraction of vehicles with velocity 0

    return flow, avg_velocity, stay_metric, M3, M4


n_lanes = 3
lane_length = 100
v_max = 5
samples = 20
n_steps = 1000
n_measure_steps = 500

densities = np.linspace(0.05, 0.9, 20)
slowdowns = np.linspace(0.01, 0.99, 20)

# gather all combinations of parameters we want to test
params_list = []
for rho in densities:
    for p in slowdowns:
        for sample in range(samples):
            params_list.append((n_lanes, lane_length, v_max, rho, p, n_steps, n_measure_steps))

# run simulations on as many logical cores as available >:D
print("stage 1: generic metrics")
results = process_map(run_simulation, params_list, max_workers=cpu_count() - 2, chunksize=4)

# reshape into separate 2D arrays for each metric
n_rho = len(densities)
n_p = len(slowdowns)

flow_mean = np.zeros((n_rho, n_p))
flow_std = np.zeros((n_rho, n_p))
velocity_mean = np.zeros((n_rho, n_p))
velocity_std = np.zeros((n_rho, n_p))
stay_mean = np.zeros((n_rho, n_p))
stay_std = np.zeros((n_rho, n_p))
M_mean = np.zeros((n_rho, n_p))
M_std = np.zeros((n_rho, n_p))

idx = 0
for i in range(n_rho):
    for j in range(n_p):
        block = results[idx : idx + samples]
        flows = [result[0] for result in block]
        velocities = [result[1] for result in block]
        stay_metrics = [result[2] for result in block]
        M = [result[4] for result in block]

        flow_mean[i, j] = np.mean(flows)
        flow_std[i, j] = np.std(flows)
        velocity_mean[i, j] = np.mean(velocities)
        velocity_std[i, j] = np.std(velocities)
        stay_mean[i, j] = np.mean(stay_metrics)
        stay_std[i, j] = np.std(stay_metrics)
        M_mean[i, j] = np.mean(M)
        M_std[i, j] = np.std(M)

        idx += samples

# contour plots

X, Y = np.meshgrid(densities, slowdowns, indexing='ij')

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# mean flow
ax = axes[0, 0]
cont = ax.contourf(X, Y, flow_mean, levels=20, cmap='viridis')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('slowdown probability')
ax.set_title('mean flow')
fig.colorbar(cont, ax=ax)

# mean velocity
ax = axes[0, 1]
cont = ax.contourf(X, Y, velocity_mean, levels=20, cmap='viridis')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('slowdown probability')
ax.set_title('mean velocity')
fig.colorbar(cont, ax=ax)

# mean stay metric
ax = axes[1, 0]
cont = ax.contourf(X, Y, stay_mean, levels=20, cmap='viridis')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('slowdown probability')
ax.set_title('mean fraction of weights towards stay')
fig.colorbar(cont, ax=ax)

# order parameter
ax = axes[1, 1]
cont = ax.contourf(X, Y, M_mean, levels=20, cmap='viridis')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('slowdown probability')
ax.set_title('order parameter M4 (fraction of max achievable avg velocity)')
fig.colorbar(cont, ax=ax)

fig.suptitle(f'v_max: {v_max}, lanes: {n_lanes}, lane length: {lane_length}, samples: {samples}, warmup steps: {n_steps}')
plt.tight_layout()
plt.show()