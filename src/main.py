from agent import Agents
from model import Model

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

def run_simulation(params):
    n_lanes, lane_length, v_max, density, slowdown, n_steps = params

    n_agents = int(density * n_lanes * lane_length)
    if (n_agents == 0):
        return 0.0, 0.0, 0.0, 0.0

    agents = Agents(n_agents=n_agents, n_lanes=n_lanes, lane_length=lane_length, v_max=v_max)
    model = Model(agents=agents, slowdown=slowdown)

    # run for a given number of steps, such that we are likely to be in a state approximating equilibrium
    for _ in range(n_steps):
        model.step()

    n_measure_steps = 25
    total_vel = 0.0
    total_flow = 0.0
    total_stay_fraction = 0.0

    # assuming we're in equilibrium, estimate measures over a small number of steps
    for _ in range(n_measure_steps):
        model.step()
        
        avg_velocity = np.mean(model.agents.velocities)
        total_vel += avg_velocity
        total_flow += avg_velocity * density

        probabability_stay = np.mean(agents.choice_weights[:, 1] / np.sum(agents.choice_weights, axis=1)) # relative prominence of staying, over switching lanes (assuming gaps in each lane are identical)
        total_stay_fraction += probabability_stay
    
    # calculate the actual metrics
    avg_velocity = total_vel / n_measure_steps
    flow = total_flow / n_measure_steps
    stay_metric = total_stay_fraction / n_measure_steps

    freeflow_velocity = agents.v_max - slowdown # avg velocity of vehicles in equilibrium, if no interaction was present (we don't break to keep distance, don't laneswap, etc...)
    M = (freeflow_velocity - avg_velocity) / freeflow_velocity

    return flow, avg_velocity, stay_metric, M


n_lanes = 3
lane_length = 100
v_max = 5
samples = 100
n_steps = 1000

densities = np.linspace(0.05, 0.9, 20)
slowdowns = np.linspace(0.01, 0.99, 20)

# =================== stage 1: Generic metrics ===================
# gather all combinations of parameters we want to test
params_list = []
for rho in densities:
    for p in slowdowns:
        for sample in range(samples):
            params_list.append((n_lanes, lane_length, v_max, rho, p, n_steps))

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

idx = 0
for i in range(n_rho):
    for j in range(n_p):
        block = results[idx : idx + samples]
        flows = [result[0] for result in block]
        velocities = [result[1] for result in block]
        stay_metrics = [result[2] for result in block]

        flow_mean[i, j] = np.mean(flows)
        flow_std[i, j] = np.std(flows)
        velocity_mean[i, j] = np.mean(velocities)
        velocity_std[i, j] = np.std(velocities)
        stay_mean[i, j] = np.mean(stay_metrics)
        stay_std[i, j] = np.std(stay_metrics)

        idx += samples

# =================== stage 2: Specific metrics ===================
# because some metrics, like the order parameter, use a transformed parameter to plot, it is nicer to sample this transformed parameter space
# TODO: use non-linear options for p, because for the NaSch model, we get interesting behaviour for M for p->1. So maybe smth like [0.5, 0.8, 0.9, 0.95, 0.98, 0.993, 0.999]
aproximate_critical_densities = (1 - slowdowns) / (v_max + 1 - 2 * slowdowns) # more exact solution, but transforms the results such that they aren't very readable
reduced_densities = np.linspace(0.1, 2.0, 20)

params_list_2 = []
for reduced_rho in reduced_densities:
    for i, p in enumerate(slowdowns):
        rho = reduced_rho * aproximate_critical_densities[i]
        for sample in range(samples):
            params_list_2.append((n_lanes, lane_length, v_max, rho, p, n_steps))

print("stage 2: unique metrics")
results_2 = process_map(run_simulation, params_list_2, max_workers=cpu_count() - 2, chunksize=4)

n_reduced = len(reduced_densities)
M_mean = np.zeros((n_reduced, n_p))
M_std = np.zeros((n_reduced, n_p))

idx = 0
for i in range(n_reduced):
    for j in range(n_p):
        block = results_2[idx : idx + samples]
        M_values = [r[3] for r in block]

        M_mean[i, j] = np.mean(M_values)
        M_std[i, j] = np.std(M_values)
        idx += samples

# contour plots

X, Y = np.meshgrid(densities, slowdowns, indexing='ij')
X_reduced, Y_reduced = np.meshgrid(reduced_densities, slowdowns, indexing='ij')

X_transformed = X / aproximate_critical_densities # transformed, for the order parameter plot, so it's similar to the De Wijn paper for comparison

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
cont = ax.contourf(X_reduced, Y_reduced, M_mean, levels=20, cmap='viridis')
ax.set_xlabel(r'$\rho / \rho_{\text{crit}}$')
ax.set_ylabel('slowdown probability')
ax.set_title('order parameter M')
fig.colorbar(cont, ax=ax)

fig.suptitle(f'v_max: {v_max}, lanes: {n_lanes}, lane length: {lane_length}, samples: {samples}, warmup steps: {n_steps}')
plt.tight_layout()
plt.show()