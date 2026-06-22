from agent import Agents
from model import Model

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm.contrib.concurrent import process_map

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import pickle
import os
import itertools


def run_simulation(params):
    lane_length, n_lanes, density, slowdown, v_max, learning_rate, risk_factor, loss_factor, loss_scale, rationality, history_vs_current = params

    n_lanes = int(round(n_lanes))
    lane_length = int(round(lane_length))
    n_agents = int(density * n_lanes * lane_length)

    agents = Agents(
        n_agents=n_agents,
        n_lanes=n_lanes,
        lane_length=lane_length,
        v_max=int(v_max),
        risk_factor=risk_factor,
        loss_factor=loss_factor,
        loss_scale=loss_scale,
        info_preference=history_vs_current,
        learning_rate=learning_rate,
        rationality=rationality
    )
    model = Model(agents=agents, slowdown=slowdown)

    for _ in range(n_steps):
        model.step()

    total_flow = 0.0
    total_vel = 0.0
    total_stopped = 0.0

    for _ in range(n_sample_steps):
        model.step()
        avg_v = np.mean(model.agents.velocities)
        total_vel += avg_v
        total_flow += avg_v * density
        total_stopped += np.mean(model.agents.velocities == 0)

    flow = total_flow / n_sample_steps
    avg_velocity = total_vel / n_sample_steps
    M3 = total_stopped / n_sample_steps

    return flow, avg_velocity, M3


def run_parameter_set(params, replicates):
    results = [run_simulation(params) for _ in range(replicates)]
    avg_results = np.mean(results, axis=0)
    return tuple(avg_results)


if __name__ == "__main__":
    distinct_samples = 1024
    replicates = 5
    n_steps = 1000
    n_sample_steps = 100

    cache_file = "sobol_results.pkl"

    problem = {
        'num_vars': 11,
        'names': ['lane_length', 'n_lanes', 'density', 'slowdown', 'v_max', 'learning_rate', 'risk_factor', 'loss_factor', 'loss_scale', 'rationality', 'history_vs_current'],
        'bounds': [[25, 250], [1, 5], [0.05, 0.95], [0.05, 0.95], [1, 5], [0, 1], [0.5, 1.5], [0.5, 2.5], [1, 3], [1, 10], [0, 1]]
    }

    param_values = sobol_sample.sample(problem, distinct_samples)
    param_list = [row for row in param_values]

    # open stored results if they exist, otherwise run the simulation for all parameters and store them
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            Y = pickle.load(f)
    else:
        print(f"Running {len(param_list)} parameter sets with {replicates} replicates each...")
        run_replicates = partial(run_parameter_set, replicates=replicates)

        Y = process_map(run_replicates, param_list, max_workers=os.cpu_count() - 2, chunksize=4)

        with open(cache_file, 'wb') as f:
            pickle.dump(Y, f)

    Y = np.array(Y)

    output_names = ['Mean Flow', 'Mean Velocity', 'M3']
    param_names = problem['names']

    all_results = {}
    for idx, name in enumerate(output_names):
        Si = sobol_analyze.analyze(problem, Y[:, idx], print_to_console=False)
        all_results[name] = Si

    # Making our fancy plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for idx, (name, Si) in enumerate(all_results.items()):
        ax = axes[0, idx]
        x = np.arange(len(param_names))
        width = 0.35

        # first order indices
        ax.bar(x - width/2, Si['S1'], width, yerr=Si['S1_conf'], label='First-order', color='steelblue', capsize=3)
        # total order indices
        ax.bar(x + width/2, Si['ST'], width, yerr=Si['ST_conf'], label='Total-order', color='coral', capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Sensitivity index')
        ax.set_title(f'{name} (First vs Total order)', fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # second order indices
        ax = axes[1, idx]
        ax.imshow(Si['S2'], cmap='Reds', vmin=0, vmax=0.5)
        ax.set_xticks(np.arange(len(param_names)))
        ax.set_yticks(np.arange(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(param_names, fontsize=8)
        ax.set_title(f'{name} (Second-order interactions)', fontsize=10)

        for i, j in itertools.product(range(len(param_names)), range(len(param_names))):
            if not np.isnan(Si['S2'][i, j]):
                ax.text(j, i, f"{Si['S2'][i, j]:.2f}", ha='center', va='center', fontsize=6)
                

    plt.tight_layout()
    plt.show()