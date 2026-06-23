from agent import Agents
from model import BaseNaSchModel, SwitchingNaSchModel

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

def run_simulation(params):
    (model_class, n_lanes, lane_length, v_max, bias_strength, rho, p, n_steps, n_measure_steps, model_kwargs) = params

    n_agents = int(rho * n_lanes * lane_length)
    if (n_agents == 0):
        return 0.0, 0.0, 0.0, 0.0

    agents = Agents(
        n_agents=n_agents,
        n_lanes=n_lanes,
        lane_length=lane_length,
        v_max=v_max,
        bias_strength=bias_strength,
        **model_kwargs
    )
    model = model_class(agents=agents, slowdown=p)

    # run for a given number of steps, such that we are likely to be in a state approximating equilibrium
    for _ in range(n_steps):
        model.step()

    total_vel = 0.0
    total_flow = 0.0
    total_stopped_fraction = 0.0

    # assuming we're in equilibrium, estimate measures over a small number of steps
    for _ in range(n_measure_steps):
        model.step()
        
        avg_velocity = np.mean(model.agents.velocities)
        total_vel += avg_velocity
        total_flow += avg_velocity * rho

        stopped_fraction = np.mean(model.agents.velocities == 0)
        total_stopped_fraction += stopped_fraction
    
    # calculate the actual metrics
    avg_velocity = total_vel / n_measure_steps
    flow = total_flow / n_measure_steps

    # order parameter for the limit p -> 1, based on paper 'Criticality in Dynamic Arrest: Correspondence between Glasses and Traffic'
    freeflow_velocity = agents.v_max - p # avg velocity of vehicles in equilibrium, if no interaction was present (we don't break to keep distance, don't laneswap, etc...)
    M1 = (freeflow_velocity - avg_velocity) / freeflow_velocity 

    # order parameter for the limit p -> 0, based on paper 'Traffic-flow cellular automaton: Order parameter and its conjugated field'
    M2 = 1.0 - avg_velocity / model.agents.v_max 

    # custom order parameter for the full range of p and density, though I think the paper 'Density fluctuations and phase transition in the Nagel-Schreckenberg traffic flow model' does something similar
    M3 = total_stopped_fraction / n_measure_steps

    # custom order parameter for the full range, where we measure what fraction of the max possible speed is achieved. The max possible speed, given global knowledge, is ~ min (v_max, (1-rho) / rho)
    optimal_avg_speed = min(model.agents.v_max, (1 - rho) / rho)
    M4 = avg_velocity / optimal_avg_speed

    return flow, avg_velocity, M3, M4

def take_measurements(
    model_class,
    n_lanes: int = 3,
    lane_length: int = 100,
    v_max: int = 5,
    bias_strength: float = 0.0,
    samples: int = 10,
    n_steps: int = 100,
    n_measure_steps: int = 100,
    **model_kwargs
):
    densities = np.linspace(0.05, 0.9, 20)
    slowdowns = np.linspace(0.01, 0.99, 20)

    # gather all combinations of parameters we want to test
    params_list = []
    for rho in densities:
        for p in slowdowns:
            for _ in range(samples):
                params_list.append((model_class, n_lanes, lane_length, v_max, bias_strength, rho, p, n_steps, n_measure_steps, model_kwargs))
                
    # run simulations on as many logical cores as available >:D
    results = process_map(run_simulation, params_list, max_workers=cpu_count() - 2, chunksize=4)

    # reshape into separate 2D arrays for each metric
    n_rho = len(densities)
    n_p = len(slowdowns)

    flow_mean = np.zeros((n_rho, n_p))
    flow_std = np.zeros((n_rho, n_p))
    velocity_mean = np.zeros((n_rho, n_p))
    velocity_std = np.zeros((n_rho, n_p))
    M3_mean = np.zeros((n_rho, n_p))
    M3_std = np.zeros((n_rho, n_p))
    M4_mean = np.zeros((n_rho, n_p))
    M4_std = np.zeros((n_rho, n_p))

    idx = 0
    for i in range(n_rho):
        for j in range(n_p):
            block = results[idx : idx + samples]
            flows = [result[0] for result in block]
            velocities = [result[1] for result in block]
            M3 = [result[2] for result in block]
            M4 = [result[3] for result in block]

            flow_mean[i, j] = np.mean(flows)
            flow_std[i, j] = np.std(flows)
            velocity_mean[i, j] = np.mean(velocities)
            velocity_std[i, j] = np.std(velocities)
            M3_mean[i, j] = np.mean(M3)
            M3_std[i, j] = np.std(M3)
            M4_mean[i, j] = np.mean(M4)
            M4_std[i, j] = np.std(M4)

            idx += samples

    return {
        'densities': densities,
        'slowdowns': slowdowns,
        'flow_mean': flow_mean,
        'flow_std': flow_std,
        'velocity_mean': velocity_mean,
        'velocity_std': velocity_std,
        'M3_mean': M3_mean,
        'M3_std': M3_std,
        'M4_mean': M4_mean,
        'M4_std': M4_std
    }

def plot_generics(
        data_list, 
        labels = None, 
        plot_names = None,
        v_max: int = 5
    ):
    if plot_names is None:
        plot_names = ['flow', 'velocity', 'M3', 'M4']
    if labels is None:
        labels = ['Baseline NaSch', 'Historic Switching', 'Biased Switching']

    n_experiments = len(data_list)
    n_plots = len(plot_names)

    fig, axes = plt.subplots(
        n_plots, n_experiments + 1,
        figsize=(5 * n_experiments + 1.2, 4 * n_plots),
        gridspec_kw={'width_ratios': [1] * n_experiments + [0.05]}
    )

    # account for edge-cases where plt.subplots wouldn't return a 2D array
    if n_experiments == 1 and n_plots == 1:
        axes = np.array([axes])
    elif n_plots == 1:
        axes = axes[np.newaxis, :]

    for experiment_index, data in enumerate(data_list):
        X, Y = np.meshgrid(data['densities'], data['slowdowns'], indexing='ij')

        row_index = 0
        if 'flow' in plot_names:
            ax = axes[row_index, experiment_index]
            cont = ax.contourf(X, Y, data['flow_mean'], levels=20, cmap='viridis', vmin=0, vmax=1)
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('slowdown probability')
            ax.text(0.98, 0.97, 'mean flow', transform=ax.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7))
            #if experiment_index == 0:
            #    ax.text(-0.25, 0.5, 'mean flow', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=12)
            if experiment_index == n_experiments - 1:
                bar_axis = axes[row_index, -1]
                fig.colorbar(cont, cax=bar_axis) 

            row_index += 1

        if 'velocity' in plot_names:
            ax = axes[row_index, experiment_index]
            cont = ax.contourf(X, Y, data['velocity_mean'], levels=20, cmap='viridis', vmin=0, vmax=v_max)
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('slowdown probability')
            ax.text(0.98, 0.97, 'mean velocity', transform=ax.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7))
            #if experiment_index == 0:
            #    ax.text(-0.25, 0.5, 'mean velocity', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=12)
            if experiment_index == n_experiments - 1:
                bar_axis = axes[row_index, -1]
                fig.colorbar(cont, cax=bar_axis) 

            row_index += 1

        if 'M3' in plot_names:
            ax = axes[row_index, experiment_index]
            cont = ax.contourf(X, Y, data['M3_mean'], levels=20, cmap='viridis', vmin=0, vmax=1)
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('slowdown probability')
            ax.text(0.98, 0.97, r'$M_3$ (fraction stopped)', transform=ax.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7))
            #if experiment_index == 0:
            #    ax.text(-0.25, 0.5, r'$M_3$ (fraction stopped)', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=12)
            if experiment_index == n_experiments - 1:
                bar_axis = axes[row_index, -1]
                fig.colorbar(cont, cax=bar_axis) 

            row_index += 1

        if 'M4' in plot_names:
            ax = axes[row_index, experiment_index]
            cont = ax.contourf(X, Y, data['M4_mean'], levels=20, cmap='viridis', vmin=0, vmax=1)
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('slowdown probability')
            ax.text(0.98, 0.97, r'$M_4$ (efficiency)', transform=ax.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7))
            #if experiment_index == 0:
            #    ax.text(-0.25, 0.5, r'$M_4$ (efficiency)', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=12)
            if experiment_index == n_experiments - 1:
                bar_axis = axes[row_index, -1]
                fig.colorbar(cont, cax=bar_axis) 

            row_index += 1

    for experiment_index, label in enumerate(labels):
       axes[0, experiment_index].set_title(label, fontsize=12)

    return fig


if __name__ == "__main__":
    n_lanes, lane_length, v_max = 5, 100, 1
    samples, n_steps, n_measure_steps = 5, 100, 100

    baseline_data = take_measurements(
        model_class=BaseNaSchModel,
        n_lanes=n_lanes,
        lane_length=lane_length,
        v_max=v_max,
        bias_strength=0.0,
        samples=samples,
        n_steps=n_steps,
        n_measure_steps=n_measure_steps
    )

    baseline_figure = plot_generics(
        data_list=[baseline_data, baseline_data, baseline_data],
        plot_names=['flow', 'M3'],
        v_max=v_max
    )

    """
    history_data = take_measurements(
        model_class=SwitchingNaSchModel,
        n_lanes=n_lanes,
        lane_length=lane_length,
        v_max=v_max,
        bias_strength=0.0,
        samples=samples,
        n_steps=n_steps,
        n_measure_steps=n_measure_steps
    )

    biased_data = take_measurements(
        model_class=SwitchingNaSchModel,
        n_lanes=n_lanes,
        lane_length=lane_length,
        v_max=v_max,
        bias_strength=0.5,
        samples=samples,
        n_steps=n_steps,
        n_measure_steps=n_measure_steps
    )

    baseline_figure = plot_generics(
        data_list=[baseline_data, history_data, biased_data],
        labels=['Baseline NaSch', 'Historic Switching', 'Biased Switching'],
        plot_names=['flow', 'M3'],
        v_max=v_max
    )
    """

    plt.show()