import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.widgets import Button

from agent import Agents
from model import BaseNaSchModel, SwitchingNaSchModel

class BasicStepwiseVisualizer:
    def __init__(self, agents, model, trail_length):
        self.agents = agents
        self.model = model
        self.trail_length = trail_length

        # Generate distinct, bright colors
        self.agent_colors = self._get_agent_colors(agents.n_agents)

        # History: list of (lane, position) per agent
        self.history = [[] for _ in range(agents.n_agents)]

        self.fig, self.ax = plt.subplots(figsize=(4, 12))
        self.fig.subplots_adjust(bottom=0.2)

        ax_button = plt.axes([0.8, 0.05, 0.1, 0.05])
        self.button = Button(ax_button, 'step')
        self.button.on_clicked(self.next_step)

        # Axes: x = lane, y = position
        self.ax.set_xlim(-0.5, agents.n_lanes - 0.5)
        self.ax.set_ylim(-1, agents.lane_length)
        self.ax.set_xticks(np.arange(agents.n_lanes))
        self.ax.set_xlabel('Lane number')
        self.ax.set_ylabel('Position along lane')
        self.ax.set_title('Traffic Simulation (Step 0)')

        self.ax.xaxis.grid(linestyle='--', alpha=1.0, linewidth=1)
        self.ax.yaxis.grid(linestyle='--', alpha=0.3, linewidth=0.6)

        # Record initial positions (lane, position)
        for i in range(agents.n_agents):
            self.history[i].append((agents.lanes[i], agents.positions[i]))

        self.redraw()

    def _get_agent_colors(self, n):
        """
        Generate n distinct, bright colors that contrast well with white background.
        Uses HSL: hue spread evenly, high saturation, moderate brightness.
        """
        colors = []
        saturation = 0.9
        brightness = 0.8   # high but not too high to avoid washing out
        for i in range(n):
            hue = i / n
            # Optionally shift hue to avoid yellow (which is light on white)
            # We don't shift; with brightness 0.8 it's fine.
            rgb = mcolors.hsv_to_rgb((hue, saturation, brightness))
            colors.append(rgb)
        return colors

    def redraw(self):
        self.ax.clear()

        self.ax.set_xlim(-0.5, self.agents.n_lanes - 0.5)
        self.ax.set_ylim(-1, self.agents.lane_length)
        self.ax.set_xticks(np.arange(self.agents.n_lanes))
        self.ax.set_xlabel('Lane number')
        self.ax.set_ylabel('Position along lane')
        self.ax.set_title(f'Traffic Simulation, step {self.model.step_count}')

        self.ax.xaxis.grid(linestyle='--', alpha=1.0, linewidth=1)
        self.ax.yaxis.grid(linestyle='--', alpha=0.3, linewidth=0.6)

        # Trails
        for i in range(self.agents.n_agents):
            hist = self.history[i]
            if len(hist) > 1:
                lanes, positions = zip(*hist)
                self.ax.plot(lanes, positions, color=self.agent_colors[i], alpha=0.6, linestyle='-', linewidth=2)

        # Cars
        self.ax.scatter(
            self.agents.lanes,
            self.agents.positions,
            s=200,
            marker='s',
            color=self.agent_colors,
            zorder=2
        )

        self.fig.canvas.draw_idle()

    def next_step(self, event):
        self.model.step()

        for i in range(self.agents.n_agents):
            self.history[i].append((self.agents.lanes[i], self.agents.positions[i]))
            if len(self.history[i]) > self.trail_length:
                self.history[i].pop(0)

        self.redraw()


if __name__ == '__main__':
    density = 0.15
    n_lanes = 5
    lane_length = 30
    slowdown = 0.2

    # Future idea: look at the distribution of agents, their learned histories, and the number of switches, and velocities
    # Future idea: it seems like for some parameter combinations, vehicles roughly fall into separate groups: ones that switch rarely, and those that switch very often (crisscrossing through the whole thing)
    # agents = Agents(n_agents=n_agents, n_lanes=n_lanes, lane_length=lane_length, info_preference=0.9, learning_rate=0.5)

    n_agents = int(density * n_lanes * lane_length)
    agents = Agents(n_agents=n_agents, n_lanes=n_lanes, lane_length=lane_length, bias_strength=0.5)
    model = SwitchingNaSchModel(agents=agents, slowdown=slowdown)

    for _ in range(100):
        model.step()
    
    visualizer = BasicStepwiseVisualizer(agents, model, trail_length=2)

    plt.show()