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

        # generate unique, contrasting colors
        self.agent_colors = []
        for i in range(self.agents.n_agents):
            rgb = mcolors.hsv_to_rgb((i/self.agents.n_agents, 0.9, 0.8))
            self.agent_colors.append(rgb)

        # set up trail history arrays
        self.history = [[] for _ in range(agents.n_agents)]

        # set up the general figure
        self.fig, self.ax = plt.subplots(figsize=(4, 12))
        self.fig.subplots_adjust(bottom=0.2)

        ax_button = plt.axes([0.8, 0.05, 0.1, 0.05])
        self.button = Button(ax_button, 'step')
        self.button.on_clicked(self.next_step)

        self.ax.set_xlim(-0.5, agents.n_lanes - 0.5)
        self.ax.set_ylim(0, agents.lane_length)
        self.ax.set_xticks(np.arange(agents.n_lanes))
        self.ax.set_xlabel('Lane')
        self.ax.set_ylabel('Position along Lane')
        self.ax.set_title('Traffic Simulation (Step 0)')

        self.ax.xaxis.grid(linestyle='--', alpha=1.0, linewidth=1)
        self.ax.yaxis.grid(linestyle='--', alpha=0.3, linewidth=0.6)

        # record starting state
        for i in range(agents.n_agents):
            self.history[i].append((agents.lanes[i], agents.positions[i]))

        self.redraw()


    def redraw(self):
        # clean up the figure, and add the basic setup back in
        self.ax.clear()

        self.ax.set_xlim(-0.5, self.agents.n_lanes - 0.5)
        self.ax.set_ylim(0, self.agents.lane_length)
        self.ax.set_xticks(np.arange(self.agents.n_lanes))
        self.ax.set_xlabel('Lane')
        self.ax.set_ylabel('Position along Lane')
        self.ax.set_title(f'Traffic Simulation, step {self.model.step_count}')

        self.ax.xaxis.grid(linestyle='--', alpha=1.0, linewidth=1)
        self.ax.yaxis.grid(linestyle='--', alpha=0.3, linewidth=0.6)

        # draw trails, accounting for wrap-around effects
        for i in range(self.agents.n_agents):
            history = self.history[i]
            if len(history) <= 1:
                continue

            segments = []
            segment = [history[0]]
            for j in range(1, len(history)):
                prev_lane, prev_pos = history[j - 1]
                curr_lane, curr_pos = history[j]

                # no wrap-around, so just draw the trail segment
                if curr_pos >= prev_pos:
                    self.ax.plot([prev_lane, curr_lane], [prev_pos, curr_pos], color=self.agent_colors[i], alpha=0.6, linestyle='-', linewidth=2)
                else:
                    # wrap-around, so draw in 2 parts
                    distance = self.agents.lane_length - prev_pos + curr_pos
                    if distance == 0:
                        continue

                    middle_point = (self.agents.lane_length - prev_pos) / distance
                    middle_lane = prev_lane + middle_point * (curr_lane - prev_lane)
                    self.ax.plot([prev_lane, middle_lane], [prev_pos, self.agents.lane_length], color=self.agent_colors[i], alpha=0.6, linestyle='-', linewidth=2)
                    self.ax.plot([middle_lane, curr_lane], [0, curr_pos], color=self.agent_colors[i], alpha=0.6, linestyle='-', linewidth=2)

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
    density = 0.5
    n_lanes = 5
    lane_length = 30
    slowdown = 0.2

    n_agents = int(density * n_lanes * lane_length)
    agents = Agents(n_agents=n_agents, n_lanes=n_lanes, lane_length=lane_length, bias_strength=0)
    model = SwitchingNaSchModel(agents=agents, slowdown=slowdown)

    for _ in range(100):
        model.step()
    
    visualizer = BasicStepwiseVisualizer(agents, model, trail_length=2)

    plt.show()