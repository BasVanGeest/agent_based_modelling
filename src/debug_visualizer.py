import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from agent import Agents
from model import Model

class BasicStepwiseVisualizer:
    def __init__(self, agents, model, trail_length):
        self.agents = agents
        self.model = model
        self.trail_length = trail_length

        # give a unique color for each vehicle, so they (hopefully) are better to track between steps
        cmap = plt.get_cmap('hsv')
        self.agent_colors = [cmap(i / self.agents.n_agents) for i in range(self.agents.n_agents)]

        # store the last few positions, so we can plot the last few steps made by each (it shows a trail behind each vehicle, making it clearer if switches are used or not and how)
        # this should probably be a numpy 2D array, but I can't be arsed
        self.history = [[] for _ in range(agents.n_agents)]

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.subplots_adjust(bottom=0.2) # leaving some space for the step button

        ax_button = plt.axes([0.8, 0.05, 0.1, 0.05])
        self.button = Button(ax_button, 'step forward')
        self.button.on_clicked(self.next_step)

        # Define grid 
        # TODO: add ticks only at integers; should make it clearer what the lanes are
        self.ax.set_xlim(-1, agents.lane_length)
        self.ax.set_ylim(-0.5, agents.n_lanes - 0.5)
        self.ax.set_xlabel('Position along lane')
        self.ax.set_ylabel('Lane number')
        self.ax.set_title('Traffic Simulation (Step 0)')

        for i in range(agents.n_agents):
            self.history[i].append((agents.positions[i], agents.lanes[i]))

        self.redraw()


    def redraw(self):
        self.ax.clear()

        # rebuild grid + limits
        self.ax.set_xlim(-1, self.agents.lane_length)
        self.ax.set_ylim(-0.5, self.agents.n_lanes - 0.5)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_xlabel('Position')
        self.ax.set_ylabel('Lane')
        self.ax.set_title('Traffic Simulation')

        # per agent, draw its trail with its unique color
        for i in range(self.agents.n_agents):
            agent_history = self.history[i]
            if len(agent_history) > 1:
                positions, lanes = zip(*agent_history)
                self.ax.plot(positions, lanes, color=self.agent_colors[i], alpha=0.6, linestyle='-')

        # draw cars as filled squares, each with its unique color
        self.ax.scatter(self.agents.positions, self.agents.lanes, s=120, marker='s', color=self.agent_colors, zorder=2) # draw on top of the trails

        # for testing, draw the agent.choice_weights for each vehicle above them. This is to debug learning or to get an idea of if weights have converged
        for i in range(self.agents.n_agents):
            weights = self.agents.choice_weights[i]
            text = f'[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]'
            
            text = self.ax.text(self.agents.positions[i], self.agents.lanes[i] + 0.1, text, ha='center', va='bottom', fontsize=6)
        self.fig.canvas.draw_idle()


    def next_step(self, event):
        self.model.step()

        # update agent trails
        for i in range(self.agents.n_agents):
            self.history[i].append((self.agents.positions[i], self.agents.lanes[i]))
            if len(self.history[i]) > self.trail_length:
                self.history[i].pop(0)

        self.redraw()

if __name__ == '__main__':
    agents = Agents(n_agents=20, n_lanes=5, lane_length=30)
    model = Model(agents=agents, slowdown=0.3)
    visualizer = BasicStepwiseVisualizer(agents, model, trail_length=3)
    plt.show()