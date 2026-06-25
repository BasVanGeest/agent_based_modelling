from agent import Agents

import numpy as np
import numpy.typing as npt

class BaseNaSchModel:
    """
    This class implements the Nagel-Schrekenberg model. Being associated with traffic flow, it consists of a set of agents, moving along a discrete line. Each step, agents try to accelerate, limited by the number of empty spaces ahead of each agent. Here the model is extended to run multiple lanes at the same time, without any mechanism for interaction between lanes.

    Attributes
    --------------
    agents : Agents
        list of agents in use
    slowdown: float
        percentage chance of a given agent slowing down
    step_count: int
        counts the number of finished iterations
    """


    def __init__(
        self, 
        agents: Agents, 
        slowdown: float = 0.05
    ) -> None:
        """
        Initialises the model, by moving input data into member variables and resetting the step count.
        
        Parameters
        --------------
        agents : Agents
            list of agents in use
        slowdown: float
            percentage chance of a given agent slowing down
        """

        self.agents = agents
        self.slowdown = slowdown
        self.step_count = 0


    def step(self) -> None:
        """
        Runs a single iteration of the model. Applies the NaSch rules for all agents, then updates their velocity.
        """

        self.agents.update_velocities(self.slowdown)
        self.agents.positions = (self.agents.positions + self.agents.velocities) % self.agents.lane_length
        self.step_count += 1



class SwitchingNaSchModel(BaseNaSchModel):
    """
    This class implements an extension onto the base Nagel-Schrekenberg model, where interaction between lanes is defined. On top of the standard NaSch behaviour, agents now make a probabilistic guess based on their velocity history and local information what lane to switch to. Agent histories are updated each step.

    Attributes
    --------------
    agents : Agents
        list of agents in use
    slowdown: float
        percentage chance of a given agent slowing down
    step_count: int
        counts the number of finished iterations
    applied_choices: npt.NDArray[np.int_]
        stores the applied choices of the last step
    """


    def __init__(
        self, 
        agents: Agents, 
        slowdown: float = 0.05
    ) -> None:
        """
        Initialises the model, by moving input data into member variables and resetting the step count.
        
        Parameters
        --------------
        agents : Agents
            list of agents in use
        slowdown: float
            percentage chance of a given agent slowing down
        """
        super().__init__(agents, slowdown)


    def step(self) -> None:
        """
        Runs a single iteration of the switching model. Generates chosen lanes for each agent, tries to apply them, applies the standard NaSch step and updates the agent histories.
        """
        # step 1: compute gaps and let each vehicle make a decision on if / where to switch
        choices = self.agents.choose_actions()

        # step 2: apply lane changing, in random order, checking for collisions
        self.applied_choices = self.apply_choices(choices)

        # step 3: NaSch velocity update
        super().step()

        # step 4: Update agents histories
        self.agents.update_histories(self.applied_choices, self.agents.velocities)


    def apply_choices(self, choices: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """
        Tries to apply a given set of choices for each agent. It goes through all agents in random order, then tries to apply the lane-switch if possible. If not, it stays on its current lane. The list of applied choices (going left, staying or going right) is returned.

        Parameters
        --------------
        choices : npt.NDArray[np.int_]
            array of choices, of identical length to the number of agents

        Returns
        --------------
        npt.NDArray[np.int_]
            array of choices that were applied, of identical length to the number of agents
        """
        n_agents = self.agents.n_agents
        applied_choices = np.ones(n_agents, dtype=int)

        # create a 2D array, storing for each grid cell if it's occupied or not
        grid_occupancy = np.zeros(shape=(self.agents.n_lanes, self.agents.lane_length), dtype=bool)
        grid_occupancy[self.agents.lanes, self.agents.positions] = True

        agent_order = np.random.permutation(n_agents)

        for agent_index in agent_order:
            choice = choices[agent_index]
            lane = self.agents.lanes[agent_index]
            position = self.agents.positions[agent_index]

            if choice == 1:
                continue
            
            target_lane = lane - 1 + choice

            if not grid_occupancy[target_lane, position]:
                self.agents.lanes[agent_index] = target_lane
                applied_choices[agent_index] = choice

                # update occupancy grid with newly moved agent
                grid_occupancy[lane, position] = False
                grid_occupancy[target_lane, position] = True

        return applied_choices