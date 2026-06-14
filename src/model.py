import numpy as np

class Model:
    def __init__(self, num_lanes, lane_length, num_cars, slowdown_probability):
        # settings
        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.num_cars = num_cars
        self.slowdown_probability = slowdown_probability
        self.max_speed = 5

        # agents
        self.positions = np.zeros(num_cars, dtype=int)
        self.lanes = np.zeros(num_cars, dtype=int)
        self.velocities = np.zeros(num_cars, dtype=int)
        self.thresholds = np.full(num_cars, 5.0)

        # environment
        # self.grid = np.full((num_lanes, lane_length), None, dtype=object)

        # agent initialization
        indices = np.random.choice(self.num_lanes * self.lane_length, size=num_cars, replace=False)

        self.lanes = indices // self.lane_length
        self.positions = indices % self.lane_length


    def step(self):
        # standard Nagel-Schreckenberg update
        gaps = self.get_gaps()

        self.velocities = np.minimum(self.velocities + 1, self.max_speed)
        self.velocities = np.minimum(self.velocities, gaps)

        slowdown_mask = (self.velocities > 0) & (np.random.random(self.num_cars) < self.slowdown_probability)
        self.velocities[slowdown_mask] = np.maximum(self.velocities[slowdown_mask] - 1, 0)
        
        self.positions = (self.positions + self.velocities) % self.lane_length

        # potential lane-switch
        


    def get_gaps(self):
        gaps = np.full(self.num_cars, self.lane_length - 1, dtype=int)

        for lane in range(self.num_lanes):
            lane_mask = self.lanes == lane
            lane_indices = np.where(lane_mask)[0]
            n_cars_lane = len(lane_indices)

            if n_cars_lane == 0: 
                continue
            elif n_cars_lane == 1: 
                gaps[lane_indices[0]] = self.lane_length - 1
            else:
                lane_positions = self.positions[lane_indices]
                sort_idx = np.argsort(lane_positions)
                sorted_positions = lane_positions[sort_idx]
                sorted_car_indices = lane_indices[sort_idx]

                next_positions = np.roll(sorted_positions, -1)
                forward_distance = (next_positions - sorted_positions) % self.lane_length
                lane_gaps = forward_distance - 1

                gaps[sorted_car_indices] = lane_gaps
        
        return gaps