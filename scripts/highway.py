import numpy as np
import random

RANDOM_SEED = 42

class Highway:
    def __init__(self, num_lanes=3, lane_length=1000):
        self.random_seed = RANDOM_SEED
        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.grid = np.full((num_lanes, lane_length), None, dtype=object)
    
    def move_car(self, t, car, new_lane, new_position):
        # check for the boundary conditions, looping cars around if they are at the end of the lane
        if new_position >= self.lane_length:
            new_position = new_position % self.lane_length
            car.lap(t)

        if not self.check_cell_valid(new_lane, new_position): return False
        if self.check_cell_occupied(new_lane, new_position): return False

        # move the car
        self.grid[car.lane, car.position] = None
        car.lane = new_lane
        car.position = new_position
        self.grid[car.lane, car.position] = car
        return True 

    def place_car(self, car, lane, position):
        """Place a car at the specified cell if empty. Returns True on success."""
        if not self.check_cell_valid(lane, position): return False
        if self.check_cell_occupied(lane, position): return False

        car.lane = lane
        car.position = position
        self.grid[lane, position] = car
        return True

    def populate_cars(self, cars, max_attempts_multiplier=100):
        """Place a list of `cars` randomly on the highway.

        Tries up to `len(cars) * max_attempts_multiplier` attempts and returns
        the number of cars successfully placed.
        """
        random.seed(self.random_seed)

        placed = 0
        attempts = 0
        max_attempts = len(cars) * max_attempts_multiplier
        while placed < len(cars) and attempts < max_attempts:
            lane = random.randrange(self.num_lanes)
            pos = random.randrange(self.lane_length)
            car = cars[placed]
            if self.place_car(car, lane, pos):
                placed += 1
            attempts += 1

        return placed

    def get_gap(self, car):
        """Return the number of empty cells directly ahead of `car` in its lane.
        """
        lane = car.lane
        start = car.position
        for offset in range(1, self.lane_length):
            pos = (start + offset) % self.lane_length
            if self.grid[lane, pos] is not None:
                return offset - 1
        return self.lane_length - 1

    def move_forward(self, t, car):
        """Move car forward by its current velocity. Assumes velocity has
        been computed according to NS rules and that the destination cell
        is free.
        """
        new_position = (car.position + car.velocity)
        if new_position >= self.lane_length:
            new_position = new_position % self.lane_length
            car.lap(t)

        if new_position == car.position: return True
        if self.check_cell_occupied(car.lane, new_position): return False

        # Move
        self.grid[car.lane, car.position] = None
        car.position = new_position
        self.grid[car.lane, car.position] = car
        return True

    def check_cell_valid(self, lane, position):
        """Check if the cell is within the bounds of the highway."""
        return 0 <= lane < self.num_lanes and 0 <= position < self.lane_length
        
    def check_cell_occupied(self, lane, position):
        """Check if the cell is occupied by a car."""
        return self.grid[lane, position] is not None
    
    def check_move_valid(self, lane, position):
        """Check if a move to the specified lane and position is valid (within bounds and not occupied)."""
        return self.check_cell_valid(lane, position) and not self.check_cell_occupied(lane, position)
    
    def get_car(self, lane, position):
        """Return the car object at the specified lane and position, or None if empty."""
        if self.check_cell_valid(lane, position):
            return self.grid[lane, position]
        return None