
from agent import Car   
from highway import Highway
from strategy import calculate_lane_cost, get_valid_adjacent_lanes, decide_next_lane
import random

TOTAL_STEPS = 1000
N_CARS = 100
N_lanes = 3
lane_length = 1000
p = 0.01  # Random slowdown probability

highway = Highway(N_lanes, lane_length)
cars = [Car(p) for _ in range(N_CARS)]

# Populate cars using Highway helper
placed = highway.populate_cars(cars)
if placed < N_CARS:
    raise RuntimeError('Could not place all cars on the highway; reduce N_CARS or increase lane_length')



for tick in range(TOTAL_STEPS):
    # Phase 1: Game Theory
    for car in cars:
        next_lane = decide_next_lane(car.lane, car, car.position, highway, threshold=5.0)
        highway.move_car(car, next_lane, car.position)  # Attempt lane change
            
    # Phase 2: Forward Physics
    # 2a: Compute velocity for all cars (parallel update)
    for car in cars:
        gap = highway.get_gap(car)
        car.calculate_next_velocity(p, gap)  # Nagel-Schreckenberg rules

    # 2b: Move all cars to their new positions (atomic update)
    moves = []
    for car in cars:
        new_pos = (car.position + car.velocity) % lane_length
        moves.append((car, car.lane, car.position, new_pos))

    # Clear old positions
    for _, lane, pos, _ in moves:
        highway.grid[lane, pos] = None

    # Place cars at new positions
    for car, lane, old_pos, new_pos in moves:
        # If destination occupied (should be rare if NS rules applied), try to keep car at old_pos
        if highway.check_cell_occupied(lane, new_pos):
            # revert to previous spot
            highway.place_car(car, lane, old_pos)
            car.position = old_pos
        else:
            highway.place_car(car, lane, new_pos)
            car.position = new_pos
        
    # Phase 3: Evolution
    # TODO