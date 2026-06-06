import random
import math

FIELD_OF_VIEW = 10  # How many cells ahead the car can "see"
BASE_COST = 5.0


def calculate_lane_cost(target_lane, car, pos, highway):
    """Calculate the cost of switching to the target lane based on traffic conditions."""
    cost = BASE_COST
    
    for i in range(1, FIELD_OF_VIEW + 1):
        comparison_car = highway.get_car(target_lane, pos + i)
        if comparison_car: 
            cost += 1.0  # check amopunt of cars ahead in the target lane
            cost += (car.velocity - comparison_car.velocity)  # check the speed difference

    return cost

def get_valid_adjacent_lanes(current_lane, pos, highway):
    """Check adjacent lanes for valid lane change options."""
    valid_options = []

    # Check Left Lane (lane - 1)
    if highway.check_move_valid(current_lane - 1, pos):
        valid_options.append(current_lane - 1)

    # Check Right Lane (lane + 1)
    if highway.check_move_valid(current_lane + 1, pos):
        valid_options.append(current_lane + 1)

    return valid_options

def decide_next_lane(current_lane, car,pos, highway, threshold, beta=1.0):
    """Decide whether to switch lanes based on a game-theoretic cost evaluation."""

    # if car can just move forward do that, no need to evaluate lane change
    if highway.check_move_valid(current_lane, pos + 1):
        return current_lane

    valid_options = get_valid_adjacent_lanes(current_lane, pos, highway)
    if not valid_options: return current_lane

    # 2 & 3. Cost Evaluation and Selection
    best_target_lane = None
    lowest_cost = float('inf')

    for target_lane in valid_options:
        cost = calculate_lane_cost(target_lane, car, pos, highway) 
        
        if cost < lowest_cost:
            lowest_cost = cost
            best_target_lane = target_lane

    # 4. The Stochastic Choice. The higher the cost difference, the more likely they are to switch.
    probability_to_switch = 1 / (1 + math.exp(beta * (lowest_cost - threshold)))
    if random.random() < probability_to_switch: return best_target_lane # Switch
    else: return current_lane # They decide to stay

