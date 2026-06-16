ACTIONS = ["L", "S", "R"]


def target_lane(car, action, W):
    if action == "L":
        return car.lane - 1
    if action == "R":
        return car.lane + 1
    return car.lane


def valid_action(car, action, W):
    new_lane = target_lane(car, action, W)
    return 0 <= new_lane < W


def expected_utility(car, action, model):
    if not valid_action(car, action, model.W):
        return -9999

    lane = target_lane(car, action, model.W)

    gap = model.gap_ahead(car.x, lane)
    expected_distance = min(car.v, gap)

    lane_change_cost = 0 if action == "S" else 1
    risk = model.local_risk(car, lane)

    utility = (
        model.alpha * expected_distance
        - model.beta * lane_change_cost
        - model.gamma * risk
        + car.q_values[action]
    )

    return utility


def choose_action(car, model):
    utilities = {
        action: expected_utility(car, action, model)
        for action in ACTIONS
    }

    best_action = max(utilities, key=utilities.get)

    car.expected_utility = utilities[best_action]
    car.action = best_action

    return best_action


def update_learning(car, eta):
    error = car.realized_utility - car.expected_utility

    car.q_values[car.action] += eta * error