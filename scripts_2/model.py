import random
import numpy as np

from car import Car
from strategy import choose_action, target_lane, update_learning


class TrafficModel:
    def __init__(self, W, L, n_cars, v_max, alpha, beta, gamma, eta, seed=None):
        self.W = W
        self.L = L
        self.n_cars = n_cars
        self.v_max = v_max

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.cars = []
        self.time = 0

        self.history = []

        self.initialize_cars()

    def initialize_cars(self):
        """
        Initialize the cars on the grid.
        """
        occupied = set()

        for car_id in range(self.n_cars):
            while True:
                x = random.randrange(self.L)
                lane = random.randrange(self.W)

                if (x, lane) not in occupied:
                    occupied.add((x, lane))
                    break

            v = random.randint(1, self.v_max)

            car = Car(car_id, x, lane, v, self.v_max)
            self.cars.append(car)

    def grid(self):
        """
        Initialize the grid representation of the model, 
        with -1 for empty cells and car IDs for occupied cells.
        """
        grid = np.full((self.W, self.L), -1)

        for car in self.cars:
            grid[car.lane, car.x] = car.id

        return grid

    def gap_ahead(self, x, lane):
        """
        Return the number of empty cells ahead of 
        position x in a specified lane.
        """
        # Find all occupied positions in the lane.
        occupied_positions = [
            car.x for car in self.cars
            if car.lane == lane and car.x != x
        ]
        # Case: no cars ahead if no occupied positions.
        if not occupied_positions:
            return self.L - 1
        
        # Calculate distances to occupied positions
        distances = [
            (pos - x) % self.L
            for pos in occupied_positions
        ]
        # Find the nearest occupied position ahead.
        nearest_distance = min(d for d in distances if d > 0)

        return nearest_distance - 1

    def local_risk(self, car, target_lane):
        """
        Simpele risicomaat:
        hoe druk is het vlak voor de auto in de doelbaan?
        """
        risk = 0

        for other in self.cars:
            if other.id == car.id:
                continue

            if other.lane == target_lane:
                distance = (other.x - car.x) % self.L

                if 0 < distance <= car.sight:
                    risk += 1

        return risk

    def step(self):
        """
        Eén iteratie:
        1. kies actie L/S/R
        2. voer lane changes uit
        3. beweeg vooruit
        4. update snelheid
        5. leer van verschil E(U) en R(U)
        """

        # fase 1: alle auto's kiezen actie
        for car in self.cars:
            choose_action(car, self)

        # fase 2: lane changes simultaan verwerken
        proposed_positions = {}

        for car in self.cars:
            new_lane = target_lane(car, car.action, self.W)
            target = (car.x, new_lane)

            if target not in proposed_positions:
                proposed_positions[target] = []

            proposed_positions[target].append(car)

        for target, cars in proposed_positions.items():
            if len(cars) == 1:
                car = cars[0]
                car.lane = target[1]
            else:
                # conflict: random winnaar
                winner = random.choice(cars)
                winner.lane = target[1]

                for loser in cars:
                    if loser.id != winner.id:
                        loser.action = "S"

        # fase 3: vooruit bewegen
        for car in self.cars:
            gap = self.gap_ahead(car.x, car.lane)

            moved_distance = min(car.v, gap)

            old_v = car.v

            car.x = (car.x + moved_distance) % self.L

            if moved_distance == old_v:
                car.v = min(car.v + 1, car.v_max)
            else:
                car.v = moved_distance

            lane_change_cost = 0 if car.action == "S" else 1

            car.realized_utility = (
                self.alpha * moved_distance
                - self.beta * lane_change_cost
            )

        # fase 4: leren
        for car in self.cars:
            update_learning(car, self.eta)

        self.record_history()
        self.time += 1

    def record_history(self):
        avg_speed = np.mean([car.v for car in self.cars])
        avg_expected = np.mean([car.expected_utility for car in self.cars])
        avg_realized = np.mean([car.realized_utility for car in self.cars])
        lane_changes = sum(car.action != "S" for car in self.cars)

        self.history.append({
            "time": self.time,
            "avg_speed": avg_speed,
            "avg_expected_utility": avg_expected,
            "avg_realized_utility": avg_realized,
            "lane_changes": lane_changes
        })