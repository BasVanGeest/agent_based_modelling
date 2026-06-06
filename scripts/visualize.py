import random
import time
from agent import Car
from highway import Highway
from strategy import decide_next_lane


def render(highway):
    rows = []
    for lane in range(highway.num_lanes):
        row = []
        for pos in range(highway.lane_length):
            car = highway.get_car(lane, pos)
            if car is None:
                row.append('_')
            else:
                # show velocity as a single digit
                v = car.velocity
                ch = str(min(9, int(v)))
                row.append(ch)
        rows.append(''.join(row))
    return '\n'.join(rows)


def run_visual(lanes=3, length=100, n_cars=30, p=0.01, steps=200, auto=False, delay=0.5):
    highway = Highway(lanes, length)
    cars = [Car(p) for _ in range(n_cars)]

    placed = highway.populate_cars(cars)
    if placed < n_cars:
        raise RuntimeError('Could not place all cars')

    print('Controls: Enter step, q+Enter to quit, a+Enter to autoplay')
    autoplay = auto

    for t in range(steps):
        # print state
        print(f'=== Step {t} ===')
        print(render(highway))
        avg_v = sum(c.velocity for c in cars) / len(cars)
        print(f'Avg speed: {avg_v:.2f}')

        if not autoplay:
            cmd = input().strip().lower()
            if cmd == 'q':
                break
            if cmd == 'a':
                autoplay = True

        if autoplay:
            time.sleep(delay)

        # Phase 1: lane-change decisions
        for car in cars:
            next_lane = decide_next_lane(car.lane, car, car.position, highway, threshold=5.0)
            highway.move_car(car, next_lane, car.position)

        # Phase 2a: compute velocities
        for car in cars:
            gap = highway.get_gap(car)
            car.calculate_next_velocity(p, gap)

        # Phase 2b: atomic forward movement
        moves = []
        for car in cars:
            new_pos = (car.position + car.velocity) % length
            moves.append((car, car.lane, car.position, new_pos))

        for _, lane, pos, _ in moves:
            highway.grid[lane, pos] = None

        for car, lane, old_pos, new_pos in moves:
            if highway.check_cell_occupied(lane, new_pos):
                highway.place_car(car, lane, old_pos)
                car.position = old_pos
            else:
                highway.place_car(car, lane, new_pos)
                car.position = new_pos


if __name__ == '__main__':
    run_visual(lanes=3, length=100, n_cars=50, p=0.1, steps=500)
