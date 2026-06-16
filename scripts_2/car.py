class Car:
    def __init__(self, car_id, x, lane, v, v_max):
        self.id = car_id
        self.x = x
        
        # initialization 
        self.lane = lane
        self.v = v

        # constant 
        self.v_max = v_max

        self.action = "S"
        self.expected_utility = 0
        self.realized_utility = 0

        self.q_values = {
            "L": 0.0,
            "S": 0.0,
            "R": 0.0
        }

    @property
    def sight(self):
        return max(1, self.v)