from config import *
from model import TrafficModel
from visualization import *

def main():
    model = TrafficModel(
        W=W,
        L=L,
        n_cars=N_CARS,
        v_max=V_MAX,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        eta=ETA,
        seed=RANDOM_SEED
    )

    animate_model(model, steps=TICKS, interval=100)


if __name__ == "__main__":
    main()