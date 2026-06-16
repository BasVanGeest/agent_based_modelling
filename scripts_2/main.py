from agent_based_modelling.scripts_2.config import *
from agent_based_modelling.scripts_2.model import TrafficModel
from agent_based_modelling.scripts_2.visualization import animate_model


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