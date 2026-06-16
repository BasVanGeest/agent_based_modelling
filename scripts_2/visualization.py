import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def grid_for_plot(model):
    grid = np.zeros((model.W, model.L))

    for car in model.cars:
        grid[car.lane, car.x] = car.v + 1

    return grid


def animate_model(model, steps=300, interval=100):
    fig, ax = plt.subplots(figsize=(12, 3))

    grid = grid_for_plot(model)

    image = ax.imshow(
        grid,
        aspect="auto",
        interpolation="none"
    )

    ax.set_title("Nagel-Schreckenberg Game-Theory Traffic Model")
    ax.set_xlabel("Position")
    ax.set_ylabel("Lane")

    def update(frame):
        model.step()
        grid = grid_for_plot(model)

        image.set_data(grid)

        ax.set_title(
            f"t = {model.time} | "
            f"avg speed = {model.history[-1]['avg_speed']:.2f}"
        )

        return [image]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval,
        blit=False
    )

    plt.show()