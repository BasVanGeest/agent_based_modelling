# agent_based_modelling
This repository hosts the implementation for a proposed multi-lane learning switching extension of the Nagel-Schrekenberg traffic model. This is part of the Agent-based Modelling course at the University of Amsterdam, 2025-2026. 

## Project structure
The 'src' folder contains all the python code. The model implementation, both baseline and extended, is split across the 'agent.py' and 'model.py' files. The remaining scripts are for various ways of analyzing the system. The 'figures' folder stores plots from the development phase, split across two learning methods.

## Installation / Running
Clone the repository, and navigate to the new project folder. The project dependencies are managed by uv, installation and further information of which can be found [here](https://docs.astral.sh/uv/). Run the following to set up the environment and install all required dependencies:
```bash
uv sync
```

A stepwise visualizer is included, showing vehicles and their choices. It can be configured by modifying the parameters set at the bottom of the 'live_visualizer.py' file. Then start it using:
```bash
uv run src/live_visualizer.py
```

The sensitivity analysis and histogram plots for the extended model can be reproduced by running:
``` bash
uv run src/sensitivity_analysis.py
```

Figures across the reduced phase-space of density and slowdown probability can be made with:
``` bash
uv run src/macroscopic_figures.py
```

## Profiling
Though this is a bit overkill, profiling tools can be nice to find crucial performance bottlenecks and have been used for this project to drastically improve runtime. You can use snakeviz to get an interactive flamegraph by running:
```bash
python -m cProfile -o output.prof src/macroscopic_figures.py
snakeviz output.prof
```