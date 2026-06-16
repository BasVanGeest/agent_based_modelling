# agent_based_modelling

## Installation / Running
Clone the repository, and navigate to the new main folder. Then run the following to set up the environment and install all required packages:
```bash
uv sync
```

Running a file is then as simple as:
```bash
uv run src/main.py
```

## Profiling
Though this is a bit overkill, it can be nice to find crucial performance bottlenecks. You can use snakeviz to get an interactive flamegraph by running:
```bash
python -m cProfile -o output.prof src/main.py
snakeviz output.prof
```