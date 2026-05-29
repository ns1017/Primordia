# Primordia

An interactive early biological life simulator using neuroevolution in Python.

Agents are controlled by small neural networks. Their weights and physical traits evolve across generations through mutation and selection based on survival and reproduction success.

## Features

- Real-time 2D visualization with smooth pan and zoom camera controls
- Compact evolvable genomes covering body traits (size, speed, metabolism, sensor morphology) and personality biases (exploration, hunger drive, memory use, diet preference)
- Tiny neural controllers (8 hidden units) with optional recurrent memory for within-lifetime adaptation
- Ray-based sensing plus internal state and three classes of transient signals (food, danger, mate)
- Dynamic environment including plant and meat food, hard and pushable rocks with habitat effects, and toxic zones
- Predation, scavenging, and dietary specialization emerge from an evolvable diet preference gene
- Inter-run elitism: diverse high performers can be carried forward when resetting a world
- Live god-mode controls for mutation rates and simulation speed
- Headless parallel experiment runner supporting named presets and multi-preset comparisons
- Detailed agent inspector showing live brain activations, input saliency, memory state, and lineage
- Automatic JSON logging of runs plus optional matplotlib summary plots

## Installation

### Prerequisites

- Python 3.11 or newer
- Works on Windows, Linux, and macOS

### Setup

```bash
git clone https://github.com/<your-username>/primordia.git
cd primordia

# Create a virtual environment (once)
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

This pulls in the required dependencies (pyglet and numpy).

A convenience launcher for Windows is included: double-click `run.bat`.

## Running the Interactive Simulation

```bash
primordia
```

Or:

```bash
python -m primordia
```

## Controls

| Key / Action          | Effect                                      |
|-----------------------|---------------------------------------------|
| Space                 | Pause or resume simulation                  |
| R                     | Reset world (carries forward diverse elites)|
| Mouse wheel           | Zoom in or out                              |
| Left mouse drag       | Pan the camera                              |
| Left click on entity  | Select agent, food, rock, or toxic zone     |
| + or =                | Increase simulation speed                   |
| - or _                | Decrease simulation speed                   |
| [ or ]                | Decrease / increase physical mutation rate  |
| ; or '                | Decrease / increase brain mutation rate     |
| S                     | Print summary and save JSON log             |
| P                     | Generate matplotlib plots from last log     |
| V                     | Toggle detailed stats in sidebar            |
| H                     | Toggle on-screen controls help              |
| Esc                   | Quit                                        |

Matplotlib is optional and only needed for the P key plots. Install it with `pip install matplotlib` if desired.

## Experiment Mode

Run multiple independent worlds in parallel for statistical evaluation.

List presets:

```bash
primordia --list-presets
```

Run with a preset:

```bash
primordia --experiment --preset signal-rich --worlds 8 --ticks 6000
```

Compare multiple presets:

```bash
primordia --experiment --compare baseline dense signal-rich pressure --worlds 6 --ticks 5000
```

Important flags:

- `--worlds` : number of parallel simulation worlds
- `--ticks` : number of simulation steps per world
- `--workers` : process count (defaults to CPU count minus one)
- `--seed` : base seed for full reproducibility
- `--preset` : apply one of the predefined environment profiles
- `--compare` : run several presets sequentially and summarize

All experiment output is written to the `runs/` directory as JSON reports.

## Project Structure

```
primordia/
├── src/primordia/
│   ├── app.py           # Pyglet window, rendering, input handling, CLI
│   ├── simulation.py    # Core loop, agents, physics, reproduction, logging
│   ├── genome.py        # Genes and Genome (heritable traits + brain)
│   ├── brain.py         # Small neural network with recurrent memory support
│   ├── config.py        # Centralized tunable parameters
│   ├── experiments.py   # Preset definitions and parallel experiment runner
│   └── __main__.py
├── pyproject.toml
├── run.bat              # Windows one-click launcher
├── runs/                # Generated logs and plots (gitignored)
└── README.md
```

## Technical Overview

Each agent carries a complete genome containing:

- Physical multipliers for size, speed, metabolism, sensory range, lifespan, and sensor layout
- Four evolvable personality traits
- The complete weights of a small neural network (18 inputs, 8 hidden, 2 action outputs + recurrent memory)

Reproduction occurs when an agent gathers sufficient energy. Offspring inherit a mutated copy of the parent's genome. Selection emerges naturally from the physics of energy, health, toxicity, competition, and predation.

A spatial grid accelerates neighbor queries for sensing and interactions. Recurrent memory cells allow agents to maintain short-term internal state across ticks.

## Philosophy

- All core capabilities (sensing, decision making, energy management, reproduction) must be evolvable.
- The world should feel alive and interesting even with modest population sizes.
- Visual emergence and exploratory fun are prioritized over strict biological fidelity.
- The code should remain small, readable, and easy to modify.

## Status

The interactive visual shell, genome system, neural controllers, rich sensors, signaling, terrain, dietary specialization, elitism, and batch experiment tooling are all implemented and functional.

---

Primordia is intended for curiosity-driven exploration of neuroevolution, artificial life, and the origins of complex behavior from simple rules.
