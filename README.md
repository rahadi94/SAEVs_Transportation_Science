# SAEVs Transportation Science

This repository provides a simulation and reinforcement learning environment for **Shared Autonomous Electric Vehicles (SAEVs)**.  
It supports modeling of fleet operations, trip matching, charging strategies, and vehicle scheduling to study the efficiency and economics of SAEVs in urban transportation systems.

📄 This repository accompanies the following paper:  
[*Fleet Operations of Shared Autonomous Electric Vehicles: Impacts of Charging Infrastructure and Charging Strategy*](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2022.1187), published in *Transportation Science*.

---
---

## 🚀 Features

- **Core Simulation (`saev/core/`)**
  - Models vehicles, trips, zones, charging stations, parking, matching, fleet, and model.
  - Central logging in `saev/core/log.py`.

- **Reinforcement Learning (`saev/rl/`)**
  - `saev/rl/environment.py` provides the gym environment for training.
  - Agents and configs exposed via `saev/rl/agents/`.

- **Visualization Tools (`Visualization/`)**
  - Scripts for analyzing trips, charging, state of charge (SoC), and performance.
  - Generates plots and PDF reports (e.g., `profit_RL.pdf`, `SoC.pdf`, `CSs.pdf`).

- **Main Entry**
  - `main.py` — runs the simulation and RL training pipeline.

---

## 📂 Repository Structure

```
saev/
  core/
    charging_station.py
    location.py
    parking.py
    trip.py
    vehicle.py
    Zone.py
    Matching.py
    read.py
    log.py
    model_impl.py
    model.py
  rl/
    environment.py
    agents/
      __init__.py
main.py
Visualization/
requirements.txt
data/
  input/
  output/
```

Set input data directory (optional):

```bash
export SAEV_DATA_DIR=$(pwd)/data/input
```
