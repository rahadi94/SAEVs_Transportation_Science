# SAEVs Transportation Science

This repository provides a simulation and reinforcement learning environment for **Shared Autonomous Electric Vehicles (SAEVs)**.  
It supports modeling of fleet operations, trip matching, charging strategies, and vehicle scheduling to study the efficiency and economics of SAEVs in urban transportation systems.

---

## 🚀 Features

- **Fleet Simulation (`Fleet_sim/`)**
  - Models vehicles, trips, zones, charging stations, and parking.
  - Implements trip matching and fleet management logic.
  - Supports multiple reinforcement learning (RL) approaches for dispatching and charging:
    - Deep Q-Networks (`DQN`, `sub_DQN`, `single_DQN`)
    - Soft Actor-Critic (`single_SAC`)

- **Reinforcement Learning Environment**
  - `rl_environment.py` provides an interface for training RL algorithms.
  - Includes reward mechanisms for profitability, service rate, and energy efficiency.

- **Visualization Tools (`Visualization/`)**
  - Scripts for analyzing trips, charging, state of charge (SoC), and performance.
  - Generates plots and PDF reports (e.g., `profit_RL.pdf`, `SoC.pdf`, `CSs.pdf`).

- **Main Entry**
  - `main.py` — runs the simulation and RL training pipeline.

---

## 📂 Repository Structure
