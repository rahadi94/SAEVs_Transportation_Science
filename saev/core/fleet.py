from saev.core.location import find_zone
import numpy as np
from math import ceil
from saev.core.log import lg
from saev.core.read import zones, charging_cost


class Fleet:

    def __init__(self, env, vehicles, charging_stations, RL_agent):
        self.env = env
        self.vehicles = vehicles
        self.charging_stations = charging_stations
        self.RL_agent = RL_agent
        self.state = None
        self.old_state = None
        self.old_action = None
        self.reward = 0
        self.charging_count = 0
        self.vehicles_to_decide = []
        self.episode = None
        self.reward = dict(charging=0, queue=0, distance=0, revenue=0, parking=0, missed=0, discharging=0,
                           interruption=0, penalty=0, sub_missed=0)

    def SOC_consumption(self, distance):
        return float(distance * self.fuel_consumption * 100.0 / self.battery_capacity)


