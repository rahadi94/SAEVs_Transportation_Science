import simpy
import pandas as pd

class ChargingStation:

    def __init__(self, id, env, location, power, Number_of_chargers):
        self.env = env
        self.capacity = Number_of_chargers
        self.plugs = simpy.PriorityResource(self.env, capacity=Number_of_chargers)
        self.id = id
        self.location = location
        self.power = power  # kwh/min
        self.queue = []
        # self.position = self.location.find_zone(zones)
