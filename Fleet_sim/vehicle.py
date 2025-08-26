from Fleet_sim.location import find_zone
import numpy as np
from math import ceil
from Fleet_sim.log import lg
from Fleet_sim.read import zones, charging_cost


class Vehicle:
    speed = 0.5  # km/min
    parking_cost = 5

    def __init__(self, id, env, initial_location, capacity, charge_state, mode):
        self.env = env
        self.info = dict()
        self.info['SOC'] = []
        self.info['location'] = []
        self.info['position'] = []
        self.info['mode'] = []
        self.location = initial_location
        self.id = id
        self.mode = mode
        self.position = find_zone(self.location, zones)
        """Allowed modes are:
             active - car is currently driving a passenger from pickup to destination
             locked - car is currently going to pickup location to pick up customer
             idle - car is currently idle and waiting for request
             relocating - car is moving to a different comb
             charging - car is currently charging
             en_route_to_charge - car is on its way to a charging station"""
        self.battery_capacity = capacity
        self.charge_state = charge_state
        self.count_request_accepted = 0
        self.rental_time = 0.0
        self.fuel_consumption = 0.20  # in kWh/km
        self.t_start_charging = None
        self.t_start_relocating = None
        self.t_arriving_CS = None
        self.costs = dict()
        self.costs['charging'] = 0.0
        self.costs['parking'] = 0.0
        self.charge_consumption_dropoff = None
        self.time_to_pickup = None
        self.charge_consumption_pickup = None
        self.distance_to_CS = None
        self.time_to_CS = None
        self.charging_threshold = None
        self.charging_demand  = None
        self.charge_duration = None
        self.time_to_relocate = None
        self.charge_consumption_relocate = None
        self.distance_to_parking = None
        self.time_to_parking = None
        self.discharge_duration = None
        self.discharging_threshold = None
        self.decision_time = 0
        self.parking_stop = env.event()
        self.circling_stop = env.event()
        self.charging_interruption = env.event()
        self.discharging_interruption = env.event()
        self.charging_start = env.event()
        self.charging_end = env.event()
        self.trip_end = env.event()
        self.relocating_end = env.event()
        self.queue_interruption = env.event()
        self.discharging_end = env.event()
        self.trip_cancellation = env.event()
        self.relocating_stop = env.event()
        self.reward = dict(charging=0, queue=0, distance=0, revenue=0, parking=0, missed=0, discharging=0,
                           interruption=0, penalty=0, sub_missed=0)
        self.total_rewards = dict(state=[], action=[], reward=[])
        self.profit = 0
        self.final_reward = 0
        self.old_action = 0
        self.old_sub_action = 0
        self.old_location = None
        self.old_state = None
        self.old_sub_state = None
        self.charging_count = 0
        self.charging_station = None
        self.estimated_SOC = 0
        self.operations_check = 0


    def SOC_consumption(self, distance):
        return float(distance * self.fuel_consumption * 100.0 / self.battery_capacity)

    def send(self, trip):
        self.mode = 'locked'
        distance_duration = self.location.distance(trip.origin)
        distance_to_pickup = distance_duration[0]
        distance_to_dropoff = trip.distance
        self.time_to_pickup = distance_duration[1]
        self.charge_consumption_pickup = self.SOC_consumption(distance_to_pickup)
        self.charge_consumption_dropoff = self.SOC_consumption(distance_to_dropoff)
        trip.info['assigned_time'] = self.env.now
        self.rental_time = trip.duration
        lg.info(f'Vehicle {self.id} is sent to the request {trip.id}')

    def pick_up(self, trip):
        self.mode = 'active'
        lg.info(f'Vehicle {self.id} picks up the user {trip.id} at {self.env.now}')
        self.charge_state -= self.charge_consumption_pickup
        trip.info['pickup_time'] = self.env.now
        trip.info['waiting_time'] = trip.info['pickup_time'] - trip.info['arrival_time']
        self.location = trip.origin

    def drop_off(self, trip):
        self.mode = 'idle'
        self.charge_state -= self.charge_consumption_dropoff
        self.location = trip.destination
        self.position = find_zone(self.location, zones)
        lg.info(f'Vehicle {self.id} drops off the user {trip.id} at {self.env.now}')

    def send_charge(self, charging_station):
        self.mode = 'ertc'
        lg.info(f'Charging state of vehicle {self.id} is {self.charge_state}')
        if self.action == 0:
            lg.info(f'Vehicle {self.id} is sent to the charging station'
                    f'{charging_station.id} at {self.env.now}')

        distance_duration = self.location.distance(charging_station.location)
        self.distance_to_CS = distance_duration[0]
        self.reward['distance'] += distance_duration[0]
        k = ceil((self.env.now - self.decision_time)/15)
        self.reward['distance'] = self.reward['distance'] * 0.9 ** k
        if isinstance(self.reward['distance'], np.ndarray):
            self.reward['distance'] = self.reward['distance'][0]
        self.time_to_CS = distance_duration[1]
        self.position = find_zone(self.location, zones)

    def charging(self, charging_station):
        self.mode = 'charging'
        charge_consumption_to_charging = self.SOC_consumption(self.distance_to_CS)
        self.charge_state -= charge_consumption_to_charging
        time = self.env.now
        if time % 1440 < 0.25 * 1440:
            self.charging_threshold = 100
        elif time % 1440 < 0.50 * 1440:
            self.charging_threshold = 100
        elif time % 1440 < 0.75 * 1440:
            self.charging_threshold = 100
        else:
            self.charging_threshold = 100
        self.charge_duration = (((self.charging_threshold - self.charge_state) * self.battery_capacity / 100)
                                / charging_station.power)
        self.location = charging_station.location
        self.position = find_zone(self.location, zones)
        lg.info(f'Vehicle {self.id} enters the station at {self.env.now}')

    def finish_charging(self, charging_station):
        self.mode = 'idle'
        for j in range(0, 24):
            if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                h = j
        self.reward['charging'] += (self.charging_threshold - self.charge_state) / 100 * 50 * charging_cost[h]/100
        self.profit -= (self.charging_threshold - self.charge_state) / 100 * 50 * charging_cost[h]/100
        k = ceil((self.env.now - self.decision_time)/15)
        self.reward['charging'] = self.reward['charging'] * 0.9 ** k
        if isinstance(self.reward['charging'], np.ndarray):
            self.reward['charging'] = self.reward['charging'][0]
        self.costs['charging'] += (self.charging_threshold - self.charge_state) / 100 * 50 * charging_cost[h]/100
        self.charge_state += (charging_station.power * self.charge_duration) / (self.battery_capacity / 100)
        if self.charge_state > 110:
            print("WHAAAAAAAAAT")
        lg.info(f'Finished charging, Charging state of vehicle {self.id} is {self.charge_state} at {self.env.now}')

    def discharging(self, charging_station):
        self.mode = 'discharging'
        self.charge_state -= self.SOC_consumption(self.distance_to_CS)
        self.discharging_threshold = 50
        discharge_rate = charging_station.power
        self.discharge_duration = (((self.charge_state - self.discharging_threshold) * self.battery_capacity / 100)
                                / discharge_rate)
        self.location = charging_station.location
        self.position = find_zone(self.location, zones)
        lg.info(f'Vehicle {self.id} enters the station at {self.env.now}')

    def finish_discharging(self, charging_station):
        self.mode = 'idle'
        for j in range(0, 24):
            if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                h = j
        self.reward['discharging'] += (self.charge_state - self.discharging_threshold) / 100 * 50 * charging_cost[h]/100
        self.profit += (self.charge_state - self.discharging_threshold) / 100 * 50 * charging_cost[h]/100
        k = ceil((self.env.now - self.decision_time)/15)
        self.reward['discharging'] = self.reward['discharging'] * 0.9 ** k
        if isinstance(self.reward['charging'], np.ndarray):
            self.reward['discharging'] = self.reward['discharging'][0]
        self.charge_state -= (charging_station.power * self.discharge_duration) / (self.battery_capacity / 100)
        lg.info(f'Finished discharging, Charging state of vehicle {self.id} is {self.charge_state} at {self.env.now}')

    def relocate(self, target_zone):
        distance_duration = self.location.distance(target_zone.centre)
        distance_to_target = distance_duration[0]
        self.location = target_zone.centre
        self.time_to_relocate = distance_duration[1]
        self.charge_consumption_relocate = self.SOC_consumption(distance_to_target)
        lg.info(f'Vehicle {self.id} is relocated to the zone {target_zone.id}')
        self.mode = 'relocating'

    def finish_relocating(self, target_zone):
        self.charge_state -= self.charge_consumption_relocate
        self.location = target_zone.centre
        self.position = find_zone(self.location, zones)
        self.mode = 'idle'

    def send_parking(self, parking):
        self.mode = 'ertp'
        if self.env.now != 0:
            lg.info(f'Vehicle {self.id} is sent to the parking {parking.id} at {self.env.now}')
        distance_duration = self.location.distance(parking.location)
        self.distance_to_parking = distance_duration[0]
        self.time_to_parking = distance_duration[1]
        charge_consumption_to_parking = self.SOC_consumption(self.distance_to_parking)
        self.charge_state -= charge_consumption_to_parking

    def parking(self, parking):
        self.mode = 'parking'
        self.location = parking.location
        self.position = find_zone(self.location, zones)
        if self.env.now >= 5:
            lg.info(f'Vehicle {self.id} starts parking at {self.env.now}')
