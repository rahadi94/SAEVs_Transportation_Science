from saev.core.charging_station import ChargingStation
from saev.core.location import Location
from saev.core.vehicle import Vehicle
import simpy
import random

env = simpy.Environment()


def generate_location():
    return Location(random.normalvariate(5, 0.5), random.normalvariate(5, 0.5))


vehicle_data = [

    dict(id=1, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=2, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=3, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=4, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=5, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=6, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=7, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=8, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=9, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=10, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=11, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=12, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=13, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=14, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=15, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=16, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle'),
    dict(id=17, env=env, initial_location=generate_location(), capacity=150, charge_state=100, mode='idle')

]

vehicles = list()

for data in vehicle_data:
    vehicle = (Vehicle(
        data['id'],
        data['env'],
        data['initial_location'],
        data['capacity'],
        data['charge_state'],
        data['mode']
    ))
    vehicles.append(vehicle)

# Initializing charging stations

CS_data = [

    dict(id=1, env=env, location=Location(5.5, 5.5), power=5),
    dict(id=2, env=env, location=Location(4.8, 4.8), power=5),
    dict(id=3, env=env, location=Location(5.0, 5.0), power=5),
    dict(id=4, env=env, location=Location(5.2, 4.9), power=5),
    dict(id=5, env=env, location=Location(5.1, 4.9), power=5),

]

charging_stations = list()

for data in CS_data:
    charging_station = (ChargingStation(
        data['id'],
        data['env'],
        data['location'],
        data['power'],

    ))
    charging_stations.append(charging_station)
