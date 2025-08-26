from Fleet_sim.charging_station import ChargingStation
from Fleet_sim.location import Location
from Fleet_sim.model import Model, lg
import simpy
import random
from Fleet_sim.parking import Parking
from Fleet_sim.read import zones
from Fleet_sim.vehicle import Vehicle
from datetime import datetime
import pandas as pd



# sub penalty for old_location
episode = 0
while episode <= 50:

    start_time = datetime.now()
    # try:
    lg.error(f'iteration:{episode}')
    print(f'iteration:{episode}')
    env = simpy.Environment()


    def generate_location():
        return Location(random.uniform(52.40, 52.60), random.uniform(13.25, 13.55))


    vehicles_data = []
    for i in range(200):
        vehicle_data = dict(id=i, env=env, initial_location=generate_location(), capacity=50,
                            charge_state=random.randint(70, 75), mode='idle')
        vehicles_data.append(vehicle_data)

    vehicles = list()

    for data in vehicles_data:
        vehicle = Vehicle(
            data['id'],
            data['env'],
            data['initial_location'],
            data['capacity'],
            data['charge_state'],
            data['mode']
        )
        vehicles.append(vehicle)

    CSs_data = []
    CSs_optimum = [z for z in zones if z.id in [3, 11, 15, 18, 27, 36, 40, 42, 45, 52, 59, 65, 73, 74, 86, 88]]
    c = [4 * 2, 3 * 2, 6 * 2, 5 * 2, 4 * 2, 3 * 2, 3 * 2, 4 * 2, 4 * 2, 5 * 2, 4 * 2, 5 * 2, 3 * 2, 5 * 2, 8 * 2,
         4 * 2]
    p = [11 / 60, 11 / 60, 11 / 60, 50 / 60, 11 / 60, 11 / 60, 50 / 60, 11 / 60, 50 / 60, 11 / 60,
         50 / 60, 11 / 60, 50 / 60, 11 / 60, 11 / 60, 11 / 60]
    CSs_zones = []
    for s in range(len(c)):
        CSs_zones.append(dict(base=CSs_optimum[s], Number_of_chargers=c[s], power=p[s]))

    for zone in CSs_zones:
        CS_data = dict(id=zone['base'].id, env=env, location=zone['base'].centre, power=zone['power'],
                       Number_of_chargers=zone['Number_of_chargers'])
        CSs_data.append(CS_data)

    '''CSs_data = []
    for i in zones:
        CS_data = dict(id=i.id, env=env, location=i.centre, power=11 / 60, Number_of_chargers=500)
        CSs_data.append(CS_data)'''

    # Initialize Charging Stations
    charging_stations = list()

    for data in CSs_data:
        charging_station = (ChargingStation(
            data['id'],
            data['env'],
            data['location'],
            data['power'],
            data['Number_of_chargers']

        ))
        charging_stations.append(charging_station)

    PKs_data = []
    for i in range(100):
        PK_data = dict(id=i, env=env, location=generate_location(), Number_of_parkings=40)
        PKs_data.append(PK_data)

    parkings = list()

    for data in PKs_data:
        parking = (Parking(
            data['id'],
            data['env'],
            data['location'],
            data['Number_of_parkings']

        ))
        parkings.append(parking)

    # Run simulation
    sim = Model(env, vehicles=vehicles, charging_stations=charging_stations, zones=zones, parkings=parkings,
                simulation_time=1440 * 7, episode=episode)
    for zone in zones:
        env.process(sim.trip_generation(zone=zone))
    env.process(sim.run())

    for vehicle in vehicles:
        env.process(sim.run_vehicle(vehicle))

    env.process(sim.hourly_charging_relocating())
    # env.process(sim.charging_interruption())

    for vehicle in vehicles:
        env.process(sim.obs_Ve(vehicle=vehicle))

    for charging_station in charging_stations:
        env.process(sim.obs_CS(charging_station=charging_station))

    """for parking in parkings:
        env.process(sim.obs_PK(parking))"""

    env.process(sim.missed_trip())
    env.process(sim.reward_monitor())
    env.process(sim.reset_vehicles_operations())

    env.run(until=sim.simulation_time)
    total_profit = 0
    total_reward = 0
    for vehicle in vehicles:
        total_reward += vehicle.final_reward
        total_profit += vehicle.profit
    lg.error(f'total_profit={total_profit}, total_reward={total_reward}')
    sim.save_results(episode)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    lg.error('Duration: {}'.format(end_time - start_time))
    episode += 1
'''except:
    episode = episode'''
