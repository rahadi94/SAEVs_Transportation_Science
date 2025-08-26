import random
from saev.core.location import generate_random
from Fleet_sim.read import zones


class Trip:

    def __init__(self, env, id, zone):
        self.env = env
        self.id = id
        self.zone = zone

        # We generate origin and destination of trips based on a origin-destination pattern of our case study
        time = self.env.now
        self.origin = generate_random(zone.hexagon)
        r = random.uniform(0, 1)
        for i in range(1, 88):
            if zone.destination.iloc[i - 1, 1] < r <= zone.destination.iloc[i, 1]:
                zone_destination_hex = zone.destination.iloc[i, 0]

        zone_destination = [z for z in zones if z.hexagon == zone_destination_hex][0]
        self.destination = generate_random(zone_destination.hexagon)
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                demand = zone.demand[str(i)].values/1

        # We generate time-varying trips (i.e. trips are being generated exponentially, in which
        # arrival-rate is a gaussian function of time during the day)
        if demand == 0:
            arrival_rate = 1440
        else:
            arrival_rate = 60 / demand
        self.interarrival = random.expovariate(1 / arrival_rate)
        self.start_time = None
        distance_duration = self.origin.distance(self.destination)
        self.distance = distance_duration[0]
        self.duration = distance_duration[1]
        self.end_time = self.interarrival + self.duration

        self.mode = 'unassigned'
        self.miss_prob = 0
        self.info = dict()
        self.info['id'] = self.id
        self.info['origin'] = [self.origin.lat, self.origin.long]
        self.info['destination'] = [self.destination.lat, self.destination.long]
        self.info['origin_zone'] = zone.id
        self.info['mode'] = self.mode
        self.info['distance'] = self.distance
        self.info['destination_zone'] = zone_destination.id
        self.info['assigned_time'] = None
        self.info['arrival_time'] = None
        self.info['pickup_time'] = None
        self.info['waiting_time'] = None
        self.cancellation = env.event()
        self. revenue = max(((self.distance * 1.11 + self.duration * 0.31) + 2), 5)
    """
    Allowed modes are:
    unassigned - no vehicle is assigned to it
    assigned - a vehicle is assigned and sent
    in vehicle - it is being served
    finished - it is finished   """


