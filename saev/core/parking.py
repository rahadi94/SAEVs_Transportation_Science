import simpy

class Parking:

    def __init__(self, id, env, location, Number_of_parkings):
        self.env = env
        self.capacity = simpy.PreemptiveResource(self.env, capacity=Number_of_parkings)
        self.id = id
        self.location = location
        self.queue = []


