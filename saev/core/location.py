from geopy.distance import geodesic
import random
from shapely.geometry import Point, shape
from h3 import h3


def find_zone(loc, zones):
    distances_to_centers = [loc.distance_1(zone.centre) for zone in zones]
    position = [x for x in zones
                if x.centre.distance_1(loc) == min(distances_to_centers)][0]
    return position


class Location:

    def __init__(self, lat, long):
        self.lat = lat
        self.long = long

    def distance_1(self, loc):
        origin = [self.lat, self.long]
        destination = [loc.lat, loc.long]
        return geodesic(origin, destination).kilometers * 1.5

    def distance(self, loc):
        origin = [self.lat, self.long]
        destination = [loc.lat, loc.long]
        dis = geodesic(origin, destination).kilometers
        dur = dis / 0.5
        return [dis * 1.5, dur * 1.5 + 2]


def generate_random(hex):
    polygon = shape(
        {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(hex, geo_json=True)], "properties": ""})
    minx, miny, maxx, maxy = polygon.bounds
    c = True
    while c:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))  # x=long, y=lat
        if polygon.contains(pnt):
            c = False

        return Location(pnt.y, pnt.x)


def closest_facility(facilities, vehicle):
    distances = [vehicle.location.distance_1(f.location) for f in facilities]
    facility = [x for x in facilities
                if x.location.distance_1(vehicle.location) == min(distances)][0]
    return facility


