from geopy.distance import geodesic
import random
from shapely.geometry import Point, shape
from h3 import h3

# import osmnx as ox
# import networkx as nx
# ox.config(use_cache=True, log_console=True)
# from saev.core.read1 import G
# network = nx.Graph(G)

'''def find_zone(loc, zones):
    hexagon = h3.geo_to_h3(loc.long, loc.lat, 7)
    position = [x for x in zones
                if x.hexagon == f'{hexagon}'][0]
    return position'''

'''def find_zone(loc, zones):
    p = Point(loc.long, loc.lat)
    position = [x for x in zones if x.polygon.contains(p)][0]
    return position'''


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

    '''def distance(self, loc):
        try:
            orig = ox.get_nearest_node(G, (self.lat, self.long))
            dest = ox.get_nearest_node(G, (loc.lat, loc.long))
            route = ox.shortest_path(G, orig, dest, weight='travel_time')
            edge_lengths = ox.utils_graph.get_route_edge_attributes(G, route, 'length')
            edge_time = ox.utils_graph.get_route_edge_attributes(G, route, 'travel_time')
            dis = sum(edge_lengths) / 1000
            dur = sum(edge_time) / 60
            return [dis, dur]
        except:
            origin = [self.lat, self.long]
            destination = [loc.lat, loc.long]
            dis = geodesic(origin, destination).kilometers
            dur = dis/0.5
            return [dis, dur]'''

    '''def distance(self, loc):
        wp_1_long = self.long
        wp_1_lat = self.lat
        wp_2_long = loc.long
        wp_2_lat = loc.lat
        url = f'http://dev.virtualearth.net/REST/V1/Routes/Driving?wp.0={wp_1_lat},{wp_1_long}&wp.1={wp_2_lat},{wp_2_long}&optimize=timeWithTraffic&ra=excludeItinerary&key=Am9VAgnNzXCUOzAS_WCBQHsWrZQOG53xa8dPp7bZFiYWn-m2CJxCFo2yXsNNBDYa'
        r = requests.get(url)
        d = r.json()['resourceSets'][0]['resources'][0]['routeLegs'][0]['travelDistance']
        return d'''


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

    """import googlemaps
    API_key = 'AIzaSyCxGGUs - xbyFZFsiDDSKNP7QIjGr - Is1DA'
    gmaps = googlemaps.Client(key=API_key)
    result = gmaps.distance_matrix(origins, destination, mode='walking')["rows"][0]["elements"][0]["distance"]["value"]"""
