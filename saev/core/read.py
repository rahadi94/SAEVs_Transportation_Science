import os
import pandas as pd

from saev.core.Zone import Zone

DATA_DIR = os.getenv('SAEV_DATA_DIR', os.path.join(os.getcwd(), 'data', 'input'))
print(DATA_DIR)


demand_table = pd.read_csv(os.path.join(DATA_DIR, 'demand_table.csv'))
OD_table = pd.read_csv(os.path.join(DATA_DIR, 'origin_destination.csv'))

z = 0
zones = list()
for hex in demand_table['h3_hexagon_id_start'].values:
    z += 1
    demand = (demand_table[demand_table['h3_hexagon_id_start'] == hex]).drop('h3_hexagon_id_start', axis=1)
    destination = (OD_table[OD_table['h3_hexagon_id_start'] == hex]
                   .drop('h3_hexagon_id_start', axis=1)).sort_values(by=z - 1, axis=1).T.reset_index()
    zone = Zone(z, hex, demand, destination)
    zones.append(zone)
charging_threshold = [40, 45, 50, 55, 52, 50, 48, 45, 45, 42, 40, 40, 40, 40, 40, 38, 35, 32, 30, 30, 27, 30, 32, 35]

charging_cost = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
            , 23, 23, 23, 23, 23, 23, 8, 8]


