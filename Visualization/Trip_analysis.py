import pandas as pd

from Visualization.analysis import clean

PATH_RULE_BASED = '/home/rahadi/PycharmProjects/ssh/rule-based/results/trips0.csv'
PATH_central = '/home/rahadi/PycharmProjects/ssh/central_charging/results/trips0.csv'
PATH_DQN = '/home/rahadi/PycharmProjects/ssh/DQN/results/trips20.csv'
df_trip_rule_based = pd.read_csv(PATH_RULE_BASED).drop('Unnamed: 0', axis=1)
df_trip_central = pd.read_csv(PATH_central).drop('Unnamed: 0', axis=1)
df_trip_DQN = pd.read_csv(PATH_DQN).drop('Unnamed: 0', axis=1)

SL = pd.read_csv('../results/service_level_RL.csv')
P = pd.read_csv('../results/profit_RL.csv')
# for i in ['new', 'sub_action_in_state', 'DQN', 'full_action', 'Q_learning', 'benchmark', 'upper_bound']:
for i in ['benchmark']:
    # if i == 'new':
    #     x = 'Full_HDQN'
    # if i == 'sub_action_in_state':
    #     x = 'Limited_HDQN'
    # if i == 'DQN':
    #     x = 'Limited_DQN'
    # if i == 'full_action':
    #     x = 'Full_DQN'
    # if i == 'Q_learning':
    #     x = 'Q_learning'
    if i == 'benchmark':
        x = 'Benchmark'
    # if i == 'upper_bound':
    #     x = 'Upperbound'

    for j in range(19):
        path = f'/home/rahadi/PycharmProjects/ssh/central_charging/results/trips{j}.csv'
        df = pd.read_csv(path).drop('Unnamed: 0', axis=1)
        sl = 1-(df.loc[df['mode'] == 'missed'].count()[0] / df.count()[0])
        cost = df.loc[df['mode'] == 'finished']['distance'].sum() * 0.02
        P.loc[(P['Episode'] == j), x] -= cost
        SL.loc[(SL['Episode'] == j), x] = sl
        SL.to_csv('../results/service_level_RL.csv')
        P.to_csv('../results/profit_RL.csv')

def describe(trip):
    trip['waiting_time'] = trip.apply(lambda row: clean(row['waiting_time']), axis=1)
    trip['waiting_time'] = trip['waiting_time'].astype('float64')
    print(trip.describe())
    print(trip.loc[trip['mode'] == 'missed'].count()[0] / trip.count()[0])


describe(df_trip_rule_based)
describe(df_trip_central)
describe(df_trip_DQN)

