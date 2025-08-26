import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import set_matplotlib_formats

from Visualization.analysis import clean

set_matplotlib_formats('retina')


def charge_duration(SOC_send, SOC_end, start):
    duration = (SOC_end - SOC_send) * 50 / (100 * 11 / 60)
    end = start + duration
    return [duration, end]


def charging(df):
    df['time_start'] = df.apply(lambda row: clean(row['time_start']), axis=1)
    df['time_start'] = df['time_start'].astype('float')
    df['CS_location'] = df.apply(lambda row: clean(row['CS_location']), axis=1)
    df['duration'] = df.apply(lambda row: charge_duration(row['SOC_send'], row['SOC_end'], row['time_start'])[0],
                              axis=1)
    df['time_end'] = df.apply(lambda row: charge_duration(row['SOC_send'], row['SOC_end'], row['time_start'])[1],
                              axis=1)

    df['time_start'] = dt.datetime(2020, 1, 1) + pd.TimedeltaIndex(df['time_start'], unit='m')
    df['time_end'] = dt.datetime(2020, 1, 1) + pd.TimedeltaIndex(df['time_end'], unit='m')

    df["day_start"] = df["time_start"].dt.strftime('%d').astype('int64')
    df["day_end"] = df["time_end"].dt.strftime('%d').astype('int64')
    df["hour_start"] = (df['day_start'] - 1) * 24 + df["time_start"].dt.strftime('%H').astype('int64')
    df["hour_end"] = (df['day_end'] - 1) * 24 + df["time_end"].dt.strftime('%H').astype('int64')
    df["hour"] = df["time_start"].dt.strftime('%H').astype('int64')
    df = df.groupby(['hour', 'day_start']).agg({'vehicle_id': 'count'})
    return df


def plot_charge(dg1, dg2, dg3, dg4):
    dg1 = dg1.reset_index()
    dg1 = dg1.groupby(['hour']).agg({'vehicle_id': 'mean'})
    dg2 = dg2.reset_index()
    dg2 = dg2.groupby(['hour']).agg({'vehicle_id': 'mean'})
    dg3 = dg3.reset_index()
    dg3 = dg3.groupby(['hour']).agg({'vehicle_id': 'mean'})
    dg4 = dg4.reset_index()
    dg4 = dg4.groupby(['hour']).agg({'vehicle_id': 'mean'})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
    ax.plot(dg1.index, dg1['vehicle_id'], label='Central', linestyle='--', marker='.')
    ax.plot(dg2.index, dg2['vehicle_id'], label='DQN', linestyle='--', marker='.')
    ax.plot(dg3.index, dg3['vehicle_id'], label='sub', linestyle='--', marker='.')
    ax.plot(dg3.index, dg4['vehicle_id'], label='full', linestyle='--', marker='.')
    ax.set_xlabel("hour")
    ax.set_ylabel("Charging demand")
    ax.set_title("")
    ax.legend()
    plt.show()


PATH_sub = '/home/rahadi/PycharmProjects/ssh/sub_action_in_state/results/charging10.csv'
PATH_full = '/home/rahadi/PycharmProjects/ssh/new/results/charging20.csv'
PATH_central = '/home/rahadi/PycharmProjects/ssh/central_charging/results/charging0.csv'
PATH_DQN = '/home/rahadi/PycharmProjects/ssh/DQN/results/charging10.csv'
df_charge_sub = pd.read_csv(PATH_sub).drop('Unnamed: 0', axis=1)
df_charge_full = pd.read_csv(PATH_full).drop('Unnamed: 0', axis=1)
df_charge_central = pd.read_csv(PATH_central).drop('Unnamed: 0', axis=1)
df_charge_DQN = pd.read_csv(PATH_DQN).drop('Unnamed: 0', axis=1)

dg_charge_sub = charging(df_charge_sub)
dg_charge_full = charging(df_charge_full)
dg_charge_central = charging(df_charge_central)
dg_charge_DQN = charging(df_charge_DQN)
plot_charge(dg_charge_central, dg_charge_DQN, dg_charge_sub, dg_charge_full)
