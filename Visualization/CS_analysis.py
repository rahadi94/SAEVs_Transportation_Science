import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import set_matplotlib_formats
from saev.core.log import lg

set_matplotlib_formats('retina')
# BASE_PATH = '/home/rahadi/PycharmProjects/ssh/'
BASE_PATH = '/home/rahadi/Projects/SAEV/TS_revision/results'
PATH_flat_tariff = f'{BASE_PATH}/CSs14.csv'
PATH_high_price = f'{BASE_PATH}/CSs12.csv'
PATH_Q_learning = f'{BASE_PATH}/CSs18.csv'
PATH_central = f'{BASE_PATH}/CSs11.csv'
PATH_No_FC_central = f'{BASE_PATH}/CSs0.csv'
PATH_No_FC = f'{BASE_PATH}/CSs5.csv'
PATH_Num_CS_central = f'{BASE_PATH}/CSs0.csv'
PATH_Num_CS = f'{BASE_PATH}/CSs18.csv'
PATH_DQN = f'{BASE_PATH}/CSs20.csv'
PATH_sub = f'{BASE_PATH}/CSs20.csv'
PATH_full = f'{BASE_PATH}/CSs20.csv'
PATH_sub_250 = f'{BASE_PATH}/CSs14.csv'
PATH_sub_300 = f'{BASE_PATH}/CSs14.csv'
PATH_sub_150 = f'{BASE_PATH}/CSs20.csv'
PATH_sub_75 = f'{BASE_PATH}/CSs21.csv'
PATH_sub_100 = f'{BASE_PATH}/CSs20.csv'
PATH_single_RL = f'{BASE_PATH}/CSs20.csv'
# df_CS_DQN = pd.read_csv(PATH_DQN).drop('Unnamed: 0', axis=1)
# df_CS_central = pd.read_csv(PATH_central).drop('Unnamed: 0', axis=1)
# df_CS_Q_learning = pd.read_csv(PATH_Q_learning).drop('Unnamed: 0', axis=1)
df_CS_sub = pd.read_csv(PATH_sub).drop('Unnamed: 0', axis=1)
df_CS_single_RL = pd.read_csv(PATH_single_RL).drop('Unnamed: 0', axis=1)
# df_CS_sub_250 = pd.read_csv(PATH_sub_250).drop('Unnamed: 0', axis=1)
# df_CS_sub_150 = pd.read_csv(PATH_sub_150).drop('Unnamed: 0', axis=1)
# df_CS_sub_300 = pd.read_csv(PATH_sub_300).drop('Unnamed: 0', axis=1)
# df_CS_sub_75 = pd.read_csv(PATH_sub_75).drop('Unnamed: 0', axis=1)
# df_CS_sub_100 = pd.read_csv(PATH_sub_100).drop('Unnamed: 0', axis=1)
# df_CS_No_FC_central = pd.read_csv(PATH_No_FC_central).drop('Unnamed: 0', axis=1)
# df_CS_No_FC = pd.read_csv(PATH_No_FC).drop('Unnamed: 0', axis=1)
# df_CS_sub_flat_tariff = pd.read_csv(PATH_flat_tariff).drop('Unnamed: 0', axis=1)
# df_CS_sub_high_price = pd.read_csv(PATH_high_price).drop('Unnamed: 0', axis=1)
# df_CS_sub_high_price = pd.read_csv(PATH_high_price).drop('Unnamed: 0', axis=1)
# df_CS_Num_CS_central = pd.read_csv(PATH_Num_CS_central).drop('Unnamed: 0', axis=1)
# df_CS_Num_CS = pd.read_csv(PATH_Num_CS).drop('Unnamed: 0', axis=1)

sizes = [4 * 2, 3 * 2, 6 * 2, 5 * 2, 4 * 2, 3 * 2, 3 * 2, 4 * 2, 4 * 2, 5 * 2, 4 * 2, 5 * 2, 3 * 2, 5 * 2, 8 * 2,
         4 * 2]


# sizes = [x/2 for x in sizes]


def clean(x):
    if type(x) != float:
        x = x.replace('[', '').replace(']', '')
    return x


def set_style():
    # This sets reasonable defaults for font size for a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    # sns.set_color_codes("pastel")
    sns.set_palette("mako", n_colors=2)
    # Make the background white, and specify the font family
    sns.set_style("ticks", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def charging_type(x):
    list_fast_chargers = [3, 6, 8, 10, 12]
    if x in list_fast_chargers:
        return 'Fast'
    else:
        return 'Standard'


def data_CS(CS, output='utilization'):
    for i in CS.columns.values:
        CS[i] = CS.apply(lambda row: clean(row[i]), axis=1)
        counts = CS
    for i in counts.columns.values:
        counts[i] = counts.apply(lambda row: row[i].split(',')[0], axis=1)
    counts = counts.astype('float64')
    utilization = CS
    charging_load = CS
    charging_rate = CS
    '''counts = counts.transpose().reset_index().rename({'index': 'time'}, axis=1).astype('float64')
    counts['time'] = dt.datetime(2020, 1, 1) + pd.TimedeltaIndex(counts['time'], unit='m')
    counts["day"] = counts["time"].dt.strftime('%d').astype('int64')
    counts["hour"] = counts["time"].dt.strftime('%H').astype('int64')
    dg = counts.groupby(['hour', 'day']).agg('mean')
    dgg = pd.DataFrame(dg.stack()).reset_index()
    dgg.columns = ['hour', 'day', 'CS', 'count']'''
    if output == 'utilization':
        for i in range(16):
            utilization.iloc[i] = counts.iloc[i] / sizes[i] * 100
        utilization = utilization.transpose().reset_index().rename({'index': 'time'}, axis=1).astype('float64')
        utilization['time'] = dt.datetime(2020, 1, 1) + pd.TimedeltaIndex(utilization['time'], unit='m')
        utilization["day"] = utilization["time"].dt.strftime('%d').astype('int64')
        utilization["hour"] = utilization["time"].dt.strftime('%H').astype('int64')
        dg = utilization.groupby(['hour', 'day']).agg('mean')
        dgg = pd.DataFrame(dg.stack()).reset_index()
        dgg.columns = ['hour', 'day', 'CS', 'count']
        dgg['charging_speed'] = 'Standard'
        dgg['charging_speed'] = dgg['CS'].apply(lambda x: charging_type(x))
    if output == 'charging_rate':
        charging_rate = charging_rate.transpose().reset_index().rename({'index': 'time'}, axis=1).astype('float64')
        charging_rate['time'] = dt.datetime(2020, 1, 1) + pd.TimedeltaIndex(charging_rate['time'], unit='m')
        charging_rate["day"] = charging_rate["time"].dt.strftime('%d').astype('int64')
        charging_rate["hour"] = charging_rate["time"].dt.strftime('%H').astype('int64')
        dg = charging_rate.groupby(['hour', 'day']).agg('mean')
        dgg = pd.DataFrame(dg.stack()).reset_index()
        dgg.columns = ['hour', 'day', 'CS', 'count']
        dgg['charging_speed'] = 'Standard'
        dgg['charging_speed'] = dgg['CS'].apply(lambda x: charging_type(x))
        dg = dgg.groupby(['hour', 'day', 'charging_speed']).agg('mean')
        dgg = dg.reset_index()
        dgg.columns = ['hour', 'day', 'charging_speed', 'CS', 'count']
        dgg['charging_ratio'] = 0
        for i in range(0,335,2):
            dgg.loc[i, 'charging_ratio'] = dgg.loc[i, 'count'] / (dgg.loc[i, 'count'] + dgg.loc[i+1, 'count'])
        dgg = dgg[dgg.loc[:,'charging_speed']=='Fast']


    elif output == 'load':
        for i in range(16):
            if i in [3, 6, 8, 10, 12]:
                charging_load.loc[i] = counts.iloc[i] * 55
            else:
                charging_load.loc[i] = counts.iloc[i] * 11
        charging_load = charging_load.transpose().reset_index().rename({'index': 'time'}, axis=1).astype('float64')
        charging_load['time'] = dt.datetime(2020, 1, 1) + pd.TimedeltaIndex(charging_load['time'], unit='m')
        charging_load["day"] = charging_load["time"].dt.strftime('%d').astype('int64')
        charging_load["hour"] = charging_load["time"].dt.strftime('%H').astype('int64')
        dg = charging_load.groupby(['hour', 'day']).agg('mean')
        dgg = pd.DataFrame(dg.stack()).reset_index()
        dgg.columns = ['hour', 'day', 'CS', 'count']
        dgg['charging_speed'] = 'Standard'
        dgg['charging_speed'] = dgg['CS'].apply(lambda x: charging_type(x))
        dgg['fast_percentage'] = dgg['CS'].apply(lambda x: charging_type(x))
    return dgg


def plot_CS(dg1, dg2, dg3, dg4):
    set_style()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), constrained_layout=True)
    sns.barplot(data=dg1, x='CS', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[0][0], capsize=0.1)
    sns.barplot(data=dg2, x='CS', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[0][1], capsize=0.1)
    sns.barplot(data=dg3, x='CS', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[1][0], capsize=0.1)
    sns.barplot(data=dg4, x='CS', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[1][1], capsize=0.1)
    ax[0][0].set_ylim(0, 100)
    ax[0][0].set_xlabel("Charging station ID", fontsize=12)
    ax[0][0].set_ylabel("Utilization(%)", fontsize=12)
    ax[0][0].set_title("150 vehicles", fontsize=12)
    ax[0][0].tick_params(axis='both', labelsize=12)
    ax[0][1].set_ylim(0, 100)
    ax[0][1].set_xlabel("Charging station ID", fontsize=12)
    ax[0][1].set_ylabel("Utilization(%)", fontsize=12)
    ax[0][1].set_title("200 vehicles", fontsize=12)
    ax[0][1].tick_params(axis='both', labelsize=12)
    ax[1][0].set_ylim(0, 100)
    ax[1][0].set_xlabel("Charging station ID", fontsize=12)
    ax[1][0].set_ylabel("Utilization(%)", fontsize=12)
    ax[1][0].set_title("250 vehicles", fontsize=12)
    ax[1][0].tick_params(axis='both', labelsize=12)
    ax[1][1].set_ylim(0, 100)
    ax[1][1].set_xlabel("Charging station ID", fontsize=12)
    ax[1][1].set_ylabel("Utilization(%)", fontsize=12)
    ax[1][1].set_title("300 vehicles", fontsize=12)
    ax[1][1].tick_params(axis='both', labelsize=12)
    plt.savefig("CSs.pdf")
    plt.show()


# def plot_CS(dg1, dg2, dg3):
#     set_style()
#     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4), constrained_layout=True)
#     sns.barplot(data=dg1, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[0], capsize=0.1)
#     sns.barplot(data=dg2, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[1], capsize=0.1)
#     sns.barplot(data=dg3, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[2], capsize=0.1)
#     # sns.barplot(data=dg4, x='CS', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[1][1], capsize=0.1)
#     ax[0].set_ylim(0, 100)
#     ax[0].set_xlabel("Time of the day", fontsize=12)
#     ax[0].set_ylabel("Utilization(%)", fontsize=12)
#     ax[0].set_title("50 kWh", fontsize=12)
#     ax[0].tick_params(axis='both', labelsize=11)
#     ax[1].set_ylim(0, 100)
#     ax[1].set_xlabel("Time of the day", fontsize=12)
#     ax[1].set_ylabel("Utilization(%)", fontsize=12)
#     ax[1].set_title("75 kWh", fontsize=12)
#     ax[1].tick_params(axis='both', labelsize=11)
#     ax[2].set_ylim(0, 100)
#     ax[2].set_xlabel("Time of the day", fontsize=12)
#     ax[2].set_ylabel("Utilization(%)", fontsize=12)
#     ax[2].set_title("100 kWh", fontsize=12)
#     ax[2].tick_params(axis='both', labelsize=11)
#     plt.savefig("CSs.pdf")
#     plt.show()


# def plot_CS(dg1, dg2): set_style() fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4),
# constrained_layout=True) sns.barplot(data=dg1, x='CS', y='count', hue='charging_speed', palette=sns.color_palette(
# 'mako', n_colors=2),ci = "sd", capsize = 0.1, ax=ax[0],dodge=False) sns.barplot(data=dg2, x='CS', y='count',
# hue='charging_speed', palette=sns.color_palette('mako', n_colors=2),ci = "sd", capsize = 0.1, ax=ax[1],dodge=False)
# sns.barplot(data=dg4, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[1][1]) ax[
# 0].set_ylim(0, 105) ax[0].set_xlabel("Charging Station ID", fontsize=12) ax[0].set_ylabel("Utilization(%)",
# fontsize=12) ax[0].set_title("Proposed Model", fontsize=12) ax[0].tick_params(axis='both', labelsize=12) ax[
# 1].set_ylim(0, 105) ax[1].set_xlabel("Charging Station ID", fontsize=12) ax[1].set_ylabel("Utilization(%)",
# fontsize=12) ax[1].set_title("Benchmark Model", fontsize=12) ax[1].tick_params(axis='both', labelsize=12)
# plt.savefig("CSs.pdf") plt.show()

def plot_CS(dg1, dg2):
    set_style()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), constrained_layout=True)
    # sns.barplot(data=dg1, x='hour', y='count', hue='charging_speed', palette=sns.color_palette('mako', n_colors=2),
    # ci = "sd", capsize = 0.1, ax=ax[0],dodge=False)
    sns.barplot(data=dg1, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1), capsize=0.1, ax=ax[0], dodge=False)
    sns.barplot(data=dg2, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1),
                capsize=0.1, ax=ax[1], dodge=False)
    # sns.barplot(data=dg4, x='hour', y='count', palette=sns.color_palette('mako', n_colors=1), ax=ax[1][1])
    ax[0].set_ylim(0, 105)
    ax[0].set_xlabel("Time of the day", fontsize=12)
    ax[0].set_ylabel("Utilization(%)", fontsize=12)
    ax[0].set_title("Proposed Model", fontsize=12)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("Time of the day", fontsize=12)
    ax[1].set_ylabel("Utilization(%)", fontsize=12)
    ax[1].set_title("Benchmark Model", fontsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    plt.savefig("CSs.pdf")
    plt.show()


# dg_CS_central_SOC = data_CS(df_CS_central)
dg_CS_sub_SOC = data_CS(df_CS_sub)
dg_CS_single_RL = data_CS(df_CS_single_RL)

print(dg_CS_single_RL)


# dg_vehicle_central_mode = data_vehicles(df_vehicle_central, strategy='central')[1]
# dg_vehicle_sub_mode = data_vehicles(df_vehicle_sub)[1]


# dg_CS_sub_75_SOC = data_CS(df_CS_sub_75)
# dg_CS_sub_100_SOC = data_CS(df_CS_sub_100)
# dg_CS_sub_250_SOC = data_CS(df_CS_sub_250)
# dg_CS_sub_150_SOC = data_CS(df_CS_sub_150)
# dg_CS_sub_300_SOC = data_CS(df_CS_sub_300)

# dg_CS_sub_flat_tariff = data_CS(df_CS_sub_flat_tariff)
# dg_CS_sub_high_price = data_CS(df_CS_sub_high_price)
# dg_vehicle_Num_CS_central_SOC = data_CS(df_CS_Num_CS_central)
# dg_vehicle_Num_CS_SOC = data_CS(df_CS_Num_CS)

# dg_CS_No_FC_central_SOC = data_CS(df_CS_No_FC_central)
# dg_CS_No_FC_SOC = data_CS(df_CS_No_FC)

# dg_vehicle_No_FC_central_mode = data_vehicles(df_vehicle_No_FC_central, strategy='central')[1]
# dg_vehicle_No_FC_mode = data_vehicles(df_vehicle_No_FC)[1]

'''dg_vehicle_central_SOC = data_vehicles(df_vehicle_central, strategy='central')[0]
dg_vehicle_sub_SOC = data_vehicles(df_vehicle_sub)[0]
dg_vehicle_Q_learning_SOC = data_vehicles(df_vehicle_Q_learning)[0]

dg_vehicle_central_mode = data_vehicles(df_vehicle_central, strategy='central')[1]
dg_vehicle_sub_mode = data_vehicles(df_vehicle_sub)[1]
dg_vehicle_Q_learning_mode = data_vehicles(df_vehicle_Q_learning)[1]'''

# plot_CS(dg_vehicle_Num_CS_SOC, dg_vehicle_Num_CS_central_SOC)
# plot_CS(dg_CS_sub_150_SOC, dg_CS_sub_SOC, dg_CS_sub_250_SOC, dg_CS_sub_300_SOC)
# plot_CS(dg_CS_sub_flat_tariff, dg_CS_sub_250_SOC, dg_CS_sub_high_price)
plot_CS(dg_CS_sub_SOC, dg_CS_single_RL)