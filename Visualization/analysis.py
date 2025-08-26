import shapely
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from h3 import h3
from shapely.geometry import shape
import contextily as ctx
import geopandas as gpd
import seaborn as sns

set_matplotlib_formats('retina')


def set_style():
    # This sets reasonable defaults for font size for a figure that will go in a paper
    sns.set_context("paper")
    plt.rcParams.update({'font.size': 20})
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    # sns.set_color_codes("pastel")
    sns.set_palette("mako", n_colors=2)
    # Make the background white, and specify the font family
    sns.set_style("ticks", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def transform_df_to_gdf(df, crs='EPSG:4326'):
    return gpd.GeoDataFrame(data=df,
                            geometry=list(map(lambda h3_index: shapely.geometry.Polygon(
                                h3.h3_to_geo_boundary(h=h3_index, geo_json=True)), df.index)),
                            crs=crs)


# A function plotting a choropleth map for a geo data frame. It zooms in to Berlin and adds points of interest.
def plot_static_choropleth_map(gdf, ax, title, column='num_trips', label='Number of trips by hexagon', cmap='Reds',
                               crs='EPSG:4326'):
    # Construct choropleth map
    gdf.cx[13.25:13.55, 52.4:52.6].plot(column=column, ax=ax, alpha=0.6, cmap=cmap, legend=True,
                                        legend_kwds={'label': label, 'orientation': 'horizontal'})
    ctx.add_basemap(ax=ax, crs=crs)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Add main station
    ax.scatter(x=13.369402, y=52.525084, label='Main station', color='blue')
    # Add Humboldt university
    ax.scatter(x=13.3914611, y=52.5178862, label='Humboldt university', color='green')
    ax.legend()


def clean(x):
    if type(x) != float:
        x = x.replace('[', '').replace(']', '')
    return x


# def profit_analysis(profit_data, service_level_data):
#     set_style()
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
#     ax[0].plot(profit_data['Episode'], profit_data['Full_DQN'], label='Full_MARL', linestyle='dotted', marker='.')
#     ax[0].plot(profit_data['Episode'], profit_data['Full_HDQN'], label='Full_HMARL', linestyle='dotted', marker='o')
#     ax[0].plot(profit_data['Episode'], profit_data['Limited_DQN'], label='Limited_MARL', linestyle='dotted', marker='v')
#     ax[0].plot(profit_data['Episode'], profit_data['Limited_HDQN'], label='Limited_HMARL', linestyle='dotted', marker='x')
#
#
#
#     ax[0].ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
#     ax[0].set_xlabel("Episode", fontsize = 12)
#     ax[0].set_ylabel("Profit($)", fontsize = 12)
#     ax[0].set_title("Fleet profit", fontsize = 12)
#     ax[0].legend(fontsize = 11)
#     ax[0].tick_params(axis='both', labelsize=12)
#
#     ax[1].plot(service_level_data['Episode'], service_level_data['Full_DQN'], label='Full_MARL', linestyle='dotted'
#                , marker='.')
#     ax[1].plot(service_level_data['Episode'], service_level_data['Full_HDQN'], label='Full_HMARL', linestyle='dotted'
#                , marker='o')
#     ax[1].plot(service_level_data['Episode'], service_level_data['Limited_DQN'], label='Limited_MARL',
#                linestyle='dotted', marker='v')
#     ax[1].plot(service_level_data['Episode'], service_level_data['Limited_HDQN'], label='Limited_HMARL',
#                linestyle='dotted', marker='x')
#     ax[1].ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
#     ax[1].set_xlabel("Episode", fontsize = 12)
#     ax[1].set_ylabel("Served trips", fontsize = 12)
#     ax[1].set_title("Service quality", fontsize = 12)
#
#     ax[1].legend(fontsize = 11)
#     ax[1].tick_params(axis='both', labelsize=12)
#     plt.savefig("profit_RL.pdf")
#     plt.show()

def profit_analysis(profit_data, service_level_data):
    set_style()
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), constrained_layout=True)
    ax[0].plot(profit_data['Episode'], profit_data['Limited_HDQN'], label='Limited_HMARL', linestyle='solid', marker='x')
    ax[0].plot(profit_data['Episode'], profit_data['Q_learning'], label='Q_learning', linestyle='solid', marker='x')
    ax[0].plot(profit_data['Episode'], profit_data['Upperbound'], label='Upper bound', linestyle='dashed', marker='_')
    ax[0].plot(profit_data['Episode'], profit_data['Benchmark'], label='Benchmark', linestyle='dotted', marker='.')
    ax[0].ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
    ax[0].set_xlabel("Episode", fontsize=12)
    ax[0].set_ylabel("Profit($)", fontsize=12)
    ax[0].set_title("Fleet profit", fontsize=12)
    ax[0].legend(fontsize=12)
    ax[0].tick_params(axis='both', labelsize=12)

    ax[1].plot(service_level_data['Episode'], service_level_data['Limited_HDQN'], label='Limited_HMARL',
               linestyle='solid', marker='x')
    ax[1].plot(service_level_data['Episode'], service_level_data['Q_learning'], label='Q_learning',
               linestyle='solid', marker='x')
    ax[1].plot(service_level_data['Episode'], service_level_data['Upperbound'], label='Upper bound', linestyle='dashed',
               marker='_')
    ax[1].plot(service_level_data['Episode'], service_level_data['Benchmark'], label='Benchmark', linestyle='dotted', marker='.')
    ax[1].ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
    ax[1].set_xlabel("Episode", fontsize=12)
    ax[1].set_ylabel("Served trips", fontsize=12)
    ax[1].set_title("Service quality", fontsize=12)
    ax[1].legend(fontsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    plt.savefig("profit_RL.pdf")
    plt.show()


profit_data = pd.read_csv('../results/profit_RL.csv')
service_level_data = pd.read_csv('../results/service_level_RL.csv')
print(profit_data['Limited_HDQN'])
print(service_level_data['Limited_HDQN'])
profit_analysis(profit_data, service_level_data)
