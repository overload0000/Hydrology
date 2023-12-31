import pandas as pd
import numpy as np
import os
from tqdm import tqdm, trange
from typing import List
from matplotlib import pyplot as plt


def get_extreme_pcp(df: pd.DataFrame, threshold=0.95):
    """
    get the extreme precipitation for each threshold

    Args:
        df: the dataframe
        threshold: the threshold of the extreme precipitation
    """
    df = df.sort_values(by="pcp", ascending=False)
    # Get the number of samples that pcp > 0
    num = df[df["pcp"] > 0].shape[0]
    # return df.iloc[: int(df.shape[0] * (1 - threshold)), :]
    return df.iloc[: int(num * (1 - threshold)), :]


def get_extreme_for_each_temp(
    df: pd.DataFrame, step=0.5, thresholds=[0.9, 0.95, 0.99], drop_threshold=300
):
    """
    group the data by temperature with a given step
    drop the group with less than drop_threshold samples
    for each group, get the extreme precipitation for each threshold

    Args:
        df: the dataframe(long format)
        step: the step of the temperature
        thresholds: the thresholds of the extreme precipitation
        drop_threshold: the threshold of the number of samples
    """
    min_temp = df["temp"].min()
    max_temp = df["temp"].max()

    groups = []

    for i in trange(int((max_temp - min_temp) // step), leave=None):
        groups.append(df[(df["temp"] >= i) & (df["temp"] < i + step)])
    groups = [group for group in groups if group.shape[0] >= drop_threshold]

    extreme_pcps = []
    avg_extreme_pcps = []

    for threshold in tqdm(thresholds, leave=None):
        extreme_pcp = []
        avg_extreme_pcp = []
        for group in tqdm(groups, leave=None):
            extreme_pcp.append(get_extreme_pcp(group, threshold))

        for i in tqdm(range(len(extreme_pcp)), leave=None):
            avg_extreme_pcp.append(extreme_pcp[i][["pcp", "temp"]].mean())

        avg_extreme_pcps.append(pd.concat(avg_extreme_pcp))

        extreme_pcps.append(pd.concat(extreme_pcp))

    # return extreme_pcps, avg_extreme_pcps
    return avg_extreme_pcps


def draw_avg_extreme_pcp(avg_extreme_pcps: List[pd.DataFrame], thresholds: [0.9, 0.95, 0.99], latitude: int):
    """
    draw the average extreme precipitation for each threshold

    Args:
        avg_extreme_pcps: the list of the average extreme precipitation
        thresholds: the thresholds of the extreme precipitation
    """
    # plt.figure(figsize=(12, 8))
    # for i in range(len(avg_extreme_pcps)):
    #     plt.plot(
    #         avg_extreme_pcps[i]["temp"],
    #         avg_extreme_pcps[i]["pcp"],
    #         label=f"{thresholds[i]*100}%",
    #     )
    # plt.legend()
    # plt.savefig(os.path.join("pic", "test.png"))

    plt.figure(figsize=(12, 8))
    for i in range(len(avg_extreme_pcps)):
        plt.plot(
            avg_extreme_pcps[i]["temp"],
            np.log(avg_extreme_pcps[i]["pcp"]),
            label=f"{thresholds[i]*100}%",
        )

    # add background line, with slope = 0.07, intercept = [0,1,2,3,4]
    x = np.arange(0, 40, 0.2)
    for i in range(6):
        plt.plot(x, 0.07 * x + i, color="grey", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join("pic/latitude_extreme_pcp_log", f"lat{latitude}.png"))


# df = pd.DataFrame()
# for i in range(250):
#     df_ = pd.read_pickle(f"geo/geo_{i}.pkl")
#     df = pd.concat([df, df_], ignore_index=True)
# avg_extreme_pcps = get_extreme_for_each_temp(df)
# draw_avg_extreme_pcp(avg_extreme_pcps, [0.9, 0.95, 0.99])

df = pd.read_pickle("data/gpt_data.pkl")
df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
for lat in range(53, 99, 5):
    df_lat = df[(df['LAT'] >= lat - 2) & (df['LAT'] <= lat + 2)]
    avg_extreme_pcps = get_extreme_for_each_temp(df_lat)
    draw_avg_extreme_pcp(avg_extreme_pcps, [0.9, 0.95, 0.99], lat)
