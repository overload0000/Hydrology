import pandas as pd
import numpy as np
import os
from rich.progress import track
from typing import List
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols


def check_dir(dir):
    """
    check the directory
    """
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_extreme_pcp(df: pd.DataFrame, threshold=0.95):
    """
    get the extreme precipitation for each threshold
    """
    df = df.sort_values(by="pcp", ascending=False)
    return df.iloc[: int(df.shape[0] * (1 - threshold)), :]


def get_percentiles(df: pd.DataFrame):
    """
    get the percentiles of the precipitation
    """
    length = df.shape[0]
    df = df.sort_values(by="temp")
    temps = []
    pcps = []
    for i in track(range(100)):
        curr = df.iloc[int(length * i / 100) : int(length * (i + 1) / 100), :]
        temps.append(curr["temp"].mean())
        pcps.append(curr["pcp"].mean())

    return temps, pcps


def get_extreme_for_each_temp(
    df: pd.DataFrame, step=0.5, thresholds=[0.9, 0.95, 0.99], drop_threshold=1000
):
    """
    group the data by temperature with a given step
    drop the group with less than drop_threshold samples
    for each group, get the extreme precipitation for each threshold
    """
    min = df["temp"].min()
    max = df["temp"].max()

    groups = []
    for i in track(np.arange(min, max, step)):
        groups.append(df[(df["temp"] >= i) & (df["temp"] < i + step)])
    groups = [group for group in groups if group.shape[0] >= drop_threshold]

    extreme_pcps = []
    avg_extreme_pcps = []
    for threshold in track(thresholds):
        extreme_pcp = []
        avg_extreme_pcp = []
        for group in track(groups):
            extreme_pcp.append(get_extreme_pcp(group, threshold))

        for i in track(len(extreme_pcp)):
            avg_extreme_pcp.append(extreme_pcp[i].mean())

        avg_extreme_pcps.append(pd.concat(avg_extreme_pcp))

        extreme_pcps.append(pd.concat(extreme_pcp))

    return extreme_pcps, avg_extreme_pcps


def draw_avg_extreme_pcp(avg_extreme_pcps: List[pd.DataFrame], thresholds: List[float]):
    """
    draw the average extreme precipitation for each threshold
    """
    plt.figure(figsize=(12, 8))
    for i in range(len(avg_extreme_pcps)):
        plt.plot(
            avg_extreme_pcps[i]["temp"],
            avg_extreme_pcps[i]["pcp"],
            label=f"{thresholds[i]*100}%",
        )
    plt.legend()
    plt.savefig(os.path.join("pic", "extreme_pcp.png"))

    plt.figure(figsize=(12, 8))
    for i in range(len(avg_extreme_pcps)):
        plt.plot(
            avg_extreme_pcps[i]["temp"],
            np.log(avg_extreme_pcps[i]["pcp"]),
            label=f"{thresholds[i]*100}%",
        )

    # add background line, with slope = 0.07, intercept = [0,1,2,3,4]
    x = np.arange(-40, 40, 0.2)
    for i in range(6):
        plt.plot(x, 0.07 * x + i, color="grey", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join("pic", "extreme_pcp_log.png"))


def draw_percentiles(extreme_pcps: List[pd.DataFrame], thresholds: List[float]):
    """
    draw the percentiles of the precipitation
    """
    temps = []
    pcps = []

    for i in track(range(len(extreme_pcps))):
        temp, pcp = get_percentiles(extreme_pcps[i])
        temps.append(temp)
        pcps.append(pcp)

    plt.figure(figsize=(12, 8))
    for i in range(len(temps)):
        plt.plot(temps[i], pcps[i], label=f"{thresholds[i]*100}%")
    plt.legend()
    plt.savefig(os.path.join("pic", "percentile_regression.png"))

    plt.figure(figsize=(12, 8))
    for i in range(len(temps)):
        plt.plot(temps[i], np.log(np.array(pcps[i])), label=f"{thresholds[i]*100}%")
    plt.legend()
    x = np.arange(-40, 40, 0.2)
    for i in range(6):
        plt.plot(x, 0.07 * x + i, color="grey", alpha=0.5)
    plt.savefig(os.path.join("pic", "extreme_pcp_percentile_log.png"))


def log_linear_regression(extreme_pcps: List[pd.DataFrame], epsilon=1e-3):
    for i in track(range(len(extreme_pcps))):
        extreme_pcp = extreme_pcps[i]
        extreme_pcp["log_pcp"] = np.log(extreme_pcp["pcp"] + epsilon)

        extreme_pcp = extreme_pcp[
            extreme_pcp["temp"] > extreme_pcp["temp"].quantile(0.01)
        ]
        extreme_pcp = extreme_pcp[
            extreme_pcp["temp"] < extreme_pcp["temp"].quantile(0.99)
        ]

        model = ols("log_pcp ~ temp", data=extreme_pcp).fit()

        with open(os.path.join("output", "log_regression.txt"), "w") as f:
            f.write(model.summary().as_text())
