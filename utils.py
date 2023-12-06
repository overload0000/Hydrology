import pandas as pd
import numpy as np
import os
import re
import logging
from tqdm import tqdm, trange
from typing import List
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)


def check_dir(dir):
    """
    check the directory
    """
    if not os.path.exists(dir):
        os.mkdir(dir)


def extract_geo_data(
    df: pd.DataFrame,
    data_name: str,
):
    """
    extract geo data from the dataframe
    it's OK you ignore this function
    """
    logger.info(f"extracting {data_name} data")
    new_columns = (
        df.columns.to_series()
        .apply(lambda x: re.findall(r"\d+", x))
        .apply(lambda x: tuple(x))
    )
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    df = df.reset_index().melt(id_vars="index").set_index("index")
    df = df.reset_index().rename(columns={"index": "time"})
    df = df.rename(
        columns={"value": data_name, "variable_0": "LONG", "variable_1": "LAT"}
    )
    return df


def merge_data(
    dfs: List[pd.DataFrame],
):
    """
    merge the data
    it's OK you ignore this function
    """
    df = dfs[0]
    logger.info("merging data")
    for i in trange(1, len(dfs), leave=None):
        current = dfs[i]
        df[current.columns[-1]] = current.iloc[:, -1]
    return df


def get_extreme_pcp(df: pd.DataFrame, threshold=0.95):
    """
    get the extreme precipitation for each threshold
    """
    df = df.sort_values(by="pcp", ascending=False)
    return df.iloc[: int(df.shape[0] * (1 - threshold)), :]


def split_data_by_pcp(df: pd.DataFrame, pcp_step: float):
    """
    split the data by precipitation
    """
    min_pcp = df["pcp"].min()
    max_pcp = df["pcp"].max()
    logger.info("splitting data")
    groups = []
    for i in trange(int((max_pcp - min_pcp) // pcp_step), leave=False):
        groups.append(df[(df["pcp"] >= i) & (df["pcp"] < i + pcp_step)])
    return groups


def split_data_by_geo(df: pd.DataFrame, lon_step: float, lat_step: float):
    """
    split the data by longitude and latitude
    """
    min_lon = df["LONG"].min()
    max_lon = df["LONG"].max()
    min_lat = df["LAT"].min()
    max_lat = df["LAT"].max()

    groups = []
    logger.info("splitting data")
    for i in trange(
        int((max_lon - min_lon) // lon_step * (max_lat - min_lat) // lat_step),
        leave=None,
    ):
        for j in trange(int((max_lon - min_lon) // lon_step), leave=None):
            groups.append(
                df[
                    (df["LONG"] >= i)
                    & (df["LONG"] < i + lon_step)
                    & (df["LAT"] >= j)
                    & (df["LAT"] < j + lat_step)
                ]
            )

    return groups


def get_percentiles(df: pd.DataFrame):
    """
    get the percentiles of the precipitation
    """
    length = df.shape[0]
    df = df.sort_values(by="temp")
    temps = []
    pcps = []

    logger.info("getting percentiles")
    for i in trange(100, leave=None):
        curr = df.iloc[int(length * i / 100) : int(length * (i + 1) / 100), :]
        temps.append(curr["temp"].mean())
        pcps.append(curr["pcp"].mean())
    return temps, pcps


def get_extreme_for_each_temp(
    df: pd.DataFrame, step=0.5, thresholds=[0.9, 0.95, 0.99], drop_threshold=300
):
    """
    group the data by temperature with a given step
    drop the group with less than drop_threshold samples
    for each group, get the extreme precipitation for each threshold
    """
    min_temp = df["temp"].min()
    max_temp = df["temp"].max()

    groups = []

    logger.info("grouping data by temperature with step {}".format(step))
    for i in trange(int((max_temp - min_temp) // step), leave=None):
        groups.append(df[(df["temp"] >= i) & (df["temp"] < i + step)])
    logger.info("dropping groups with less than {} samples".format(drop_threshold))
    groups = [group for group in groups if group.shape[0] >= drop_threshold]

    extreme_pcps = []
    avg_extreme_pcps = []

    logger.info("getting average extremes for thresholds {}".format(thresholds))
    for threshold in tqdm(thresholds, leave=None):
        extreme_pcp = []
        avg_extreme_pcp = []
        for group in tqdm(groups, leave=None):
            extreme_pcp.append(get_extreme_pcp(group, threshold))

        for i in tqdm(range(len(extreme_pcp)), leave=None):
            avg_extreme_pcp.append(extreme_pcp[i][["pcp", "temp"]].mean())

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

    for i in trange(len(extreme_pcps), leave=None):
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


def log_linear_regression(extreme_pcps: List[pd.DataFrame], name: str, epsilon=1e-3):
    """
    run log linear regression

    Args:
        extreme_pcps: list of extreme precipitation for each threshold
        name: name of the regression
        epsilon: epsilon for log to avoid log(0)
    """
    logger.info("log regression")
    with open(os.path.join("output", name + "log_regression.txt"), "w") as f:
        f.write("log regression\n")
        f.write("epsilon = {}\n".format(epsilon))
        f.write("criterion = {}\n".format(name))
        for i in trange(len(extreme_pcps), leave=None):
            extreme_pcp = extreme_pcps[i]
            extreme_pcp["log_pcp"] = np.log(extreme_pcp["pcp"] + epsilon)

            extreme_pcp = extreme_pcp[
                extreme_pcp["temp"] > extreme_pcp["temp"].quantile(0.01)
            ]
            extreme_pcp = extreme_pcp[
                extreme_pcp["temp"] < extreme_pcp["temp"].quantile(0.99)
            ]

            model = ols("log_pcp ~ temp", data=extreme_pcp).fit()
            f.write(f"threshold: {i}\n")
            f.write(model.summary().as_text())
