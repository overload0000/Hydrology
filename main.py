import pandas as pd
import numpy as np
import logging
from utils import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def pcp_split_regression(df: pd.DataFrame, pcp_bin: int):
    """
    split the data by precipitation
    """
    gpt_data_by_pcp_groups = split_data_by_pcp(df, pcp_bin)

    logger.info("getting extreme precipitation and run log linear regression")
    for i, group in tqdm(
        enumerate(gpt_data_by_pcp_groups), total=len(gpt_data_by_pcp_groups)
    ):
        extreme_pcps, avg_extreme_pcps = get_extreme_for_each_temp(group)
        log_linear_regression(extreme_pcps, f"pcp_{i * pcp_bin}-{(i + 1) * pcp_bin}")


def geo_split_regression(df: pd.DataFrame, lat_bin: int, lon_bin: int):
    """
    split the data by lat and lon
    """
    gpt_data_by_geo_groups = split_data_by_geo(df, lat_bin, lon_bin)

    logger.info("getting extreme precipitation and run log linear regression")
    for i, group in tqdm(
        enumerate(gpt_data_by_geo_groups), total=len(gpt_data_by_geo_groups)
    ):
        extreme_pcps, avg_extreme_pcps = get_extreme_for_each_temp(group)
        log_linear_regression(extreme_pcps, f"geo_{i}")


if __name__ == "__main__":
    logger.info("loading data")
    gpt_data = pd.read_pickle("data/gpt_data.pkl")

    logger.info("splitting data by precipitation and apply log linear regression")
    pcp_split_regression(gpt_data, 200)
