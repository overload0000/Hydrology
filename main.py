import pandas as pd
import logging
from utils import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def pcp_split_regression(df: pd.DataFrame, pcp_bin: int):
    """
    split the data by precipitation

    Args:
        df: the dataframe
        pcp_bin: the bin of precipitation
    """
    gpt_data_by_pcp_groups = split_data_by_pcp(df, pcp_bin)

    logger.info("getting extreme precipitation and run log linear regression")
    for i, group in tqdm(
        enumerate(gpt_data_by_pcp_groups), total=len(gpt_data_by_pcp_groups)
    ):
        extreme_pcps, avg_extreme_pcps = get_extreme_for_each_temp(group)
        log_linear_regression(extreme_pcps, f"pcp_{i * pcp_bin}-{(i + 1) * pcp_bin}")


def geo_split(df: pd.DataFrame, lat_bin: int, lon_bin: int):
    """
    split the data by lat and lon

    Args:
        df: the dataframe
        lat_bin: the bin of latitude
        lon_bin: the bin of longitude
    """
    # gpt_data_by_geo_groups = split_data_by_geo(df, lat_bin, lon_bin)
    split_data_by_geo_parallel(df, lat_bin, lon_bin, 6)


def geo_split_regression():
    logger.info("getting extreme precipitation and run log linear regression")
    for i in tqdm(range(13, 20)):
        group = pd.read_pickle(f"data\\geo\\geo_{i}.pkl")
        try:
            extreme_pcps, avg_extreme_pcps = get_extreme_for_each_temp(group)
            log_linear_regression(extreme_pcps, f"geo_{i}")
        except Exception as e:
            print(f"error in geo_{i}")
            logger.error(e)
            continue


def per_station_analysis(gpt_data: pd.DataFrame, save_name, temp_bin: int = 1):
    """
    对于每个测站，在指定温度范围内，每隔1度找到top5% percent
    线性回归，获得每个测站的 beta_1 作为y，以之前提取的特征（降水量特征（年均值，年际方差 etc.)、温度、地形特征）为X，做整体预测
    """
    import numpy as np
    from statsmodels.formula.api import ols

    # 按照LONG和LAT 一起分组
    gpt_data.sort_values(by=["LONG", "LAT"], inplace=True)
    grouped = gpt_data.groupby(["LONG", "LAT"])

    with open(save_name, "w") as f:
        f.write("LONG,LAT,beta_1\n")
        for name, group in grouped:
            logger.info("-----------long: {}, lat: {}".format(name[0], name[1]))
            min_temp = group["temp"].min()
            max_temp = group["temp"].max()
            # 对于每个测站，从最小温度到最大温度，每隔1度，在区间中提取降水top5%的数据（0.5度太小数据不够）
            # 然后用线性回归：log(降水)对温度回归，得到的beta添加到beta_list中
            sub_df_by_temp_list = []
            for temp in np.arange(min_temp, max_temp, temp_bin):
                sub_sub_df = group[
                    (group["temp"] >= temp) & (group["temp"] < temp + temp_bin)
                ]
                sub_sub_df = sub_sub_df.nlargest(int(len(sub_sub_df) * 0.05), "pcp")
                sub_df_by_temp_list.append(sub_sub_df)
            sub_df_by_temp_list = pd.concat(sub_df_by_temp_list)
            # take log of pcp
            sub_df_by_temp_list["logpcp"] = np.log(
                sub_df_by_temp_list["pcp"] + 1e-6
            )  # avoid log(0)
            # linear regression pandas ols
            lm = ols("logpcp ~ temp", data=sub_df_by_temp_list).fit()
            f.write("{},{},{}\n".format(name[0], name[1], lm.params[1]))


if __name__ == "__main__":
    logger.info("loading data")
    gpt_data = pd.read_pickle("data/gpt_data.pkl")
    logger.info("splitting data by precipitation and apply log linear regression")
    per_station_analysis(gpt_data, "output/per_station_analysis.csv")
