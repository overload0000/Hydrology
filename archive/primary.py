# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols


max_longitude = 135
min_longitude = 73
max_latitude = 54
min_latitude = 18
pic_output_path = './pic'



# %%


# %%


# %% [markdown]
# ### 整体范围内，极端降水与温度的关系

# %%
# np.array(precipitation.drop(columns=['year']).unstack()).shape

# %%
# np.array(temperature.unstack()).shape

# %%

# data

# %%
# sns.displot(data, x="temp")

# %%
def get_extreme_pcp(df, threshold=0.95):
    # print(df)
    # print(df['pcp'].quantile(threshold))
    df = df.sort_values(by='pcp', ascending=False)
    
    
    return df.iloc[:int(df.shape[0]*(1-threshold)), :]

def get_extreme_for_each_temp(df):
    """
    group the data by temperature, step 0.5;
    drop the group with less than 300 samples;
    for each group, get the extreme precipitation for each threshold: 0.9, 0.95, 0.99
    return 3 dataframes for each threshold
    """
    min = df['temp'].min()
    max = df['temp'].max()
    step = 0.5
    groups = []
    for i in np.arange(min, max, step):
        groups.append(df[(df['temp'] >= i) & (df['temp'] < i+step)])
    # print(len(groups))
    groups = [group for group in groups if group.shape[0] >= 1000]
    # print(len(groups))
    extreme_pcp_90 = []
    extreme_pcp_95 = []
    extreme_pcp_99 = []
    for group in groups:
        # print(group.shape)
        extreme_pcp_90.append(get_extreme_pcp(group, 0.9))
        extreme_pcp_95.append(get_extreme_pcp(group, 0.95))
        extreme_pcp_99.append(get_extreme_pcp(group, 0.99))
        # print(get_extreme_pcp(group, 0.95).shape)
    
    # calculate the average extreme precipitation for each threshold
    avg_extreme_pcp_90 = []
    avg_extreme_pcp_95 = []
    avg_extreme_pcp_99 = []
    for i in range(len(extreme_pcp_90)):
        avg_extreme_pcp_90.append(extreme_pcp_90[i].mean())
        avg_extreme_pcp_95.append(extreme_pcp_95[i].mean())
        avg_extreme_pcp_99.append(extreme_pcp_99[i].mean())
        
    avg_extreme_pcp_90 = pd.concat(avg_extreme_pcp_90)
    avg_extreme_pcp_95 = pd.concat(avg_extreme_pcp_95)
    avg_extreme_pcp_99 = pd.concat(avg_extreme_pcp_99)
    # plot the average extreme precipitation for each threshold
    plt.figure(figsize=(12, 8))
    plt.plot(avg_extreme_pcp_90["temp"], avg_extreme_pcp_90['pcp'], label='90%')
    plt.plot(avg_extreme_pcp_95["temp"], avg_extreme_pcp_95['pcp'], label='95%')
    plt.plot(avg_extreme_pcp_99["temp"], avg_extreme_pcp_99['pcp'], label='99%')
    plt.legend()
    plt.savefig(os.path.join(pic_output_path, 'extreme_pcp.png'))
    
    plt.figure(figsize=(12, 8))
    plt.plot(avg_extreme_pcp_90["temp"], np.log(avg_extreme_pcp_90['pcp']), label='90%')
    plt.plot(avg_extreme_pcp_95["temp"], np.log(avg_extreme_pcp_95['pcp']), label='95%')
    plt.plot(avg_extreme_pcp_99["temp"], np.log(avg_extreme_pcp_99['pcp']), label='99%')
    # add background line, with slope = 0.07, intercept = [0,1,2,3,4]
    x = np.arange(-40, 40, 0.2)
    for i in range(6):
        plt.plot(x, 0.07*x+i, color='grey', alpha=0.5)
    
    plt.legend()
    plt.savefig(os.path.join(pic_output_path, 'extreme_pcp_log.png'))
    
    
    
    extreme_pcp_90 = pd.concat(extreme_pcp_90)
    extreme_pcp_95 = pd.concat(extreme_pcp_95)
    extreme_pcp_99 = pd.concat(extreme_pcp_99)
    return extreme_pcp_90, extreme_pcp_95, extreme_pcp_99





# %%
def percentiles(df):
    length = df.shape[0]
    df = df.sort_values(by='temp')
    temps = []
    pcps = []
    for i in range(100):
        curr = df.iloc[int(length*i/100):int(length*(i+1)/100), :]
        temps.append(curr['temp'].mean())
        pcps.append(curr['pcp'].mean())
    
    return temps, pcps

def percentile_regression(extreme_pcp_90, extreme_pcp_95, extreme_pcp_99):
    temps90, pcps90 = percentiles(extreme_pcp_90)
    temps95, pcps95 = percentiles(extreme_pcp_95)
    temps99, pcps99 = percentiles(extreme_pcp_99)

    plt.figure(figsize=(12, 8))
    plt.plot(temps90, pcps90, label='90%')
    plt.plot(temps95, pcps95, label='95%')
    plt.plot(temps99, pcps99, label='99%')
    plt.legend()
    plt.savefig(os.path.join(pic_output_path, 'extreme_pcp_percentile.png'))

    plt.figure(figsize=(12, 8))
    plt.plot(temps90, np.log(np.array(pcps90)), label='90%')
    plt.plot(temps95, np.log(np.array(pcps95)), label='95%')
    plt.plot(temps99, np.log(np.array(pcps99)), label='99%')
    plt.legend()
    x = np.arange(-40, 40, 0.2)
    for i in range(6):
        plt.plot(x, 0.07*x+i, color='grey', alpha=0.5)
    plt.savefig(os.path.join(pic_output_path, 'extreme_pcp_percentile_log.png'))


def log_linear_regression(extreme_pcp_90, extreme_pcp_95, extreme_pcp_99):
    """
    +0.01mm, 防止log(0)出现
    """
    
    extreme_pcp_90['log_pcp'] = np.log(extreme_pcp_90['pcp'] + 0.001)
    extreme_pcp_95['log_pcp'] = np.log(extreme_pcp_95['pcp'] + 0.001)
    extreme_pcp_99['log_pcp'] = np.log(extreme_pcp_99['pcp'] + 0.001)
    
    # percentile(1-99)
    extreme_pcp_90 = extreme_pcp_90[extreme_pcp_90['temp'] > extreme_pcp_90['temp'].quantile(0.01)]
    extreme_pcp_95 = extreme_pcp_95[extreme_pcp_95['temp'] > extreme_pcp_95['temp'].quantile(0.01)]
    extreme_pcp_99 = extreme_pcp_99[extreme_pcp_99['temp'] > extreme_pcp_99['temp'].quantile(0.01)]
    
    extreme_pcp_90 = extreme_pcp_90[extreme_pcp_90['temp'] < extreme_pcp_90['temp'].quantile(0.99)]
    extreme_pcp_95 = extreme_pcp_95[extreme_pcp_95['temp'] < extreme_pcp_95['temp'].quantile(0.99)]
    extreme_pcp_99 = extreme_pcp_99[extreme_pcp_99['temp'] < extreme_pcp_99['temp'].quantile(0.99)]

    print("percentile 90")
    model = ols("log_pcp ~ temp", data=extreme_pcp_90).fit()
    print(model.summary())
    
    print("percentile 95")
    model = ols("log_pcp ~ temp", data=extreme_pcp_95).fit()
    print(model.summary())

    model = ols("log_pcp ~ temp", data=extreme_pcp_99).fit()
    print("percentile 99")
    print(model.summary())
    
    

# %%
def describe_relationship(temperature, precipitation):
    temp = np.array(temperature.unstack())
    pcp = np.array(precipitation.drop(columns=['year']).unstack())
    data = pd.DataFrame({'temp':temp, 'pcp':pcp})
    extreme_pcp_90, extreme_pcp_95, extreme_pcp_99 = get_extreme_for_each_temp(data)
    plt.figure(figsize=(8, 6))
    sns.distplot(extreme_pcp_90['temp'])
    plt.savefig(os.path.join(pic_output_path, 'temp_dist_90.png'))
    plt.figure(figsize=(8, 6))
    sns.distplot(extreme_pcp_95['temp'])
    plt.savefig(os.path.join(pic_output_path, 'temp_dist_95.png'))
    plt.figure(figsize=(8, 6))
    
    percentile_regression(extreme_pcp_90, extreme_pcp_95, extreme_pcp_99)
    log_linear_regression(extreme_pcp_90, extreme_pcp_95, extreme_pcp_99)
    
# # %%
# extreme_pcp_99

def main():
    precipitation = pd.read_pickle('data/pcp_data.pkl')
    temperature = pd.read_pickle('data/temp_data.pkl')


    precipitation['year'] = precipitation.index.year

    precipitation_yearly = precipitation.groupby('year').sum()


# normality test for precipitation

    p_values = []
    for col in precipitation_yearly.columns:
        _, p = shapiro(np.log(precipitation_yearly[col]))
        p_values.append(p)
        
    p_values = np.array(p_values)

    print("p < 0.05: ", np.sum(p_values < 0.05)/p_values.shape[0])
    print("p < 0.01: ", np.sum(p_values < 0.01)/p_values.shape[0])
    print("p < 0.001: ", np.sum(p_values < 0.001)/p_values.shape[0])

# %% [markdown]
# #### 在0.05显著水平下，无法拒绝90%以上的样本属于对数正态分布，故使用对数正态近似


    pcpSummary = pd.read_csv("./CMADS/For-swat-2012/Fork/PCPFORK.txt")
    pcpSummary = pcpSummary[(pcpSummary['LONG'] <= max_longitude) & (pcpSummary['LONG'] >= min_longitude) & (pcpSummary['LAT'] <= max_latitude) & (pcpSummary['LAT'] >= min_latitude)]


# %%

# %%

    pcpSummary["precipitation"] = precipitation_yearly.mean().values
    pcpSummary["precipitation_std"] = precipitation_yearly.std().values
    pcpSummary["avg_temp"] = temperature.mean().values
    
    # example, overall analysis
    describe_relationship(temperature, precipitation)
    

if __name__ == "__main__":
    main()


# %% [markdown]
# 

# # %%
# extreme_pcp_99.mean()

# # %%
# data.sort_values(by=['temp'])

# %%



