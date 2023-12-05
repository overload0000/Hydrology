import pandas as pd
import numpy as np
import os
import logging
from rich.progress import track

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_data_summary(filepath, min_longitude, max_longitude, min_latitude, max_latitude):
    """
    get the data summary of the region

    Args:
        filepath: the path of the file
        min_longitude: the min longitude of the region
        max_longitude: the max longitude of the region
        min_latitude: the min latitude of the region
        max_latitude: the max latitude of the region
    """
    assert os.path.exists(filepath), f'{filepath} does not exist'
    logging.info(f'get data summary from {filepath}')
    df = pd.read_csv(filepath)
    df = df[(df['LONG'] <= max_longitude) & (df['LONG'] >= min_longitude) & (df['LAT'] <= max_latitude) & (df['LAT'] >= min_latitude)]
    return df

def get_raw_data(dir, file_list):
    """
    get the data of the region

    Args:
        dir: the directory of the data
        file_list: the list of the file
    """
    assert os.path.exists(dir), f'{dir} does not exist'
    columns = np.array([])

    logging.info(f'get raw data from {dir}')
    for file in track(file_list):
        filepath = os.path.join(dir, file)+'.txt'
        assert os.path.exists(filepath), f'{filepath} does not exist'
        tmp = np.loadtxt(filepath,skiprows=1)
        columns = np.append(columns, np.nanmean(tmp, axis=1))
        
    df = pd.DataFrame(columns=columns)
    df.columns = file_list
    date = pd.date_range(start='2008-01-01', end='2016-12-31', freq='D')
    df.set_index(date, inplace=True)
    return df

def clean_data(min_longitude, max_longitude, min_latitude, max_latitude, data_type):
    """
    clean the data

    Args:
        min_longitude: the min longitude of the region
        max_longitude: the max longitude of the region
        min_latitude: the min latitude of the region
        max_latitude: the max latitude of the region
        data_type: the type of the data(pcp or temp)
    """
    if not os.path.exists(f'data/{data_type}_data.pkl'):
        logging.info(f'clean the {data_type} data')
        summary_filepath = f'./CMADS/For-swat-2012/Fork/{data_type.upper()}FORK.txt'
        summary = get_data_summary(summary_filepath, min_longitude, max_longitude, min_latitude, max_latitude)

        if data_type == 'pcp':
            data_filepath = "./CMADS/For-swat-2012/Station/Precipitation-104000-txt"
        elif data_type == 'temp':
            data_filepath = "./CMADS/For-swat-2012/Station/Temperature-104000-txt"
        else:
            raise ValueError(f'invalid data type: {data_type}')
        
        data = get_raw_data(data_filepath, summary.NAME)

        if not os.path.exists('data'):
            os.mkdir('data')

        data.to_pickle(f'data/{data_type}_data.pkl')

def main():
    max_longitude = 135
    min_longitude = 73
    max_latitude = 54
    min_latitude = 18

    clean_data(min_longitude, max_longitude, min_latitude, max_latitude, 'pcp')
    clean_data(min_longitude, max_longitude, min_latitude, max_latitude, 'temp')

if __name__ == '__main__':
    main()


