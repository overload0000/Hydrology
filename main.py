import pandas as pd
import numpy as np
import logging
from utils import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    precipitation = pd.read_pickle('data/pcp_data.pkl')
    temperature = pd.read_pickle('data/temp_data.pkl')

    # import pdb; pdb.set_trace()
    pcp_geo = extract_geo_data(precipitation, "pcp")
    temp_geo = extract_geo_data(temperature, "temp")

    data = merge_data([pcp_geo, temp_geo])