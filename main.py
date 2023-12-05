import pandas as pd
import numpy as np
from .utils import *

if __name__ == "__main__":
    precipitation = pd.read_pickle('data/pcp_data.pkl')
    temperature = pd.read_pickle('data/temp_data.pkl')

    import pdb; pdb.set_trace()