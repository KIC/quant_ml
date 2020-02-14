import os
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DF_TEST_MULTI = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spy_gld.pickle"))
DF_TEST = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SPY.csv"), index_col='Date')
DF_DEBUG = pd.DataFrame({"Close": np.random.random(10)})
