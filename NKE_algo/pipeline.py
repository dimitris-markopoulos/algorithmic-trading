import pandas as pd
import numpy as np

class Pipeline:
    def __init__(self, a):
        self.a = a

    def returns(adj_close: pd.core.series.Series) -> list:
        log_returns = np.log(adj_close) - np.log(adj_close.shift(1))
        return log_returns