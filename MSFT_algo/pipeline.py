import pandas as pd
import numpy as np
import yfinance as yf
import sklearn
from sklearn.preprocessing import StandardScaler

class Pipeline:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def prepare_data(self):
        params = {
            'tickers':['MSFT','^VIX','^TNX','SPY','XLK'],
            'start':self.start,
            'end':self.end,
            'auto_adjust':False,
            'progress':False
        }

        raw_data = yf.download(**params)
        raw_data = raw_data.ffill() # Fill NA values with last previous trading day

        assert raw_data.isna().sum().sum() == 0, 'exists NA' # Error Check -> throw error if missing data exists.

        #=====Engineer Features=====

        #---Returns---
        tickers = params['tickers']
        engineered_dict = {f'{ticker}_logreturns': [] for ticker in tickers if ticker not in ['^VIX', '^TNX']}
        for ticker in tickers:
            if ticker in ['^VIX', '^TNX']:
                pass
            else:
                adj = raw_data.loc[:,('Adj Close',ticker)]
                log_returns = np.log(adj) - np.log(adj.shift(1))
                log_returns_vals = list(log_returns.values)
                engineered_dict[str(f'{ticker}_logreturns')].extend(log_returns)

            if ticker in ['MSFT']:
                ewma = log_returns.ewm(alpha=0.1, adjust=False).mean()
                engineered_dict[f'{ticker}_ewm'] = ewma

        vix = raw_data.loc[:, ('Adj Close', '^VIX')]
        engineered_dict['VIX'] = vix

        tnx = raw_data.loc[:, ('Adj Close', '^TNX')]
        engineered_dict['TNX'] = tnx
        returns_ewm_df = pd.DataFrame(engineered_dict, index=raw_data.index).dropna() # Drop NA

        df = returns_ewm_df

        #---Moving Average Differential/Spread---
        long_window = 50
        short_window = 20
        long_ma = raw_data.loc[params['start']:params['end'],('Adj Close','MSFT')].rolling(long_window).mean()
        short_ma = raw_data.loc[params['start']:params['end'],('Adj Close','MSFT')].rolling(short_window).mean()
        ma_spread = long_ma - short_ma
        ma_spread = ma_spread.rename(f'{long_window}_{short_window}_ma_spread')

        #---Volume (not engineered just appended)---
        raw_data.loc[:, ('Volume','MSFT')]

        vol_dict = {'Date':[], 'MSFT_volume':[], 'SPY_volume':[], 'XLK_volume':[]}
        for ticker in ['MSFT','SPY','XLK']:
            
            if vol_dict['Date'] == []:
                vol_dict['Date'].extend(raw_data.index)

            vol_dict[ticker + '_volume'].extend(raw_data.loc[:, ('Volume',ticker)].values)

        volume_df = pd.DataFrame(vol_dict)
        volume_df.index = volume_df['Date']
        volume_df = volume_df.drop(columns='Date')

        #---Working DataFrame---
        df = pd.concat([df,ma_spread,volume_df], axis=1).dropna()
        df2 = df.copy() # copy before target
        
        #---Targets--- Attained by shifting future date back in time and assigning to 1 positive, 0 negative
        shifted = df['MSFT_logreturns'].shift(-1)
        df['Target'] = shifted.apply(lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else np.nan))
        df = df.dropna()

        df2['Target'] = shifted.apply(lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) else np.nan))
        df2 = df2.dropna()

        self.df = df
        self.df2 = df2 # Use when 1,-1 mapping is required such as SVMs
        
        return df
    
    def split(self, df, train_proportion = 0.90, scale = True):

        #---Split---
        split = {'prop_tr':train_proportion,'prop_ts':1-train_proportion}
        assert abs(sum(split.values()) - 1.0) < 1e-8, "Split proportions must sum to 1.0"

        n,p = df.shape
        n_tr = int(np.ceil(n*split['prop_tr']))

        X = df.drop(columns='Target')
        y = df['Target']

        X_tr, y_tr = X.iloc[:n_tr].to_numpy(), y.iloc[:n_tr].to_numpy()
        X_ts, y_ts = X.iloc[n_tr:].to_numpy(), y.iloc[n_tr:].to_numpy()
        
        if scale == False:
            return {'X_tr': X_tr, 'X_ts': X_ts, 'y_tr':y_tr, 'y_ts':y_ts}

        else:
            scaler = StandardScaler() # Scale features
            X_tr_scaled = scaler.fit_transform(X_tr) # fit on training and scale
            X_ts_scaled = scaler.transform(X_ts) # use trained scaler (dont leak test set by using it to scale!)
            return {'X_tr_scaled': X_tr_scaled, 'X_ts_scaled': X_ts_scaled, 'y_tr':y_tr, 'y_ts':y_ts}
        
        