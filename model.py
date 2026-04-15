import numpy as np
import pandas as pd
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def prepare_data(df):

    def to_day_seconds(ts):
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.hour*3600 + dt.minute*60 + dt.second

    df['actual_sec'] = df['arrival_time'].apply(to_day_seconds)

    df['hour'] = (df['actual_sec'] // 3600) % 24
    df['stop_sequence'] = np.random.randint(1, 50, size=len(df))

    df['delay'] = np.random.randint(60, 300, size=len(df))

    df = df.sort_values(['trip_id', 'stop_sequence'])
    df['prev_delay'] = df.groupby('trip_id')['delay'].shift(1)
    df = df.dropna()

    return df


def train_models(df):
    features = ['hour', 'stop_sequence', 'prev_delay']

    X = df[features]
    y = df['delay']

    models = {
        "lr": LinearRegression(),
        "rf": RandomForestRegressor(),
        "xgb": XGBRegressor(),
        "lgb": LGBMRegressor()
    }

    for m in models.values():
        m.fit(X, y)

    return models


def predict(models, hour, stop_sequence, prev_delay):
    X = np.array([[hour, stop_sequence, prev_delay]])

    preds = [m.predict(X)[0] for m in models.values()]

    return np.mean(preds), preds