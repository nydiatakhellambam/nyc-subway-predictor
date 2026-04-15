import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import pydeck as pdk

from realtime_fetch import fetch_realtime_data
from model import prepare_data, train_models, predict

st.set_page_config(layout="wide")

st.title("🚇 Live Subway AI System")

# ==============================
# FETCH LIVE DATA
# ==============================
df = fetch_realtime_data()

if df.empty:
    st.warning("No live data available")
    st.stop()

# ==============================
# PREPARE + TRAIN
# ==============================
df = prepare_data(df)
models = train_models(df)

# ==============================
# USER INPUT
# ==============================
hour = st.slider("Hour", 0, 23, 12)
stop_sequence = st.slider("Stop Sequence", 1, 50, 10)
prev_delay = st.slider("Previous Delay", 0, 300, 60)

# ==============================
# PREDICT
# ==============================
prediction, preds = predict(models, hour, stop_sequence, prev_delay)

st.metric("Predicted Delay", f"{round(prediction)} sec")

# ==============================
# LOAD STATIONS
# ==============================
stops = pd.read_csv("gtfs_subway/stops.txt")
stops = stops[['stop_id', 'stop_lat', 'stop_lon']]

df['stop_id'] = df['stop_id'].str[:-1]

map_df = df.merge(stops, on='stop_id', how='inner')

# ==============================
# ANIMATED MAP 🔥
# ==============================
st.subheader("🚆 Live Train Movement")

placeholder = st.empty()

for i in range(20):
    sample = map_df.sample(200)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sample,
        get_position='[stop_lon, stop_lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=100,
    )

    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10
    )

    placeholder.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state
    ))

    time.sleep(1)