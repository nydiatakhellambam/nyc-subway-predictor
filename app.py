import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import pydeck as pdk
import os

from realtime_fetch import fetch_realtime_data
from model import prepare_data, train_models, predict

st.set_page_config(layout="wide")

st.title("🚇 Live Subway AI System")

# ==============================
# FETCH LIVE DATA (SAFE)
# ==============================
try:
    df = fetch_realtime_data()
except:
    df = pd.DataFrame()

# fallback if API fails
if df.empty:
    st.warning("⚠️ Live API failed — using fallback data")
    try:
        df = pd.read_csv("data/realtime_data.csv")
    except:
        st.error("❌ No data available. Please upload dataset.")
        st.stop()

# ==============================
# PREPARE + TRAIN
# ==============================
df = prepare_data(df)
models = train_models(df)

# ==============================
# USER INPUT
# ==============================
st.subheader("⚙️ Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider("Hour", 0, 23, 12)

with col2:
    stop_sequence = st.slider("Stop Sequence", 1, 50, 10)

with col3:
    prev_delay = st.slider("Previous Delay", 0, 300, 60)

# ==============================
# PREDICTION
# ==============================
prediction, preds = predict(models, hour, stop_sequence, prev_delay)

st.subheader("📊 Prediction Result")
st.metric("Predicted Delay", f"{round(float(prediction))} sec")

# ==============================
# LOAD STATIONS (SAFE)
# ==============================
map_df = pd.DataFrame()

try:
    if os.path.exists("gtfs_subway/stops.txt"):
        stops = pd.read_csv("gtfs_subway/stops.txt")
        stops = stops[['stop_id', 'stop_lat', 'stop_lon']]

        df['stop_id'] = df['stop_id'].astype(str).str[:-1]

        map_df = df.merge(stops, on='stop_id', how='inner')
    else:
        st.warning("⚠️ Map disabled: stops.txt not found")

except Exception as e:
    st.warning("⚠️ Error loading map data")

# ==============================
# MAP VISUALIZATION
# ==============================
st.subheader("🚆 Live Train Movement")

placeholder = st.empty()

if not map_df.empty:

    for i in range(20):

        sample = map_df.sample(min(200, len(map_df)))

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

else:
    st.info("📍 Map not available (missing station data)")

# ==============================
# INFO BOX
# ==============================
st.info("""
This system predicts subway delays using:
- Time of day (rush hour patterns)
- Stop sequence (delay accumulation)
- Previous station delay

Model: Ensemble (Linear + RF + XGBoost + LightGBM)
""")
