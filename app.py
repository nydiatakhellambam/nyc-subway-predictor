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

# ==============================
# HEADER
# ==============================
st.markdown("""
# 🚇 NYC Subway AI Predictor  
### Real-Time Delay Prediction & Visualization System
""")

# ==============================
# ROUTE FILTER
# ==============================
route = st.selectbox("🚇 Select Train Line", ["All", "A", "B", "C"])

# ==============================
# FETCH DATA
# ==============================
try:
    df = fetch_realtime_data()
except:
    df = pd.DataFrame()

if df.empty:
    st.warning("⚠️ Live API failed — using fallback data")
    try:
        df = pd.read_csv("data/realtime_data.csv")
    except:
        st.error("❌ No data available")
        st.stop()

# ==============================
# PREPARE DATA
# ==============================
df = prepare_data(df)

# route filter
if route != "All" and 'trip_id' in df.columns:
    df = df[df['trip_id'].astype(str).str.contains(route, na=False)]

models = train_models(df)

# ==============================
# INPUT PANEL
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

colA, colB = st.columns(2)

with colA:
    st.metric("⏱ Delay", f"{round(prediction)} sec")

with colB:
    current_time = datetime.datetime.now()
    arrival = current_time + datetime.timedelta(seconds=float(prediction))
    st.metric("🚉 Arrival Time", arrival.strftime("%H:%M:%S"))

# ==============================
# CONFIDENCE
# ==============================
if prediction < 120:
    confidence = "High ✅"
elif prediction < 300:
    confidence = "Medium ⚠️"
else:
    confidence = "Low ❌"

st.write(f"🔎 Confidence Level: **{confidence}**")

st.markdown("---")

# ==============================
# LOAD MAP DATA
# ==============================
map_df = pd.DataFrame()

try:
    if os.path.exists("gtfs_subway/stops.txt"):
        stops = pd.read_csv("gtfs_subway/stops.txt")
        stops = stops[['stop_id', 'stop_lat', 'stop_lon']]

        df['stop_id'] = df['stop_id'].astype(str).str[:-1]
        map_df = df.merge(stops, on='stop_id', how='inner')
    else:
        st.warning("⚠️ Map disabled: stops.txt missing")

except:
    st.warning("⚠️ Map error")

# ==============================
# MAP
# ==============================
st.subheader("🚆 Live Subway Map")

placeholder = st.empty()

def get_color(delay):
    if delay < 120:
        return [0, 200, 0]   # green
    elif delay < 300:
        return [255, 165, 0] # orange
    else:
        return [255, 0, 0]   # red

if not map_df.empty:

    st.success("🚆 Live trains updating...")

    for i in range(20):

        sample = map_df.sample(min(200, len(map_df))).copy()

        # simulate movement
        sample['stop_lat'] += np.random.uniform(-0.001, 0.001, len(sample))
        sample['stop_lon'] += np.random.uniform(-0.001, 0.001, len(sample))

        # color coding
        if 'delay' in sample.columns:
            sample['color'] = sample['delay'].apply(get_color)
        else:
            sample['color'] = [[255, 0, 0] for _ in range(len(sample))]

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=sample,
            get_position='[stop_lon, stop_lat]',
            get_color='color',
            get_radius=120,
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0060,
            zoom=10,
            pitch=40
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Delay: {delay} sec\nStop: {stop_id}"}
        )

        placeholder.pydeck_chart(deck)

        time.sleep(1)

else:
    st.info("📍 Map not available")

# ==============================
# LEGEND
# ==============================
st.subheader("🎨 Delay Legend")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("🟢 Low Delay (<120 sec)")
with col2:
    st.markdown("🟠 Medium Delay (120–300 sec)")
with col3:
    st.markdown("🔴 High Delay (>300 sec)")

# ==============================
# INFO
# ==============================
st.info("""
This system predicts subway delays using ensemble machine learning.

✔ Multiple models combined  
✔ Real-time simulation  
✔ Interactive geographic visualization  
""")
