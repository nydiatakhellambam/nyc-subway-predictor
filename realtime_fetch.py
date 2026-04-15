import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2

FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"

def fetch_realtime_data():
    response = requests.get(FEED_URL)

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)

    rows = []

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            trip = entity.trip_update

            for stop_time in trip.stop_time_update:
                if stop_time.HasField("arrival"):
                    rows.append({
                        "trip_id": trip.trip.trip_id,
                        "stop_id": stop_time.stop_id,
                        "arrival_time": stop_time.arrival.time
                    })

    return pd.DataFrame(rows)