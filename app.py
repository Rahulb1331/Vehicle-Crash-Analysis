import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import re
from Preprocessing import load_d


st.title("Motor Vehicle Collisions in New York City")
st.markdown("This application is a streamlit dashboard that can be used "
"to analyze motor vehicle collisions in NYC")

data = load_d()

with st.expander("Show Additional"):
    # road type distribution
    st.header("Crash Counts by Road Type")
    pc = data['road_type'].value_counts().reset_index()
    pc.columns = ['road_type','count']
    st.bar_chart(pc.set_index('road_type'))

    # severity ranking by street
    st.header("Top 10 Streets by Average Severity")
    street_sev = (
        data.groupby('on_street_name')['severity_score']
            .mean()
            .sort_values(ascending=False)
            .head(10)
    )
    st.table(street_sev.reset_index().rename(columns={0:'avg_severity'}))

    # high-severity heatmap
    st.header("High-Severity Crash Hotspots")
    threshold = data['severity_score'].quantile(0.9)
    high_sev = data[data['severity_score'] >= threshold]
    mid = (high_sev['latitude'].mean(), high_sev['longitude'].mean())

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude":mid[0], "longitude":mid[1], "zoom":10, "pitch":45},
        layers=[
            pdk.Layer(
                "HeatmapLayer",
                data=high_sev[['latitude','longitude','severity_score']],
                get_position=['longitude','latitude'],
                get_weight='severity_score',
                radiusPixels=60,
            )
        ]
    ))

    # factor word cloud example (needs additional implementation)
    st.header("Top Contributing Factors by Weight")
    fac_stats = (
        data.groupby('contributing_factor_vehicle_1')['factor_weight']
            .mean()
            .sort_values(ascending=False)
            .head(10)
    )
    st.bar_chart(fac_stats)

    # show raw data toggle
if st.checkbox("Show Raw Data", False, key = "one"):
    st.write(data)



st.dataframe(data.head(10))
original_data = data

st.header("Where are the most people injured in NYC?")
injured_people = st.slider("Number of injured persons in vehicle collision", 0, 19)
st.map(data.query("injured_persons >= @injured_people")[["latitude","longitude"]].dropna(how = 'any'))

st.subheader("Borough Imputation Summary")
counts = data["borough"].isna().value_counts()
st.write(pd.DataFrame({
    "Missing Borough?": counts.index.map({True:"Yes",False:"No"}),
    "Count": counts.values
}))


st.header("How many collisions occur during a given time of day?")
hour = st.slider("Hour to look at", 0, 23)
data = data[data['date/time'].dt.hour== hour]

st.markdown("Vehicle collisions between hour %i:00 and %i:00" %(hour, (hour + 1) % 24))
midpoint = (np.average(data['latitude']),np.average(data['longitude']))

st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
    "latitude" : midpoint[0],
    "longitude" : midpoint[1],
    "zoom": 11,
    "pitch":50
    },
    layers = [
        pdk.Layer(
        "HexagonLayer",
        data = data[['date/time','latitude','longitude']],
        get_position = ['longitude','latitude'],
        radius = 100,
        extruded = True,
        pickable = True,
        elevation_scale = 4,
        elevation_range = [0,1000],
        ),
    ],
))

st.subheader("Breakdown by minute between %i:00 and %i:00"%(hour,(hour+1)%24))
filtered = data[
    (data['date/time'].dt.hour >= hour) & (data['date/time'].dt.hour <(hour + 1))
]

hist = np.histogram(filtered['date/time'].dt.minute, bins = 60, range = (0,60))[0]
chart_data = pd.DataFrame({'minute': range(60), 'crashes':hist})
fig = px.bar(chart_data,x = 'minute', y='crashes', hover_data=['minute', 'crashes'], height = 400)
st.write(fig)

st.header("Top 5 dangerous streets by affected group")
select= st.selectbox('Affected type of people', ['Pedestrians', 'Cyclists', 'Motorists'])

if select == 'Pedestrians':
    st.write(original_data.query("injured_pedestrians >= 1")[["on_street_name", "injured_pedestrians"]].sort_values(by =['injured_pedestrians'], ascending  = False).dropna(how='any')[:5])

elif select == "Cyclists":
    st.write(original_data.query("injured_cyclists >= 1")[["on_street_name", "injured_cyclists"]].sort_values(by =['injured_cyclists'], ascending  = False).dropna(how='any')[:5])

else:
    st.write(original_data.query("injured_motorists >= 1")[["on_street_name", "injured_motorists"]].sort_values(by =['injured_motorists'], ascending  = False).dropna(how='any')[:5])

if st.checkbox("Show Raw Data", False, key = "2"):
    st.subheader('Raw Data')
    st.write(data)
