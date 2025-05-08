import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import re

st.title("Motor Vehicle Collisions in New York City")
st.markdown("This application is a streamlit dashboard that can be used "
"to analyze motor vehicle collisions in NYC")

@st.cache_data(persist=True)
def load_data(nrows):
    data = pd.read_csv("Data/Motor_Vehicle_Collisions_-_Crashes.csv", nrows = nrows, parse_dates=[['CRASH_DATE','CRASH_TIME']])
    data.dropna(subset=['LATITUDE','LONGITUDE'], inplace = True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace = True)
    data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace= True)
    return data

@st.cache_data
def load_borough_boundaries():
    # 1) Read the CSV
    df = pd.read_csv(
        "Data/Borough Boundaries.csv",
        dtype=str
    )
    # 2) Parse the_geom WKT into shapely geometries
    df["geometry"] = df["the_geom"].apply(wkt.loads)
    # 3) Build a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df[["BoroName", "geometry"]],
        crs="EPSG:4326",
        geometry="geometry"
    )
    # Rename to match your collisions DF
    gdf = gdf.rename(columns={"BoroName": "borough_from_geom"})
    return gdf

@st.cache_data
def load_and_impute(df):
    # 1) Load borough polygons
    bnds = load_borough_boundaries()
    # 2) Build crash GeoDataFrame for those with coords
    crashes = df.dropna(subset=["latitude", "longitude"]).copy()
    crashes = gpd.GeoDataFrame(
        crashes,
        geometry=[Point(xy) for xy in zip(crashes.longitude, crashes.latitude)],
        crs="EPSG:4326"
    )
    # 3) Spatial join → get borough_from_geom
    joined = gpd.sjoin(crashes, bnds, how="left", predicate="within")
    # 4) Impute only where original borough is missing
    df.loc[joined.index, "borough"] = df.loc[joined.index, "borough"].fillna(
        joined["borough_from_geom"]
    )
    return df

# --- 3. SEVERITY INDEX & FEATURE WEIGHTING ---
@st.cache_data
def compute_weights(df):
    # base severity
    df['base_severity'] = 5*df['killed_persons'] + 1*df['injured_persons']
    # vehicle weights
    veh_stats = (
        df.groupby('vehicle_type_1')['base_severity']
          .mean()
          .reset_index(name='avg_severity')
    )
    mn, mx = veh_stats['avg_severity'].min(), veh_stats['avg_severity'].max()
    veh_stats['vehicle_weight'] = (veh_stats['avg_severity'] - mn) / (mx - mn) * 3
    vehicle_map = dict(zip(veh_stats['vehicle_type_1'], veh_stats['vehicle_weight']))
    df['vehicle_weight'] = df['vehicle_type_1'].map(vehicle_map).fillna(0)
    # factor weights
    fac_stats = (
        df.groupby('contributing_factor_vehicle_1')['base_severity']
          .mean()
          .reset_index(name='avg_severity')
    )
    mn_f, mx_f = fac_stats['avg_severity'].min(), fac_stats['avg_severity'].max()
    fac_stats['factor_weight'] = (fac_stats['avg_severity'] - mn_f) / (mx_f - mn_f) * 3
    factor_map = dict(zip(fac_stats['contributing_factor_vehicle_1'], fac_stats['factor_weight']))
    df['factor_weight'] = df['contributing_factor_vehicle_1'].map(factor_map).fillna(0)
    return df

@st.cache_data
def compute_severity_score(df):
    # ensure road type exists
    df['road_type'] = df['on_street_name'].apply(classify_road)
    # map road type weight
    rt_map = {'highway':3, 'bridge':2, 'local_street':0}
    df['road_weight'] = df['road_type'].map(rt_map)
    # compute composite score
    df['severity_score'] = (
        df['base_severity']
      + df['road_weight']
      + df['vehicle_weight']
      + df['factor_weight']
    )
    return df

data = load_data(100000)
data = load_and_impute(data) # fills missing boroughs
data = compute_weights(data)
data = compute_severity_score(data)

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
if st.checkbox("Show Raw Data", False):
    st.write(data)

# Creating a new attribute "road_type"
# 1) Compile your patterns once
highway_rx = re.compile(r"\b(EXPY|EXPRESSWAY|PKWY|PARKWAY|TPKE|TURNPIKE)\b", flags=re.IGNORECASE)
bridge_rx  = re.compile(r"\b(BRIDGE)\b",                                   flags=re.IGNORECASE)

def classify_road(name):
    """
    - highway if it contains any expressway/parkway/turnpike token
    - bridge  if it contains 'bridge'
    - else     local_street
    """
    if not isinstance(name, str):
        return "local_street"   # fallback if name is missing
    if highway_rx.search(name):
        return "highway"
    if bridge_rx.search(name):
        return "bridge"
    # everything else → local street
    return "local_street"

# 2) Apply to your DataFrame
data["road_type"] = data["on_street_name"].apply(classify_road)

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

if st.checkbox("Show Raw Data", False):
    st.subheader('Raw Data')
    st.write(data)
