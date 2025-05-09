import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import re

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
data["road_type"] = data["on_street_name"].apply(classify_road)
data = compute_weights(data)
data = compute_severity_score(data)

@st.cache_data
def load_d():
    return data

__all__ = [
  "data"
]
