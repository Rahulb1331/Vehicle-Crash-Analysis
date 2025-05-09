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

## Doing NLP Based contributing factor analysis

# --- 4. NLP-BASED CONTRIBUTING FACTOR ANALYSIS ---
@st.cache_data
def nlp_contributing_factors(df):
    factor_cols = [f'contributing_factor_vehicle_{i}' for i in range(1,6)]
    # Melt into long format
    factors_long = (
        df[factor_cols]
          .fillna('Unspecified')
          .melt(value_name='factor')
          .query("factor != ''")
    )
    # Count factor frequencies
    factor_counts = (
        factors_long['factor']
          .value_counts()
          .reset_index()
          .rename(columns={'index':'factor','0':'count'})
    )
    top20 = factor_counts.head(20)
    # Time-trend per factor
    df['year'] = df['date/time'].dt.year
    factors_trend = (
        df[['year'] + factor_cols]
          .fillna('Unspecified')
          .melt(id_vars=['year'], value_name='factor')
          .query("factor != ''")
    )
    trend = (
        factors_trend
          .groupby(['year','factor'])
          .size()
          .reset_index(name='count')
    )
    top_factors = top20['factor']
    trend_top = trend[trend['factor'].isin(top_factors)]
    return factor_counts, top20, trend_top

with st.expander("Show NLP based contributing factor analysis"):
    # NLP-based contributing factor analysis
    st.header("Contributing Factor NLP Analysis")
    factor_counts, top20, trend_top = nlp_contributing_factors(data)
    # Top 20 factors bar chart
    fig_factors = px.bar(
        top20,
        x='factor', y='count',
        title='Top 20 Contributing Factors',
        labels={'factor':'Contributing Factor','count':'Crash Count'},
        height=500
    )
    fig_factors.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_factors)
    # Yearly trend line chart
    fig_trend = px.line(
        trend_top,
        x='year', y='count', color='factor',
        title='Yearly Trend of Top 20 Contributing Factors'
    )
    st.plotly_chart(fig_trend)

    # Topic Modeling
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(data['factors_clean'])
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    topics = lda.fit_transform(X)

    # Show top words per topic
    for idx, comp in enumerate(lda.components_):
        words = [vec.get_feature_names_out()[i] for i in comp.argsort()[-10:]]
        st.write(f"Topic {idx}: " + ", ".join(words))


# show raw data toggle
if st.checkbox("Show Raw Data", False, key = "one"):
    st.write(data)

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
