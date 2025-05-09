import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import shap
from Preprocessing import load_d

# --- Title and Description ---
st.set_page_config(layout="wide")
st.title("🚗 Motor Vehicle Collisions in New York City")
st.markdown("A Streamlit dashboard to explore and analyze motor vehicle collisions across NYC using geospatial and machine learning tools.")

# --- Load Data ---
data = load_d()
original_data = data.copy()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
injured_people = st.sidebar.slider("Min Injured Persons", 0, 19, 0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 0)

# --- Preprocessing Summary ---
with st.expander("🔍 Preprocessing Summary"):
    st.info("""
    - Missing boroughs imputed based on nearby crash locations.
    - `severity_score` combines injury and fatality counts.
    - Engineered weights for roads, vehicles, and contributing factors.
    - Extracted datetime features for temporal insights.
    """)

# --- Crash Overview ---
with st.expander("🧭 Crash Severity & Location Insights"):
    st.subheader("Crash Counts by Road Type")
    road_counts = data['road_type'].value_counts().reset_index()
    road_counts.columns = ['road_type', 'count']
    st.bar_chart(road_counts.set_index('road_type'))

    st.subheader("Top 10 Streets by Average Severity")
    street_sev = (
        data.groupby('on_street_name')['severity_score']
        .mean().sort_values(ascending=False).head(10)
    )
    st.table(street_sev.reset_index().rename(columns={'severity_score':'avg_severity'}))

    st.subheader("High-Severity Crash Hotspots")
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

# --- NLP-Based Contributing Factor Analysis ---
@st.cache_data
def nlp_contributing_factors(df):
    factor_cols = [f'contributing_factor_vehicle_{i}' for i in range(1,6)]
    df['year'] = df['date/time'].dt.year
    factors_long = df[factor_cols].fillna('Unspecified').melt(value_name='factor').query("factor not in ('', 'Unspecified')")
    factor_counts = factors_long['factor'].value_counts().reset_index().rename(columns={'index':'factor','factor':'count'})
    top20 = factor_counts.head(20)

    trend = df[['year'] + factor_cols].fillna('Unspecified').melt(id_vars=['year'], value_name='factor').query("factor != ''")
    trend = trend.groupby(['year','factor']).size().reset_index(name='count')
    trend_top = trend[trend['factor'].isin(top20['factor'])]

    docs = df[factor_cols].fillna('').agg(' '.join, axis=1)
    vectorizer = CountVectorizer(stop_words='english', min_df=50)
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = [(i+1, [terms[i] for i in comp.argsort()[-10:][::-1]]) for i, comp in enumerate(lda.components_)]
    return factor_counts, top20, trend_top, topics

with st.expander("🧠 NLP-Based Contributing Factor Analysis"):
    st.subheader("Top 20 Contributing Factors")
    factor_counts, top20, trend_top, topics = nlp_contributing_factors(data)
    st.plotly_chart(px.bar(top20, x='factor', y='count', title='Top 20 Contributing Factors').update_layout(xaxis_tickangle=45))
    st.plotly_chart(px.line(trend_top, x='year', y='count', color='factor', title='Yearly Trend of Top Factors'))
    st.subheader("LDA Topic Modeling of Factors")
    for tid, terms in topics:
        st.write(f"**Topic {tid}:**", ", ".join(terms))

# --- Vehicle & Factor Clustering ---
@st.cache_data
def compute_clusters(df, n_clusters=5):
    feat_cols = [f'vehicle_type_{i}' for i in range(1,6)] + [f'contributing_factor_vehicle_{i}' for i in range(1,6)]
    docs = df[feat_cols].fillna('').agg(' '.join, axis=1)
    vec = CountVectorizer(stop_words='english', min_df=50)
    X = vec.fit_transform(docs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    df['cluster'] = labels
    terms = vec.get_feature_names_out()
    cluster_terms = [(i, [terms[j] for j in center.argsort()[-10:][::-1]]) for i, center in enumerate(kmeans.cluster_centers_)]
    return df, cluster_terms

with st.expander("🔬 Crash Clusters by Vehicle & Factor Profiles"):
    st.subheader("Cluster Sizes and Top Terms")
    data, cluster_terms = compute_clusters(data)
    st.write(data['cluster'].value_counts().sort_index())
    for cid, terms in cluster_terms:
        st.write(f"**Cluster {cid}:**", ", ".join(terms))

    st.subheader("Geographic Cluster Map")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude":data['latitude'].mean(),"longitude":data['longitude'].mean(),"zoom":10},
        layers=[pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position=["longitude","latitude"],
            get_fill_color=["cluster * 50 % 255", "cluster * 80 % 255", "cluster * 120 % 255"],
            get_radius=20,
            pickable=True
        )],
        tooltip={"text": "Cluster: {cluster}"}
    ))

# --- Time of Day & Injury Map ---
with st.expander("🕓 Time-of-Day & Injury Mapping"):
    st.subheader("Crashes by Time of Day & Road Type")
    data['hour'] = data['date/time'].dt.hour
    time_road_df = data.groupby(['hour', 'road_type']).size().reset_index(name='crash_count')
    fig = px.line(time_road_df, x='hour', y='crash_count', color='road_type', title='Hourly Crash Frequency by Road Type')
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig)

    st.subheader("Where Are the Most People Injured?")
    st.map(original_data.query("injured_persons >= @injured_people")[["latitude","longitude"]].dropna())

# --- Dangerous Streets ---
st.subheader("🚨 Top 5 Dangerous Streets by Affected Group")
select = st.selectbox("Affected Group", ["Pedestrians", "Cyclists", "Motorists"])
col = f"injured_{select.lower()}"
st.write(original_data.query(f"{col} >= 1")[["on_street_name", col]].sort_values(by=col, ascending=False).dropna().head(5))

# --- Hourly Hexbin Map ---
st.subheader("🗺️ Collision Map for Hour %i:00 to %i:00" % (hour, (hour+1)%24))
hour_data = original_data[original_data['date/time'].dt.hour == hour]
mid = (hour_data['latitude'].mean(), hour_data['longitude'].mean())
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={"latitude": mid[0], "longitude": mid[1], "zoom": 11, "pitch": 50},
    layers=[pdk.Layer(
        "HexagonLayer",
        data=hour_data[['date/time','latitude','longitude']],
        get_position=["longitude", "latitude"],
        radius=100,
        extruded=True,
        pickable=True,
        elevation_scale=4,
        elevation_range=[0, 1000],
    )]
))

# --- Minute Breakdown ---
st.subheader("📊 Minute-by-Minute Breakdown (%i:00 - %i:00)" % (hour, (hour+1)%24))
hist = np.histogram(hour_data['date/time'].dt.minute, bins=60, range=(0,60))[0]
fig = px.bar(pd.DataFrame({'minute': range(60), 'crashes': hist}), x='minute', y='crashes')
st.plotly_chart(fig)

# --- Predictive Modeling & SHAP ---
with st.expander("📈 Predictive Modeling: High Severity Classifier"):
    threshold = data['severity_score'].quantile(0.9)
    data['high_severity'] = (data['severity_score'] >= threshold).astype(int)

    X = data[['hour', 'road_weight', 'vehicle_weight', 'factor_weight']]
    X = pd.concat([X, pd.get_dummies(data[['road_type', 'borough']], drop_first=True)], axis=1)
    y = data['high_severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**AUC:** {roc_auc_score(y_test, y_proba):.2f}")

    st.subheader("🔍 SHAP Feature Importance")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals_pos = shap_vals[1]
    else:
        shap_vals_pos = shap_vals
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    order = np.argsort(mean_abs_shap)
    feature_names = X_test.columns.to_list()
    fig, ax = plt.subplots(figsize=(6, len(order) * 0.3))
    ax.barh([feature_names[i] for i in order], mean_abs_shap[order])
    ax.set_title("Mean |SHAP Value| per Feature")
    st.pyplot(fig)

# --- Final Insights ---
with st.expander("🧠 Insights & Recommendations"):
    st.markdown("### Key Findings")
    st.markdown("""
    - Expressways dominate crash volume, especially at rush hours.
    - "Following Too Closely" is a leading factor but often goes unspecified.
    - Bridges carry disproportionate high-severity crash risk.
    """)

    st.markdown("### Recommendations")
    st.markdown("""
    - Use dynamic signage on expressways during 5–8 PM to alert drivers.
    - Deploy rumble strips on Staten Island expressways.
    - Improve reporting precision to reduce “Unspecified” labels.
    - Audit bridges for winter conditions and enhance surface safety.
    """)

# --- Optional Raw Data Display ---
if st.checkbox("Show Raw Data"):
    st.write(original_data)
