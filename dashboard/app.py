# from pathlib import Path

# import geopandas as gpd
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI",
#     layout="wide",
# )

# st.title("RainGuardAI - Autonomous Flood Alert System")
# st.markdown(
#     "AI-driven flood prediction, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # ANN PROXY FUNCTION (TensorFlow-Free)
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     """
#     ANN-like risk approximation.
#     This replaces TensorFlow model ONLY in dashboard for stability.
#     """
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)

# # ----------------------------------
# # GEO HELPERS
# # ----------------------------------
# def gdf_to_latlon(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
#     if gdf.empty:
#         return pd.DataFrame(columns=["lat", "lon"])

#     if gdf.crs is None:
#         gdf = gdf.set_crs(epsg=4326)
#     else:
#         gdf = gdf.to_crs(epsg=4326)

#     centroids = gdf.geometry.centroid
#     return pd.DataFrame({"lat": centroids.y, "lon": centroids.x})

# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR INPUTS (LIVE SIMULATION)
# # ----------------------------------
# st.sidebar.header("Live Environmental Inputs")

# rainfall = st.sidebar.slider(
#     "Rainfall Intensity (mm)", 0, 300, 120
# )
# soil_moisture = st.sidebar.slider(
#     "Soil Saturation Index", 0.0, 1.0, 0.65
# )
# elevation_risk = st.sidebar.slider(
#     "Elevation Risk Factor", 0.0, 1.0, 0.70
# )

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
# else:
#     risk_level = "LOW"

# st.write(f"**Risk Level:** `{risk_level}`")

# # ----------------------------------
# # RL ALERT DECISION
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# alert_decision = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# if alert_decision == 1:
#     st.error("SEND ALERT - Immediate Response Required")
# else:
#     st.success("NO ALERT - Continue Monitoring")

# # ----------------------------------
# # NDMA-BASED EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write(
#         """
#         **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#         and low-lying terrain.

#         **NDMA Recommended Actions:**
#         - Activate emergency response teams
#         - Prepare evacuation shelters
#         - Continuous drainage monitoring
#         - Issue early public alerts
#         """
#     )

# elif risk_level == "MEDIUM":
#     st.write(
#         """
#         **Flood risk is MODERATE**.

#         **Recommended Actions:**
#         - Monitor rainfall trends
#         - Keep emergency teams on standby
#         - Inspect drainage infrastructure
#         """
#     )

# else:
#     st.write(
#         """
#         **Flood risk is LOW**.

#         **Recommended Actions:**
#         - Routine monitoring
#         - No immediate emergency actions required
#         """
#     )

# # ----------------------------------
# # MAP VISUALIZATION
# # ----------------------------------
# st.subheader("Flood Impact Visualization")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### Low-Lying Flood-Prone Zones")
#     low_zones = gpd.read_file(LOW_ZONES_PATH)
#     st.map(gdf_to_latlon(low_zones))

# with col2:
#     st.markdown("### Critical Infrastructure at Risk")
#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH)
#     st.map(gdf_to_latlon(impact_zones))

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + GenAI + Reinforcement Learning"
# )

# from pathlib import Path

# import geopandas as gpd
# import joblib
# import numpy as np
# import pandas as pd
# import pydeck as pdk
# import streamlit as st

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI",
#     layout="wide",
# )

# st.title("RainGuardAI - Autonomous Flood Alert System")
# st.markdown(
#     "AI-driven flood prediction, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # ANN PROXY FUNCTION (TensorFlow-Free)
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)

# # ----------------------------------
# # COLOR LOGIC
# # ----------------------------------
# def risk_color(risk):
#     if risk >= 0.75:
#         return [255, 0, 0]      # Red
#     elif risk >= 0.5:
#         return [255, 255, 0]    # Yellow
#     else:
#         return [0, 200, 0]      # Green

# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR INPUTS
# # ----------------------------------
# st.sidebar.header("Live Environmental Inputs")

# rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
# soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
# elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
# else:
#     risk_level = "LOW"

# st.write(f"**Risk Level:** `{risk_level}`")

# # ----------------------------------
# # RL ALERT DECISION
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# alert_decision = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# if alert_decision == 1:
#     st.error("SEND ALERT - Immediate Response Required")
# else:
#     st.success("NO ALERT - Continue Monitoring")

# # ----------------------------------
# # NDMA EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write("""
#     **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#     and low-lying terrain.

#     **NDMA Recommended Actions:**
#     - Activate emergency response teams
#     - Prepare evacuation shelters
#     - Continuous drainage monitoring
#     - Issue early public alerts
#     """)

# elif risk_level == "MEDIUM":
#     st.write("""
#     **Flood risk is MODERATE**.

#     **Recommended Actions:**
#     - Monitor rainfall trends
#     - Keep emergency teams on standby
#     - Inspect drainage infrastructure
#     """)

# else:
#     st.write("""
#     **Flood risk is LOW**.

#     **Recommended Actions:**
#     - Routine monitoring
#     - No immediate emergency actions required
#     """)

# # ----------------------------------
# # MAP VISUALIZATION (PYDECK)
# # ----------------------------------
# st.subheader("Flood Impact Visualization")

# color = risk_color(risk_prob)

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### Low-Lying Flood-Prone Zones")

#     low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
#     low_zones["lat"] = low_zones.geometry.centroid.y
#     low_zones["lon"] = low_zones.geometry.centroid.x
#     low_zones["color"] = [color] * len(low_zones)

#     low_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=low_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=120,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[low_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#         )
#     )

# with col2:
#     st.markdown("### Critical Infrastructure at Risk")

#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
#     impact_zones["lat"] = impact_zones.geometry.centroid.y
#     impact_zones["lon"] = impact_zones.geometry.centroid.x

#     infra_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=impact_zones,
#         get_position="[lon, lat]",
#         get_color=[255, 0, 0],
#         get_radius=180,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[infra_layer],
#             tooltip={"text": "{infra_type}"},
#         )
#     )

# # ----------------------------------
# # INFRASTRUCTURE SUMMARY
# # ----------------------------------
# st.subheader("🚨 Critical Infrastructure Summary")

# if "infra_type" in impact_zones.columns:
#     infra_counts = impact_zones["infra_type"].value_counts()
#     st.table(infra_counts)
#     st.metric("Total Critical Infrastructure Affected", int(infra_counts.sum()))
# else:
#     st.warning("Infrastructure type data not available")

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + GenAI + Reinforcement Learning"
# )

# from pathlib import Path

# import geopandas as gpd
# import joblib
# import pandas as pd
# import pydeck as pdk
# import streamlit as st

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI",
#     layout="wide",
# )

# st.title("RainGuardAI - Autonomous Flood Alert System")
# st.markdown(
#     "AI-driven flood prediction, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # ANN PROXY FUNCTION (TensorFlow-Free)
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)

# # ----------------------------------
# # COLOR LOGIC
# # ----------------------------------
# def risk_color(risk):
#     if risk >= 0.75:
#         return [255, 0, 0]      # Red
#     elif risk >= 0.5:
#         return [255, 255, 0]    # Yellow
#     else:
#         return [0, 200, 0]      # Green

# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR INPUTS
# # ----------------------------------
# st.sidebar.header("Live Environmental Inputs")

# rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
# soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
# elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)
# color = risk_color(risk_prob)

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
# else:
#     risk_level = "LOW"

# st.write(f"**Risk Level:** `{risk_level}`")

# # ----------------------------------
# # RL ALERT DECISION
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# alert_decision = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# if alert_decision == 1:
#     st.error("SEND ALERT - Immediate Response Required")
# else:
#     st.success("NO ALERT - Continue Monitoring")

# # ----------------------------------
# # NDMA EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write("""
#     **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#     and low-lying terrain.

#     **NDMA Recommended Actions:**
#     - Activate emergency response teams
#     - Prepare evacuation shelters
#     - Continuous drainage monitoring
#     - Issue early public alerts
#     """)
# elif risk_level == "MEDIUM":
#     st.write("""
#     **Flood risk is MODERATE**.

#     **Recommended Actions:**
#     - Monitor rainfall trends
#     - Keep emergency teams on standby
#     - Inspect drainage infrastructure
#     """)
# else:
#     st.write("""
#     **Flood risk is LOW**.

#     **Recommended Actions:**
#     - Routine monitoring
#     - No immediate emergency actions required
#     """)

# # ----------------------------------
# # MAP VISUALIZATION (PYDECK)
# # ----------------------------------
# st.subheader("Flood Impact Visualization")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### Low-Lying Flood-Prone Zones")

#     low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
#     low_zones["lat"] = low_zones.geometry.centroid.y
#     low_zones["lon"] = low_zones.geometry.centroid.x
#     low_zones["color"] = [color] * len(low_zones)

#     low_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=low_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=120,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[low_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#         )
#     )

# with col2:
#     st.markdown("### Critical Infrastructure at Risk")

#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
#     impact_zones["lat"] = impact_zones.geometry.centroid.y
#     impact_zones["lon"] = impact_zones.geometry.centroid.x
#     impact_zones["color"] = [color] * len(impact_zones)

#     infra_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=impact_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=180,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[infra_layer],
#             tooltip={"text": "{name}"} if "name" in impact_zones.columns else None,
#         )
#     )

# # ----------------------------------
# # INFRASTRUCTURE SUMMARY (AUTO-DETECT)
# # ----------------------------------
# st.subheader("🚨 Critical Infrastructure Summary")

# possible_cols = ["infra_type", "type", "category", "layer", "name", "source"]
# infra_col = next((c for c in possible_cols if c in impact_zones.columns), None)

# if infra_col:
#     infra_counts = impact_zones[infra_col].value_counts()
#     st.table(infra_counts)
#     st.metric("Total Critical Infrastructure Affected", int(infra_counts.sum()))
# else:
#     st.warning(f"No infrastructure classification column found.\nAvailable columns: {list(impact_zones.columns)}")

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + GenAI + Reinforcement Learning"
# )



# from pathlib import Path

# import geopandas as gpd
# import joblib
# import pandas as pd
# import pydeck as pdk
# import streamlit as st

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI",
#     layout="wide",
# )

# st.title("RainGuardAI - Autonomous Flood Alert System")
# st.markdown(
#     "AI-driven flood prediction, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # ANN PROXY FUNCTION (TensorFlow-Free)
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)

# # ----------------------------------
# # COLOR LOGIC
# # ----------------------------------
# def risk_color(risk):
#     if risk >= 0.75:
#         return [255, 0, 0]      # Red
#     elif risk >= 0.5:
#         return [255, 255, 0]    # Yellow
#     else:
#         return [0, 200, 0]      # Green

# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR INPUTS
# # ----------------------------------
# st.sidebar.header("Live Environmental Inputs")

# rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
# soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
# elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)
# color = risk_color(risk_prob)

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
# else:
#     risk_level = "LOW"

# st.write(f"**Risk Level:** `{risk_level}`")

# # ----------------------------------
# # RL ALERT DECISION
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# alert_decision = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# if alert_decision == 1:
#     st.error("SEND ALERT - Immediate Response Required")
# else:
#     st.success("NO ALERT - Continue Monitoring")

# # ----------------------------------
# # NDMA EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write("""
#     **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#     and low-lying terrain.

#     **NDMA Recommended Actions:**
#     - Activate emergency response teams
#     - Prepare evacuation shelters
#     - Continuous drainage monitoring
#     - Issue early public alerts
#     """)
# elif risk_level == "MEDIUM":
#     st.write("""
#     **Flood risk is MODERATE**.

#     **Recommended Actions:**
#     - Monitor rainfall trends
#     - Keep emergency teams on standby
#     - Inspect drainage infrastructure
#     """)
# else:
#     st.write("""
#     **Flood risk is LOW**.

#     **Recommended Actions:**
#     - Routine monitoring
#     - No immediate emergency actions required
#     """)

# # ----------------------------------
# # MAP VISUALIZATION (PYDECK)
# # ----------------------------------
# st.subheader("Flood Impact Visualization")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### Low-Lying Flood-Prone Zones")

#     low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
#     low_zones["lat"] = low_zones.geometry.centroid.y
#     low_zones["lon"] = low_zones.geometry.centroid.x
#     low_zones["color"] = [color] * len(low_zones)

#     low_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=low_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=120,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[low_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#         )
#     )

# with col2:
#     st.markdown("### Critical Infrastructure at Risk")

#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
#     impact_zones["lat"] = impact_zones.geometry.centroid.y
#     impact_zones["lon"] = impact_zones.geometry.centroid.x
#     impact_zones["color"] = [color] * len(impact_zones)

#     infra_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=impact_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=180,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[infra_layer],
#             tooltip={"text": "{name}"} if "name" in impact_zones.columns else None,
#         )
#     )

# # ----------------------------------
# # INFRASTRUCTURE SUMMARY
# # ----------------------------------
# st.subheader("🚨 Critical Infrastructure Summary")

# possible_cols = ["infrastructure_type", "infra_type", "amenity", "type", "category", "layer", "name", "source"]
# infra_col = next((c for c in possible_cols if c in impact_zones.columns), None)

# if infra_col:
#     infra_counts = impact_zones[infra_col].value_counts()
#     st.table(infra_counts)
#     total = int(infra_counts.sum())
# else:
#     total = len(impact_zones)

# st.metric("Total Critical Infrastructure Affected", total)

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + GenAI + Reinforcement Learning"
# )


# from pathlib import Path
# import geopandas as gpd
# import joblib
# import pandas as pd
# import pydeck as pdk
# import streamlit as st
# import requests
# from datetime import datetime
# import time

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI - Real-Time",
#     layout="wide",
# )

# st.title("RainGuardAI - Autonomous Flood Alert System (Real-Time)")
# st.markdown(
#     "AI-driven flood prediction with **LIVE** weather data, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # REAL-TIME DATA FETCHING FUNCTIONS
# # ----------------------------------

# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def fetch_openweather_data(lat, lon, api_key):
#     """
#     Fetch real-time weather data from OpenWeatherMap API
#     Free tier: 1000 calls/day
#     """
#     try:
#         url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         # Extract relevant data
#         rainfall = data.get('rain', {}).get('1h', 0)  # mm in last hour
#         humidity = data.get('main', {}).get('humidity', 50) / 100  # Convert to 0-1
#         temp = data.get('main', {}).get('temp', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': data.get('weather', [{}])[0].get('description', 'N/A'),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     except Exception as e:
#         st.error(f"Error fetching OpenWeatherMap data: {e}")
#         return None


# @st.cache_data(ttl=300)
# def fetch_weatherapi_data(lat, lon, api_key):
#     """
#     Fetch real-time weather data from WeatherAPI.com
#     Free tier: 1M calls/month
#     """
#     try:
#         url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         current = data.get('current', {})
#         rainfall = current.get('precip_mm', 0)
#         humidity = current.get('humidity', 50) / 100
#         temp = current.get('temp_c', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': current.get('condition', {}).get('text', 'N/A'),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     except Exception as e:
#         st.error(f"Error fetching WeatherAPI data: {e}")
#         return None


# @st.cache_data(ttl=300)
# def fetch_open_meteo_data(lat, lon):
#     """
#     Fetch real-time weather data from Open-Meteo (FREE, no API key needed!)
#     """
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         current = data.get('current', {})
#         rainfall = current.get('rain', 0) + current.get('precipitation', 0)
#         humidity = current.get('relative_humidity_2m', 50) / 100
#         temp = current.get('temperature_2m', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': 'Real-time data',
#             'timestamp': current.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#         }
#     except Exception as e:
#         st.error(f"Error fetching Open-Meteo data: {e}")
#         return None


# def estimate_soil_moisture(humidity, rainfall, temp):
#     """
#     Estimate soil saturation based on weather conditions
#     More sophisticated models can be added here
#     """
#     # Simple estimation model
#     base_moisture = humidity * 0.5
#     rain_contribution = min(rainfall / 50, 0.4)  # Max 0.4 from rain
#     temp_factor = max(0, (30 - temp) / 30) * 0.1  # Temperature affects evaporation
    
#     soil_moisture = min(base_moisture + rain_contribution + temp_factor, 1.0)
#     return soil_moisture


# def get_elevation_risk(lat, lon):
#     """
#     Get elevation data from Open-Elevation API (FREE)
#     Returns a risk factor based on elevation
#     """
#     try:
#         url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         elevation = data['results'][0]['elevation']  # meters
        
#         # Mumbai is mostly 0-100m, low elevation = higher flood risk
#         if elevation < 5:
#             risk = 0.9
#         elif elevation < 10:
#             risk = 0.7
#         elif elevation < 20:
#             risk = 0.5
#         elif elevation < 50:
#             risk = 0.3
#         else:
#             risk = 0.1
            
#         return risk, elevation
#     except Exception as e:
#         st.warning(f"Could not fetch elevation data: {e}")
#         return 0.7, None  # Default medium-high risk


# # ----------------------------------
# # ANN PROXY FUNCTION
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)


# # ----------------------------------
# # COLOR LOGIC
# # ----------------------------------
# def risk_color(risk):
#     if risk >= 0.75:
#         return [255, 0, 0]      # Red
#     elif risk >= 0.5:
#         return [255, 255, 0]    # Yellow
#     else:
#         return [0, 200, 0]      # Green


# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR - DATA SOURCE SELECTION
# # ----------------------------------
# st.sidebar.header("⚙️ Configuration")

# data_mode = st.sidebar.radio(
#     "Data Source Mode",
#     ["Real-Time API", "Manual Input"],
#     help="Choose between live weather data or manual sliders"
# )

# if data_mode == "Real-Time API":
#     st.sidebar.subheader("API Settings")
    
#     # Location for Mumbai
#     latitude = st.sidebar.number_input("Latitude", value=19.0760, format="%.4f")
#     longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.4f")
    
#     api_source = st.sidebar.selectbox(
#         "Weather Data Provider",
#         ["Open-Meteo (Free, No Key)", "OpenWeatherMap", "WeatherAPI.com"],
#         help="Open-Meteo requires no API key!"
#     )
    
#     api_key = ""
#     if api_source != "Open-Meteo (Free, No Key)":
#         api_key = st.sidebar.text_input(
#             "API Key",
#             type="password",
#             help=f"Get your free API key from {api_source}"
#         )
    
#     # Auto-refresh option
#     auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
#     if auto_refresh:
#         st.sidebar.info("⏱️ Auto-refresh enabled")
    
#     # Manual refresh button
#     refresh_button = st.sidebar.button("🔄 Refresh Data Now")
    
#     # Fetch real-time data
#     if api_source == "Open-Meteo (Free, No Key)":
#         weather_data = fetch_open_meteo_data(latitude, longitude)
#     elif api_source == "OpenWeatherMap" and api_key:
#         weather_data = fetch_openweather_data(latitude, longitude, api_key)
#     elif api_source == "WeatherAPI.com" and api_key:
#         weather_data = fetch_weatherapi_data(latitude, longitude, api_key)
#     else:
#         weather_data = None
#         if api_source != "Open-Meteo (Free, No Key)":
#             st.sidebar.warning("⚠️ Please enter your API key")
    
#     if weather_data:
#         st.sidebar.success(f"✅ Live data updated: {weather_data['timestamp']}")
        
#         # Display real-time values
#         st.sidebar.metric("🌧️ Rainfall (mm/hr)", f"{weather_data['rainfall']:.1f}")
#         st.sidebar.metric("💧 Humidity", f"{weather_data['humidity']*100:.0f}%")
#         st.sidebar.metric("🌡️ Temperature", f"{weather_data['temp']:.1f}°C")
#         st.sidebar.caption(f"Conditions: {weather_data['description']}")
        
#         # Calculate soil moisture and elevation risk
#         soil_moisture = estimate_soil_moisture(
#             weather_data['humidity'],
#             weather_data['rainfall'],
#             weather_data['temp']
#         )
        
#         elevation_risk, elevation = get_elevation_risk(latitude, longitude)
#         if elevation is not None:
#             st.sidebar.metric("📏 Elevation", f"{elevation:.1f}m")
        
#         rainfall = weather_data['rainfall']
        
#     else:
#         # Fallback to default values
#         st.sidebar.error("❌ Could not fetch real-time data. Using default values.")
#         rainfall = 120
#         soil_moisture = 0.65
#         elevation_risk = 0.70

# else:
#     # Manual input mode
#     st.sidebar.subheader("Manual Environmental Inputs")
#     rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
#     soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
#     elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)
# color = risk_color(risk_prob)

# col_a, col_b, col_c = st.columns(3)

# with col_a:
#     st.metric("🌧️ Rainfall", f"{rainfall:.1f} mm/hr")
# with col_b:
#     st.metric("💧 Soil Saturation", f"{soil_moisture:.2f}")
# with col_c:
#     st.metric("📍 Elevation Risk", f"{elevation_risk:.2f}")

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
#     st.error(f"⚠️ **Risk Level:** `{risk_level}`")
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
#     st.warning(f"⚡ **Risk Level:** `{risk_level}`")
# else:
#     risk_level = "LOW"
#     st.success(f"✅ **Risk Level:** `{risk_level}`")

# # ----------------------------------
# # RL ALERT DECISION
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# alert_decision = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# if alert_decision == 1:
#     st.error("🚨 SEND ALERT - Immediate Response Required")
# else:
#     st.success("✅ NO ALERT - Continue Monitoring")

# # ----------------------------------
# # NDMA EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write("""
#     **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#     and low-lying terrain.

#     **NDMA Recommended Actions:**
#     - ⚠️ Activate emergency response teams
#     - 🏠 Prepare evacuation shelters
#     - 🚰 Continuous drainage monitoring
#     - 📢 Issue early public alerts
#     """)
# elif risk_level == "MEDIUM":
#     st.write("""
#     **Flood risk is MODERATE**.

#     **Recommended Actions:**
#     - 📊 Monitor rainfall trends
#     - 👥 Keep emergency teams on standby
#     - 🔧 Inspect drainage infrastructure
#     """)
# else:
#     st.write("""
#     **Flood risk is LOW**.

#     **Recommended Actions:**
#     - 👀 Routine monitoring
#     - ✅ No immediate emergency actions required
#     """)

# # ----------------------------------
# # MAP VISUALIZATION (PYDECK)
# # ----------------------------------
# st.subheader("Flood Impact Visualization")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### Low-Lying Flood-Prone Zones")

#     low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
#     low_zones["lat"] = low_zones.geometry.centroid.y
#     low_zones["lon"] = low_zones.geometry.centroid.x
#     low_zones["color"] = [color] * len(low_zones)

#     low_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=low_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=120,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[low_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#         )
#     )

# with col2:
#     st.markdown("### Critical Infrastructure at Risk")

#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
#     impact_zones["lat"] = impact_zones.geometry.centroid.y
#     impact_zones["lon"] = impact_zones.geometry.centroid.x
#     impact_zones["color"] = [color] * len(impact_zones)

#     infra_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=impact_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=180,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[infra_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#             tooltip={"text": "{name}"} if "name" in impact_zones.columns else None,
#         )
#     )

# # ----------------------------------
# # INFRASTRUCTURE SUMMARY
# # ----------------------------------
# st.subheader("🚨 Critical Infrastructure Summary")

# possible_cols = ["infrastructure_type", "infra_type", "amenity", "type", "category", "layer", "name", "source"]
# infra_col = next((c for c in possible_cols if c in impact_zones.columns), None)

# if infra_col:
#     infra_counts = impact_zones[infra_col].value_counts()
#     st.table(infra_counts)
#     total = int(infra_counts.sum())
# else:
#     total = len(impact_zones)

# st.metric("Total Critical Infrastructure Affected", total)

# # ----------------------------------
# # AUTO-REFRESH LOGIC
# # ----------------------------------
# if data_mode == "Real-Time API" and auto_refresh:
#     time.sleep(300)  # Wait 5 minutes
#     st.rerun()

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + GenAI + Reinforcement Learning | Real-Time Edition"
# )




# from pathlib import Path

# import geopandas as gpd
# import joblib
# import pandas as pd
# import pydeck as pdk
# import streamlit as st
# import requests
# from datetime import datetime
# import time
# try:
#     from twilio.rest import Client
# except ImportError:
#     Client = None

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI",
#     layout="wide",
# )

# st.title("RainGuardAI - Autonomous Flood Alert System (Real-Time)")
# st.markdown(
#     "AI-driven flood prediction with **LIVE** weather data, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # INITIALIZE SESSION STATE
# # ----------------------------------
# if 'alert_sent' not in st.session_state:
#     st.session_state.alert_sent = False
# if 'alert_history' not in st.session_state:
#     st.session_state.alert_history = []
# if 'show_confirmation' not in st.session_state:
#     st.session_state.show_confirmation = False

# # ----------------------------------
# # MOBILE NOTIFICATION FUNCTIONS
# # ----------------------------------

# def send_sms_twilio(phone_number, message, account_sid, auth_token, from_number):
#     """
#     Send SMS via Twilio API
#     Sign up at: https://www.twilio.com/try-twilio (FREE trial)
#     """
#     try:
#         if Client is None:
#             return False, "❌ Install Twilio: pip install twilio"

#         client = Client(account_sid, auth_token)
        
#         sms = client.messages.create(
#             body=message,
#             from_=from_number,
#             to=phone_number
#         )
        
#         return True, f"✅ SMS sent! SID: {sms.sid}"
#     except Exception as e:
#         return False, f"❌ SMS failed: {str(e)}"


# def send_whatsapp_twilio(phone_number, message, account_sid, auth_token, from_number):
#     """
#     Send WhatsApp via Twilio API
#     """
#     try:
#         if Client is None:
#             return False, "❌ Install Twilio: pip install twilio"

#         client = Client(account_sid, auth_token)
        
#         whatsapp = client.messages.create(
#             body=message,
#             from_=f'whatsapp:{from_number}',
#             to=f'whatsapp:{phone_number}'
#         )
        
#         return True, f"✅ WhatsApp sent! SID: {whatsapp.sid}"
#     except Exception as e:
#         return False, f"❌ WhatsApp failed: {str(e)}"


# def send_email_alert(to_email, subject, message, smtp_server, smtp_port, from_email, password):
#     """
#     Send Email Alert via SMTP (Gmail, Outlook, etc.)
#     """
#     try:
#         import smtplib
#         from email.mime.text import MIMEText
#         from email.mime.multipart import MIMEMultipart
        
#         msg = MIMEMultipart()
#         msg['From'] = from_email
#         msg['To'] = to_email
#         msg['Subject'] = subject
        
#         msg.attach(MIMEText(message, 'plain'))
        
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(from_email, password)
#         server.send_message(msg)
#         server.quit()
        
#         return True, "✅ Email sent successfully!"
#     except Exception as e:
#         return False, f"❌ Email failed: {str(e)}"


# def send_telegram_alert(bot_token, chat_id, message):
#     """
#     Send alert via Telegram Bot (FREE)
#     Create bot at: https://t.me/BotFather
#     """
#     try:
#         url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
#         data = {
#             "chat_id": chat_id,
#             "text": message,
#             "parse_mode": "Markdown"
#         }
        
#         response = requests.post(url, data=data, timeout=10)
#         response.raise_for_status()
        
#         return True, "✅ Telegram alert sent!"
#     except Exception as e:
#         return False, f"❌ Telegram failed: {str(e)}"


# def generate_alert_message(risk_level, risk_prob, rainfall, soil_moisture, location="Mumbai"):
#     """
#     Generate detailed alert message based on risk level
#     """
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     if risk_level == "HIGH":
#         message = f"""
# 🚨 URGENT FLOOD ALERT - HIGH RISK 🚨

# Location: {location}
# Time: {timestamp}
# Risk Level: {risk_level}
# Risk Probability: {risk_prob:.2%}

# CURRENT CONDITIONS:
# • Rainfall: {rainfall:.1f} mm/hr
# • Soil Saturation: {soil_moisture:.2%}

# ⚠️ IMMEDIATE ACTIONS REQUIRED:
# 1. Evacuate low-lying areas immediately
# 2. Move to higher ground
# 3. Activate emergency response teams
# 4. Prepare evacuation shelters
# 5. Monitor drainage systems continuously
# 6. Issue public alerts via all channels

# NDMA Guidelines:
# - Do NOT attempt to cross flooded areas
# - Keep emergency contacts ready
# - Listen to local authorities
# - Stock emergency supplies (food, water, medicine)

# Stay Safe! Follow official instructions.

# RainGuardAI Autonomous Flood Alert System
#         """
    
#     elif risk_level == "MEDIUM":
#         message = f"""
# ⚠️ FLOOD ALERT - MODERATE RISK ⚠️

# Location: {location}
# Time: {timestamp}
# Risk Level: {risk_level}
# Risk Probability: {risk_prob:.2%}

# CURRENT CONDITIONS:
# • Rainfall: {rainfall:.1f} mm/hr
# • Soil Saturation: {soil_moisture:.2%}

# 📋 RECOMMENDED ACTIONS:
# 1. Monitor weather updates closely
# 2. Keep emergency teams on standby
# 3. Inspect drainage infrastructure
# 4. Prepare emergency supplies
# 5. Avoid low-lying areas if possible

# NDMA Guidelines:
# - Stay informed about weather conditions
# - Keep emergency kit ready
# - Avoid unnecessary travel
# - Check on vulnerable neighbors

# Stay Alert!

# RainGuardAI Autonomous Flood Alert System
#         """
    
#     else:  # LOW
#         message = f"""
# ✅ FLOOD MONITORING - LOW RISK ✅

# Location: {location}
# Time: {timestamp}
# Risk Level: {risk_level}
# Risk Probability: {risk_prob:.2%}

# CURRENT CONDITIONS:
# • Rainfall: {rainfall:.1f} mm/hr
# • Soil Saturation: {soil_moisture:.2%}

# Status: Routine monitoring in progress
# No immediate emergency actions required

# Continue normal activities.

# RainGuardAI Autonomous Flood Alert System
#         """
    
#     return message.strip()


# # ----------------------------------
# # REAL-TIME DATA FETCHING FUNCTIONS
# # ----------------------------------

# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def fetch_openweather_data(lat, lon, api_key):
#     """Fetch real-time weather data from OpenWeatherMap API"""
#     try:
#         url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         rainfall = data.get('rain', {}).get('1h', 0)
#         humidity = data.get('main', {}).get('humidity', 50) / 100
#         temp = data.get('main', {}).get('temp', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': data.get('weather', [{}])[0].get('description', 'N/A'),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     except Exception as e:
#         st.error(f"Error fetching OpenWeatherMap data: {e}")
#         return None


# @st.cache_data(ttl=300)
# def fetch_weatherapi_data(lat, lon, api_key):
#     """Fetch real-time weather data from WeatherAPI.com"""
#     try:
#         url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         current = data.get('current', {})
#         rainfall = current.get('precip_mm', 0)
#         humidity = current.get('humidity', 50) / 100
#         temp = current.get('temp_c', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': current.get('condition', {}).get('text', 'N/A'),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     except Exception as e:
#         st.error(f"Error fetching WeatherAPI data: {e}")
#         return None


# @st.cache_data(ttl=300)
# def fetch_open_meteo_data(lat, lon):
#     """Fetch real-time weather data from Open-Meteo (FREE, no API key needed!)"""
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         current = data.get('current', {})
#         rainfall = current.get('rain', 0) + current.get('precipitation', 0)
#         humidity = current.get('relative_humidity_2m', 50) / 100
#         temp = current.get('temperature_2m', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': 'Real-time data',
#             'timestamp': current.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#         }
#     except Exception as e:
#         st.error(f"Error fetching Open-Meteo data: {e}")
#         return None


# def estimate_soil_moisture(humidity, rainfall, temp):
#     """Estimate soil saturation based on weather conditions"""
#     base_moisture = humidity * 0.5
#     rain_contribution = min(rainfall / 50, 0.4)
#     temp_factor = max(0, (30 - temp) / 30) * 0.1
    
#     soil_moisture = min(base_moisture + rain_contribution + temp_factor, 1.0)
#     return soil_moisture


# def get_elevation_risk(lat, lon):
#     """Get elevation data from Open-Elevation API (FREE)"""
#     try:
#         url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         elevation = data['results'][0]['elevation']
        
#         if elevation < 5:
#             risk = 0.9
#         elif elevation < 10:
#             risk = 0.7
#         elif elevation < 20:
#             risk = 0.5
#         elif elevation < 50:
#             risk = 0.3
#         else:
#             risk = 0.1
            
#         return risk, elevation
#     except Exception as e:
#         st.warning(f"Could not fetch elevation data: {e}")
#         return 0.7, None


# # ----------------------------------
# # ANN PROXY FUNCTION
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)


# # ----------------------------------
# # COLOR LOGIC
# # ----------------------------------
# def risk_color(risk):
#     if risk >= 0.75:
#         return [255, 0, 0]      # Red
#     elif risk >= 0.5:
#         return [255, 255, 0]    # Yellow
#     else:
#         return [0, 200, 0]      # Green


# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR - DATA SOURCE SELECTION
# # ----------------------------------
# st.sidebar.header("⚙️ Configuration")

# data_mode = st.sidebar.radio(
#     "Data Source Mode",
#     ["Real-Time API", "Manual Input"],
#     help="Choose between live weather data or manual sliders"
# )

# if data_mode == "Real-Time API":
#     st.sidebar.subheader("API Settings")
    
#     latitude = st.sidebar.number_input("Latitude", value=19.0760, format="%.4f")
#     longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.4f")
    
#     api_source = st.sidebar.selectbox(
#         "Weather Data Provider",
#         ["Open-Meteo (Free, No Key)", "OpenWeatherMap", "WeatherAPI.com"],
#         help="Open-Meteo requires no API key!"
#     )
    
#     api_key = ""
#     if api_source != "Open-Meteo (Free, No Key)":
#         api_key = st.sidebar.text_input(
#             "API Key",
#             type="password",
#             help=f"Get your free API key from {api_source}"
#         )
    
#     auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
#     if auto_refresh:
#         st.sidebar.info("⏱️ Auto-refresh enabled")
    
#     refresh_button = st.sidebar.button("🔄 Refresh Data Now")
    
#     # Fetch real-time data
#     if api_source == "Open-Meteo (Free, No Key)":
#         weather_data = fetch_open_meteo_data(latitude, longitude)
#     elif api_source == "OpenWeatherMap" and api_key:
#         weather_data = fetch_openweather_data(latitude, longitude, api_key)
#     elif api_source == "WeatherAPI.com" and api_key:
#         weather_data = fetch_weatherapi_data(latitude, longitude, api_key)
#     else:
#         weather_data = None
#         if api_source != "Open-Meteo (Free, No Key)":
#             st.sidebar.warning("⚠️ Please enter your API key")
    
#     if weather_data:
#         st.sidebar.success(f"✅ Live data updated: {weather_data['timestamp']}")
        
#         st.sidebar.metric("🌧️ Rainfall (mm/hr)", f"{weather_data['rainfall']:.1f}")
#         st.sidebar.metric("💧 Humidity", f"{weather_data['humidity']*100:.0f}%")
#         st.sidebar.metric("🌡️ Temperature", f"{weather_data['temp']:.1f}°C")
#         st.sidebar.caption(f"Conditions: {weather_data['description']}")
        
#         soil_moisture = estimate_soil_moisture(
#             weather_data['humidity'],
#             weather_data['rainfall'],
#             weather_data['temp']
#         )
        
#         elevation_risk, elevation = get_elevation_risk(latitude, longitude)
#         if elevation is not None:
#             st.sidebar.metric("📏 Elevation", f"{elevation:.1f}m")
        
#         rainfall = weather_data['rainfall']
        
#     else:
#         st.sidebar.error("❌ Could not fetch real-time data. Using default values.")
#         rainfall = 120
#         soil_moisture = 0.65
#         elevation_risk = 0.70

# else:
#     st.sidebar.subheader("Manual Environmental Inputs")
#     rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
#     soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
#     elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# # ----------------------------------
# # NOTIFICATION SETTINGS
# # ----------------------------------
# st.sidebar.markdown("---")
# st.sidebar.subheader("📱 Alert Notification Settings")

# notification_enabled = st.sidebar.checkbox("Enable Mobile Alerts", value=False)

# if notification_enabled:
#     notification_method = st.sidebar.selectbox(
#         "Notification Method",
#         ["Telegram (Easiest)", "SMS (Twilio)", "WhatsApp (Twilio)", "Email (SMTP)"]
#     )
    
#     if notification_method == "Telegram (Easiest)":
#         st.sidebar.info("Get Bot Token from @BotFather on Telegram")
#         telegram_bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
#         telegram_chat_id = st.sidebar.text_input("Your Chat ID", help="Get from @userinfobot")
        
#     elif notification_method in ["SMS (Twilio)", "WhatsApp (Twilio)"]:
#         st.sidebar.info("Sign up at twilio.com/try-twilio")
#         twilio_account_sid = st.sidebar.text_input("Twilio Account SID", type="password")
#         twilio_auth_token = st.sidebar.text_input("Twilio Auth Token", type="password")
#         twilio_from_number = st.sidebar.text_input("From Number (e.g., +1234567890)")
#         user_phone_number = st.sidebar.text_input("Your Phone Number (e.g., +919876543210)")
        
#     elif notification_method == "Email (SMTP)":
#         st.sidebar.info("Use Gmail, Outlook, or any SMTP server")
#         smtp_server = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
#         smtp_port = st.sidebar.number_input("SMTP Port", value=587)
#         from_email = st.sidebar.text_input("From Email")
#         email_password = st.sidebar.text_input("Email Password/App Password", type="password")
#         to_email = st.sidebar.text_input("Alert Recipient Email")

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)
# color = risk_color(risk_prob)

# col_a, col_b, col_c = st.columns(3)

# with col_a:
#     st.metric("🌧️ Rainfall", f"{rainfall:.1f} mm/hr")
# with col_b:
#     st.metric("💧 Soil Saturation", f"{soil_moisture:.2f}")
# with col_c:
#     st.metric("📍 Elevation Risk", f"{elevation_risk:.2f}")

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
#     st.error(f"⚠️ **Risk Level:** `{risk_level}`")
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
#     st.warning(f"⚡ **Risk Level:** `{risk_level}`")
# else:
#     risk_level = "LOW"
#     st.success(f"✅ **Risk Level:** `{risk_level}`")

# # ----------------------------------
# # SMART RL ALERT DECISION (Based on Risk Level)
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# rl_prediction = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# # Override RL based on risk level for safety
# if risk_level == "HIGH":
#     alert_decision = 1  # Always alert on HIGH
#     alert_reasoning = "HIGH risk detected - Alert REQUIRED by safety protocol"
# elif risk_level == "MEDIUM":
#     alert_decision = rl_prediction  # Use RL decision
#     alert_reasoning = f"MEDIUM risk - RL Model suggests: {'ALERT' if rl_prediction == 1 else 'MONITOR'}"
# else:  # LOW
#     alert_decision = 0  # Never alert on LOW
#     alert_reasoning = "LOW risk - No alert required, continue monitoring"

# st.info(f"🤖 **Decision Logic:** {alert_reasoning}")

# # ----------------------------------
# # ALERT CONFIRMATION & SENDING
# # ----------------------------------
# if alert_decision == 1:
#     st.error("🚨 ALERT RECOMMENDED - Immediate Response Required")
    
#     # Show confirmation buttons
#     col1, col2, col3 = st.columns([1, 1, 2])
    
#     with col1:
#         if st.button("✅ CONFIRM & SEND ALERT", type="primary", use_container_width=True):
#             st.session_state.show_confirmation = True
            
#             # Generate alert message
#             alert_msg = generate_alert_message(risk_level, risk_prob, rainfall, soil_moisture)
            
#             # Send notification if enabled
#             if notification_enabled:
#                 with st.spinner("Sending alert..."):
#                     success = False
                    
#                     if notification_method == "Telegram (Easiest)" and telegram_bot_token and telegram_chat_id:
#                         success, msg = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_msg)
                        
#                     elif notification_method == "SMS (Twilio)" and twilio_account_sid and twilio_auth_token:
#                         success, msg = send_sms_twilio(user_phone_number, alert_msg, 
#                                                        twilio_account_sid, twilio_auth_token, twilio_from_number)
                        
#                     elif notification_method == "WhatsApp (Twilio)" and twilio_account_sid and twilio_auth_token:
#                         success, msg = send_whatsapp_twilio(user_phone_number, alert_msg,
#                                                             twilio_account_sid, twilio_auth_token, twilio_from_number)
                        
#                     elif notification_method == "Email (SMTP)" and from_email and email_password and to_email:
#                         success, msg = send_email_alert(to_email, 
#                                                         f"🚨 FLOOD ALERT - {risk_level} RISK",
#                                                         alert_msg, smtp_server, smtp_port, 
#                                                         from_email, email_password)
                    
#                     if success:
#                         st.success(msg)
#                         st.session_state.alert_sent = True
#                         st.session_state.alert_history.append({
#                             'timestamp': datetime.now(),
#                             'risk_level': risk_level,
#                             'risk_prob': risk_prob,
#                             'method': notification_method,
#                             'status': 'Sent'
#                         })
#                     else:
#                         st.error(msg)
#             else:
#                 st.warning("⚠️ Notification not enabled. Enable in sidebar to send alerts.")
#                 st.session_state.alert_sent = True
#                 st.session_state.alert_history.append({
#                     'timestamp': datetime.now(),
#                     'risk_level': risk_level,
#                     'risk_prob': risk_prob,
#                     'method': 'None',
#                     'status': 'Confirmed (Not Sent)'
#                 })
            
#             # Display alert message
#             st.text_area("Alert Message Sent:", alert_msg, height=400)
    
#     with col2:
#         if st.button("❌ CANCEL ALERT", use_container_width=True):
#             st.session_state.show_confirmation = False
#             st.info("Alert cancelled by user")
#             st.session_state.alert_history.append({
#                 'timestamp': datetime.now(),
#                 'risk_level': risk_level,
#                 'risk_prob': risk_prob,
#                 'method': 'None',
#                 'status': 'Cancelled'
#             })
    
#     with col3:
#         if st.session_state.alert_sent:
#             st.success("✅ Alert has been sent successfully!")

# else:
#     st.success("✅ NO ALERT - Continue Monitoring")
#     st.info("Current conditions are within safe parameters. System continues routine monitoring.")

# # ----------------------------------
# # ALERT HISTORY
# # ----------------------------------
# if st.session_state.alert_history:
#     st.subheader("📜 Alert History")
#     history_df = pd.DataFrame(st.session_state.alert_history)
#     st.dataframe(history_df, use_container_width=True)

# # ----------------------------------
# # NDMA EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write("""
#     **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#     and low-lying terrain.

#     **NDMA Recommended Actions:**
#     - ⚠️ Activate emergency response teams
#     - 🏠 Prepare evacuation shelters
#     - 🚰 Continuous drainage monitoring
#     - 📢 Issue early public alerts
#     """)
# elif risk_level == "MEDIUM":
#     st.write("""
#     **Flood risk is MODERATE**.

#     **Recommended Actions:**
#     - 📊 Monitor rainfall trends
#     - 👥 Keep emergency teams on standby
#     - 🔧 Inspect drainage infrastructure
#     """)
# else:
#     st.write("""
#     **Flood risk is LOW**.

#     **Recommended Actions:**
#     - 👀 Routine monitoring
#     - ✅ No immediate emergency actions required
#     """)

# # ----------------------------------
# # MAP VISUALIZATION (PYDECK)
# # ----------------------------------
# st.subheader("Flood Impact Visualization")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### Low-Lying Flood-Prone Zones")

#     low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
#     low_zones["lat"] = low_zones.geometry.centroid.y
#     low_zones["lon"] = low_zones.geometry.centroid.x
#     low_zones["color"] = [color] * len(low_zones)

#     low_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=low_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=120,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[low_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#         )
#     )

# with col2:
#     st.markdown("### Critical Infrastructure at Risk")

#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
#     impact_zones["lat"] = impact_zones.geometry.centroid.y
#     impact_zones["lon"] = impact_zones.geometry.centroid.x
#     impact_zones["color"] = [color] * len(impact_zones)

#     infra_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=impact_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=180,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[infra_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#             tooltip={"text": "{name}"} if "name" in impact_zones.columns else None,
#         )
#     )

# # ----------------------------------
# # INFRASTRUCTURE SUMMARY
# # ----------------------------------
# st.subheader("🚨 Critical Infrastructure Summary")

# possible_cols = ["infrastructure_type", "infra_type", "amenity", "type", "category", "layer", "name", "source"]
# infra_col = next((c for c in possible_cols if c in impact_zones.columns), None)

# if infra_col:
#     infra_counts = impact_zones[infra_col].value_counts()
#     st.table(infra_counts)
#     total = int(infra_counts.sum())
# else:
#     total = len(impact_zones)

# st.metric("Total Critical Infrastructure Affected", total)

# # ----------------------------------
# # AUTO-REFRESH LOGIC
# # ----------------------------------
# if data_mode == "Real-Time API" and auto_refresh:
#     time.sleep(300)
#     st.rerun()

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + GenAI + Reinforcement Learning | Real-Time Edition"
# )


# from pathlib import Path

# import geopandas as gpd
# import joblib
# import pandas as pd
# import pydeck as pdk
# import streamlit as st
# import requests
# from datetime import datetime
# import time
# import numpy as np

# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="RainGuardAI",
#     layout="wide",
# )

# st.title("🌊 RainGuardAI - Autonomous Flood Alert System")
# st.markdown(
#     "AI-driven flood prediction with **LIVE** weather data, **Gemini AI Insights**, geospatial risk analysis, "
#     "reinforcement learning alert optimization, and NDMA-based explanations."
# )

# # ----------------------------------
# # PATHS
# # ----------------------------------
# BASE_DIR = Path(__file__).resolve().parents[1]
# RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
# LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
# IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# # ----------------------------------
# # INITIALIZE SESSION STATE
# # ----------------------------------
# if 'alert_sent' not in st.session_state:
#     st.session_state.alert_sent = False
# if 'alert_history' not in st.session_state:
#     st.session_state.alert_history = []
# if 'show_confirmation' not in st.session_state:
#     st.session_state.show_confirmation = False
# if 'gemini_summary' not in st.session_state:
#     st.session_state.gemini_summary = None

# # ----------------------------------
# # GEMINI AI SUMMARIZATION FUNCTION
# # ----------------------------------

# def generate_gemini_summary(risk_level, risk_prob, rainfall, soil_moisture, elevation_risk, 
#                            alert_decision, location="Mumbai"):
#     """
#     Generate AI-powered summary using Google Gemini API
#     Get free API key from: https://makersuite.google.com/app/apikey
#     """
#     try:
#         import google.generativeai as genai
        
#         # Configure Gemini
#         gemini_api_key = st.session_state.get('gemini_api_key', '')
#         if not gemini_api_key:
#             return None, "Please enter Gemini API key in sidebar"
        
#         genai.configure(api_key=gemini_api_key)
#         model = genai.GenerativeModel('gemini-pro')
        
#         # Create detailed prompt
#         prompt = f"""
# You are an expert flood risk analyst for RainGuardAI system. Analyze the following flood risk data and provide a comprehensive summary.

# **CURRENT SITUATION:**
# - Location: {location}
# - Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# - Risk Level: {risk_level}
# - Risk Probability: {risk_prob:.2%}

# **ENVIRONMENTAL FACTORS:**
# - Rainfall Intensity: {rainfall:.1f} mm/hr
# - Soil Saturation: {soil_moisture:.2%}
# - Elevation Risk Factor: {elevation_risk:.2f}
# - Alert Status: {"ALERT RECOMMENDED" if alert_decision == 1 else "NO ALERT - MONITORING"}

# **ANALYSIS REQUIRED:**
# Please provide a professional flood risk assessment including:

# 1. **Situation Overview** (2-3 sentences)
#    - What is happening right now?
#    - How severe is the situation?

# 2. **Key Risk Factors** (bullet points)
#    - Which factors are most concerning?
#    - What makes this situation risky/safe?

# 3. **Immediate Recommendations** (3-5 actions)
#    - What should authorities do NOW?
#    - What should citizens do?

# 4. **Next 6-12 Hours Outlook**
#    - What should we watch for?
#    - When should we reassess?

# 5. **Confidence Assessment**
#    - How confident are we in this prediction?
#    - What could change the situation?

# Format your response in clear sections with emojis for readability. Be specific, actionable, and professional.
# Keep the tone serious but not alarmist. Focus on safety and preparedness.
# """
        
#         # Generate response
#         response = model.generate_content(prompt)
        
#         return response.text, None
        
#     except ImportError:
#         return None, "Install Google Generative AI: pip install google-generativeai"
#     except Exception as e:
#         return None, f"Gemini API Error: {str(e)}"


# # ----------------------------------
# # MOBILE NOTIFICATION FUNCTIONS
# # ----------------------------------

# def send_sms_twilio(phone_number, message, account_sid, auth_token, from_number):
#     """Send SMS via Twilio API"""
#     try:
#         from twilio.rest import Client
        
#         client = Client(account_sid, auth_token)
        
#         sms = client.messages.create(
#             body=message,
#             from_=from_number,
#             to=phone_number
#         )
        
#         return True, f"✅ SMS sent! SID: {sms.sid}"
#     except ImportError:
#         return False, "❌ Install Twilio: pip install twilio"
#     except Exception as e:
#         return False, f"❌ SMS failed: {str(e)}"


# def send_whatsapp_twilio(phone_number, message, account_sid, auth_token, from_number):
#     """Send WhatsApp via Twilio API"""
#     try:
#         from twilio.rest import Client
        
#         client = Client(account_sid, auth_token)
        
#         whatsapp = client.messages.create(
#             body=message,
#             from_=f'whatsapp:{from_number}',
#             to=f'whatsapp:{phone_number}'
#         )
        
#         return True, f"✅ WhatsApp sent! SID: {whatsapp.sid}"
#     except ImportError:
#         return False, "❌ Install Twilio: pip install twilio"
#     except Exception as e:
#         return False, f"❌ WhatsApp failed: {str(e)}"


# def send_email_alert(to_email, subject, message, smtp_server, smtp_port, from_email, password):
#     """Send Email Alert via SMTP"""
#     try:
#         import smtplib
#         from email.mime.text import MIMEText
#         from email.mime.multipart import MIMEMultipart
        
#         msg = MIMEMultipart()
#         msg['From'] = from_email
#         msg['To'] = to_email
#         msg['Subject'] = subject
        
#         msg.attach(MIMEText(message, 'plain'))
        
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(from_email, password)
#         server.send_message(msg)
#         server.quit()
        
#         return True, "✅ Email sent successfully!"
#     except Exception as e:
#         return False, f"❌ Email failed: {str(e)}"


# def send_telegram_alert(bot_token, chat_id, message):
#     """Send alert via Telegram Bot"""
#     try:
#         url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
#         data = {
#             "chat_id": chat_id,
#             "text": message,
#             "parse_mode": "Markdown"
#         }
        
#         response = requests.post(url, data=data, timeout=10)
#         response.raise_for_status()
        
#         return True, "✅ Telegram alert sent!"
#     except Exception as e:
#         return False, f"❌ Telegram failed: {str(e)}"


# def generate_alert_message(risk_level, risk_prob, rainfall, soil_moisture, location="Mumbai"):
#     """Generate detailed alert message based on risk level"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     if risk_level == "HIGH":
#         message = f"""
# 🚨 URGENT FLOOD ALERT - HIGH RISK 🚨

# Location: {location}
# Time: {timestamp}
# Risk Level: {risk_level}
# Risk Probability: {risk_prob:.2%}

# CURRENT CONDITIONS:
# • Rainfall: {rainfall:.1f} mm/hr
# • Soil Saturation: {soil_moisture:.2%}

# ⚠️ IMMEDIATE ACTIONS REQUIRED:
# 1. Evacuate low-lying areas immediately
# 2. Move to higher ground
# 3. Activate emergency response teams
# 4. Prepare evacuation shelters
# 5. Monitor drainage systems continuously
# 6. Issue public alerts via all channels

# NDMA Guidelines:
# - Do NOT attempt to cross flooded areas
# - Keep emergency contacts ready
# - Listen to local authorities
# - Stock emergency supplies (food, water, medicine)

# Stay Safe! Follow official instructions.

# RainGuardAI Autonomous Flood Alert System
#         """
    
#     elif risk_level == "MEDIUM":
#         message = f"""
# ⚠️ FLOOD ALERT - MODERATE RISK ⚠️

# Location: {location}
# Time: {timestamp}
# Risk Level: {risk_level}
# Risk Probability: {risk_prob:.2%}

# CURRENT CONDITIONS:
# • Rainfall: {rainfall:.1f} mm/hr
# • Soil Saturation: {soil_moisture:.2%}

# 📋 RECOMMENDED ACTIONS:
# 1. Monitor weather updates closely
# 2. Keep emergency teams on standby
# 3. Inspect drainage infrastructure
# 4. Prepare emergency supplies
# 5. Avoid low-lying areas if possible

# NDMA Guidelines:
# - Stay informed about weather conditions
# - Keep emergency kit ready
# - Avoid unnecessary travel
# - Check on vulnerable neighbors

# Stay Alert!

# RainGuardAI Autonomous Flood Alert System
#         """
    
#     else:  # LOW
#         message = f"""
# ✅ FLOOD MONITORING - LOW RISK ✅

# Location: {location}
# Time: {timestamp}
# Risk Level: {risk_level}
# Risk Probability: {risk_prob:.2%}

# CURRENT CONDITIONS:
# • Rainfall: {rainfall:.1f} mm/hr
# • Soil Saturation: {soil_moisture:.2%}

# Status: Routine monitoring in progress
# No immediate emergency actions required

# Continue normal activities.

# RainGuardAI Autonomous Flood Alert System
#         """
    
#     return message.strip()


# # ----------------------------------
# # REAL-TIME DATA FETCHING FUNCTIONS
# # ----------------------------------

# @st.cache_data(ttl=300)
# def fetch_openweather_data(lat, lon, api_key):
#     """Fetch real-time weather data from OpenWeatherMap API"""
#     try:
#         url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         rainfall = data.get('rain', {}).get('1h', 0)
#         humidity = data.get('main', {}).get('humidity', 50) / 100
#         temp = data.get('main', {}).get('temp', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': data.get('weather', [{}])[0].get('description', 'N/A'),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     except Exception as e:
#         st.error(f"Error fetching OpenWeatherMap data: {e}")
#         return None


# @st.cache_data(ttl=300)
# def fetch_weatherapi_data(lat, lon, api_key):
#     """Fetch real-time weather data from WeatherAPI.com"""
#     try:
#         url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         current = data.get('current', {})
#         rainfall = current.get('precip_mm', 0)
#         humidity = current.get('humidity', 50) / 100
#         temp = current.get('temp_c', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': current.get('condition', {}).get('text', 'N/A'),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     except Exception as e:
#         st.error(f"Error fetching WeatherAPI data: {e}")
#         return None


# @st.cache_data(ttl=300)
# def fetch_open_meteo_data(lat, lon):
#     """Fetch real-time weather data from Open-Meteo (FREE, no API key needed!)"""
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         current = data.get('current', {})
#         rainfall = current.get('rain', 0) + current.get('precipitation', 0)
#         humidity = current.get('relative_humidity_2m', 50) / 100
#         temp = current.get('temperature_2m', 25)
        
#         return {
#             'rainfall': rainfall,
#             'humidity': humidity,
#             'temp': temp,
#             'description': 'Real-time data',
#             'timestamp': current.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#         }
#     except Exception as e:
#         st.error(f"Error fetching Open-Meteo data: {e}")
#         return None


# def estimate_soil_moisture(humidity, rainfall, temp):
#     """Estimate soil saturation based on weather conditions"""
#     base_moisture = humidity * 0.5
#     rain_contribution = min(rainfall / 50, 0.4)
#     temp_factor = max(0, (30 - temp) / 30) * 0.1
    
#     soil_moisture = min(base_moisture + rain_contribution + temp_factor, 1.0)
#     return soil_moisture


# def get_elevation_risk(lat, lon):
#     """Get elevation data from Open-Elevation API (FREE)"""
#     try:
#         url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         elevation = data['results'][0]['elevation']
        
#         if elevation < 5:
#             risk = 0.9
#         elif elevation < 10:
#             risk = 0.7
#         elif elevation < 20:
#             risk = 0.5
#         elif elevation < 50:
#             risk = 0.3
#         else:
#             risk = 0.1
            
#         return risk, elevation
#     except Exception as e:
#         st.warning(f"Could not fetch elevation data: {e}")
#         return 0.7, None


# # ----------------------------------
# # HEATMAP GENERATION FUNCTION
# # ----------------------------------

# def generate_flood_heatmap(rainfall, soil_moisture, elevation_risk, risk_prob, geojson_data):
#     """
#     Generate heatmap data based on environmental factors
#     """
#     # Create grid points around Mumbai
#     lat_min, lat_max = 18.90, 19.25
#     lon_min, lon_max = 72.70, 73.00
    
#     # Create grid
#     grid_size = 20
#     lats = np.linspace(lat_min, lat_max, grid_size)
#     lons = np.linspace(lon_min, lon_max, grid_size)
    
#     heatmap_data = []
    
#     for i, lat in enumerate(lats):
#         for j, lon in enumerate(lons):
#             # Vary intensity based on factors
#             # Add some randomness to simulate variation
#             variation = np.random.uniform(0.7, 1.3)
            
#             # Calculate local risk with variation
#             local_rainfall_factor = (rainfall / 300) * variation
#             local_soil_factor = soil_moisture * variation * 0.8
#             local_elevation_factor = elevation_risk * variation * 0.9
            
#             # Weight calculation similar to main model
#             intensity = (
#                 0.5 * local_rainfall_factor +
#                 0.3 * local_soil_factor +
#                 0.2 * local_elevation_factor
#             )
            
#             intensity = min(max(intensity, 0), 1)
            
#             # Scale to 0-255 for visualization
#             weight = int(intensity * 255)
            
#             heatmap_data.append({
#                 'lat': lat,
#                 'lon': lon,
#                 'weight': weight
#             })
    
#     return pd.DataFrame(heatmap_data)


# # ----------------------------------
# # ANN PROXY FUNCTION
# # ----------------------------------
# def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
#     risk = (
#         0.5 * (rainfall / 300) +
#         0.3 * soil_moisture +
#         0.2 * elevation_risk
#     )
#     return min(max(risk, 0), 1)


# # ----------------------------------
# # COLOR LOGIC
# # ----------------------------------
# def risk_color(risk):
#     if risk >= 0.75:
#         return [255, 0, 0]      # Red
#     elif risk >= 0.5:
#         return [255, 255, 0]    # Yellow
#     else:
#         return [0, 200, 0]      # Green


# # ----------------------------------
# # LOAD RL MODEL
# # ----------------------------------
# rl_model = joblib.load(RL_MODEL_PATH)

# # ----------------------------------
# # SIDEBAR - DATA SOURCE SELECTION
# # ----------------------------------
# st.sidebar.header("⚙️ Configuration")

# data_mode = st.sidebar.radio(
#     "Data Source Mode",
#     ["Real-Time API", "Manual Input"],
#     help="Choose between live weather data or manual sliders"
# )

# if data_mode == "Real-Time API":
#     st.sidebar.subheader("API Settings")
    
#     latitude = st.sidebar.number_input("Latitude", value=19.0760, format="%.4f")
#     longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.4f")
    
#     api_source = st.sidebar.selectbox(
#         "Weather Data Provider",
#         ["Open-Meteo (Free, No Key)", "OpenWeatherMap", "WeatherAPI.com"],
#         help="Open-Meteo requires no API key!"
#     )
    
#     api_key = ""
#     if api_source != "Open-Meteo (Free, No Key)":
#         api_key = st.sidebar.text_input(
#             "API Key",
#             type="password",
#             help=f"Get your free API key from {api_source}"
#         )
    
#     auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
#     if auto_refresh:
#         st.sidebar.info("⏱️ Auto-refresh enabled")
    
#     refresh_button = st.sidebar.button("🔄 Refresh Data Now")
    
#     # Fetch real-time data
#     if api_source == "Open-Meteo (Free, No Key)":
#         weather_data = fetch_open_meteo_data(latitude, longitude)
#     elif api_source == "OpenWeatherMap" and api_key:
#         weather_data = fetch_openweather_data(latitude, longitude, api_key)
#     elif api_source == "WeatherAPI.com" and api_key:
#         weather_data = fetch_weatherapi_data(latitude, longitude, api_key)
#     else:
#         weather_data = None
#         if api_source != "Open-Meteo (Free, No Key)":
#             st.sidebar.warning("⚠️ Please enter your API key")
    
#     if weather_data:
#         st.sidebar.success(f"✅ Live data updated: {weather_data['timestamp']}")
        
#         st.sidebar.metric("🌧️ Rainfall (mm/hr)", f"{weather_data['rainfall']:.1f}")
#         st.sidebar.metric("💧 Humidity", f"{weather_data['humidity']*100:.0f}%")
#         st.sidebar.metric("🌡️ Temperature", f"{weather_data['temp']:.1f}°C")
#         st.sidebar.caption(f"Conditions: {weather_data['description']}")
        
#         soil_moisture = estimate_soil_moisture(
#             weather_data['humidity'],
#             weather_data['rainfall'],
#             weather_data['temp']
#         )
        
#         elevation_risk, elevation = get_elevation_risk(latitude, longitude)
#         if elevation is not None:
#             st.sidebar.metric("📏 Elevation", f"{elevation:.1f}m")
        
#         rainfall = weather_data['rainfall']
        
#     else:
#         st.sidebar.error("❌ Could not fetch real-time data. Using default values.")
#         rainfall = 120
#         soil_moisture = 0.65
#         elevation_risk = 0.70

# else:
#     st.sidebar.subheader("Manual Environmental Inputs")
#     rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
#     soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
#     elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# # ----------------------------------
# # GEMINI AI SETTINGS
# # ----------------------------------
# st.sidebar.markdown("---")
# st.sidebar.subheader("🤖 Gemini AI Insights")

# enable_gemini = st.sidebar.checkbox("Enable AI Summary", value=False)

# if enable_gemini:
#     st.sidebar.info("Get free API key from Google AI Studio")
#     gemini_api_key = st.sidebar.text_input(
#         "Gemini API Key",
#         type="password",
#         help="https://makersuite.google.com/app/apikey"
#     )
#     st.session_state['gemini_api_key'] = gemini_api_key
    
#     if st.sidebar.button("🔄 Generate AI Summary"):
#         st.session_state.gemini_summary = "generating"

# # ----------------------------------
# # NOTIFICATION SETTINGS
# # ----------------------------------
# st.sidebar.markdown("---")
# st.sidebar.subheader("📱 Alert Notification Settings")

# notification_enabled = st.sidebar.checkbox("Enable Mobile Alerts", value=False)

# if notification_enabled:
#     notification_method = st.sidebar.selectbox(
#         "Notification Method",
#         ["Telegram (Easiest)", "SMS (Twilio)", "WhatsApp (Twilio)", "Email (SMTP)"]
#     )
    
#     if notification_method == "Telegram (Easiest)":
#         st.sidebar.info("Get Bot Token from @BotFather on Telegram")
#         telegram_bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
#         telegram_chat_id = st.sidebar.text_input("Your Chat ID", help="Get from @userinfobot")
        
#     elif notification_method in ["SMS (Twilio)", "WhatsApp (Twilio)"]:
#         st.sidebar.info("Sign up at twilio.com/try-twilio")
#         twilio_account_sid = st.sidebar.text_input("Twilio Account SID", type="password")
#         twilio_auth_token = st.sidebar.text_input("Twilio Auth Token", type="password")
#         twilio_from_number = st.sidebar.text_input("From Number (e.g., +1234567890)")
#         user_phone_number = st.sidebar.text_input("Your Phone Number (e.g., +919876543210)")
        
#     elif notification_method == "Email (SMTP)":
#         st.sidebar.info("Use Gmail App Password (not regular password)")
#         smtp_server = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
#         smtp_port = st.sidebar.number_input("SMTP Port", value=587)
#         from_email = st.sidebar.text_input("From Email")
#         email_password = st.sidebar.text_input("Email App Password", type="password")
#         to_email = st.sidebar.text_input("Alert Recipient Email")

# # ----------------------------------
# # FLOOD RISK PREDICTION
# # ----------------------------------
# risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)
# color = risk_color(risk_prob)

# col_a, col_b, col_c = st.columns(3)

# with col_a:
#     st.metric("🌧️ Rainfall", f"{rainfall:.1f} mm/hr")
# with col_b:
#     st.metric("💧 Soil Saturation", f"{soil_moisture:.2f}")
# with col_c:
#     st.metric("📍 Elevation Risk", f"{elevation_risk:.2f}")

# st.subheader("Flood Risk Prediction")
# st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

# if risk_prob >= 0.75:
#     risk_level = "HIGH"
#     st.error(f"⚠️ **Risk Level:** `{risk_level}`")
# elif risk_prob >= 0.50:
#     risk_level = "MEDIUM"
#     st.warning(f"⚡ **Risk Level:** `{risk_level}`")
# else:
#     risk_level = "LOW"
#     st.success(f"✅ **Risk Level:** `{risk_level}`")

# # ----------------------------------
# # SMART RL ALERT DECISION
# # ----------------------------------
# X_rl = pd.DataFrame({"risk_score": [risk_prob]})
# rl_prediction = rl_model.predict(X_rl)[0]

# st.subheader("Alert Decision (Reinforcement Learning)")

# # Override RL based on risk level for safety
# if risk_level == "HIGH":
#     alert_decision = 1
#     alert_reasoning = "HIGH risk detected - Alert REQUIRED by safety protocol"
# elif risk_level == "MEDIUM":
#     alert_decision = rl_prediction
#     alert_reasoning = f"MEDIUM risk - RL Model suggests: {'ALERT' if rl_prediction == 1 else 'MONITOR'}"
# else:
#     alert_decision = 0
#     alert_reasoning = "LOW risk - No alert required, continue monitoring"

# st.info(f"🤖 **Decision Logic:** {alert_reasoning}")

# # ----------------------------------
# # GEMINI AI SUMMARY
# # ----------------------------------
# if enable_gemini and st.session_state.gemini_summary == "generating":
#     st.subheader("🤖 AI-Powered Risk Analysis (Gemini)")
    
#     with st.spinner("Generating AI insights..."):
#         summary, error = generate_gemini_summary(
#             risk_level, risk_prob, rainfall, soil_moisture, 
#             elevation_risk, alert_decision
#         )
        
#         if summary:
#             st.session_state.gemini_summary = summary
#             st.markdown(summary)
#         else:
#             st.error(error)
#             st.session_state.gemini_summary = None

# elif enable_gemini and st.session_state.gemini_summary and st.session_state.gemini_summary != "generating":
#     st.subheader("🤖 AI-Powered Risk Analysis (Gemini)")
#     st.markdown(st.session_state.gemini_summary)

# # ----------------------------------
# # ALERT CONFIRMATION & SENDING
# # ----------------------------------
# if alert_decision == 1:
#     st.error("🚨 ALERT RECOMMENDED - Immediate Response Required")
    
#     col1, col2, col3 = st.columns([1, 1, 2])
    
#     with col1:
#         if st.button("✅ CONFIRM & SEND ALERT", type="primary", use_container_width=True):
#             st.session_state.show_confirmation = True
            
#             alert_msg = generate_alert_message(risk_level, risk_prob, rainfall, soil_moisture)
            
#             if notification_enabled:
#                 with st.spinner("Sending alert..."):
#                     success = False
                    
#                     if notification_method == "Telegram (Easiest)" and telegram_bot_token and telegram_chat_id:
#                         success, msg = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_msg)
                        
#                     elif notification_method == "SMS (Twilio)" and twilio_account_sid and twilio_auth_token:
#                         success, msg = send_sms_twilio(user_phone_number, alert_msg, 
#                                                        twilio_account_sid, twilio_auth_token, twilio_from_number)
                        
#                     elif notification_method == "WhatsApp (Twilio)" and twilio_account_sid and twilio_auth_token:
#                         success, msg = send_whatsapp_twilio(user_phone_number, alert_msg,
#                                                             twilio_account_sid, twilio_auth_token, twilio_from_number)
                        
#                     elif notification_method == "Email (SMTP)" and from_email and email_password and to_email:
#                         success, msg = send_email_alert(to_email, 
#                                                         f"🚨 FLOOD ALERT - {risk_level} RISK",
#                                                         alert_msg, smtp_server, smtp_port, 
#                                                         from_email, email_password)
                    
#                     if success:
#                         st.success(msg)
#                         st.session_state.alert_sent = True
#                         st.session_state.alert_history.append({
#                             'timestamp': datetime.now(),
#                             'risk_level': risk_level,
#                             'risk_prob': risk_prob,
#                             'method': notification_method,
#                             'status': 'Sent'
#                         })
#                     else:
#                         st.error(msg)
#             else:
#                 st.warning("⚠️ Notification not enabled. Enable in sidebar to send alerts.")
#                 st.session_state.alert_sent = True
#                 st.session_state.alert_history.append({
#                     'timestamp': datetime.now(),
#                     'risk_level': risk_level,
#                     'risk_prob': risk_prob,
#                     'method': 'None',
#                     'status': 'Confirmed (Not Sent)'
#                 })
            
#             st.text_area("Alert Message Sent:", alert_msg, height=400)
    
#     with col2:
#         if st.button("❌ CANCEL ALERT", use_container_width=True):
#             st.session_state.show_confirmation = False
#             st.info("Alert cancelled by user")
#             st.session_state.alert_history.append({
#                 'timestamp': datetime.now(),
#                 'risk_level': risk_level,
#                 'risk_prob': risk_prob,
#                 'method': 'None',
#                 'status': 'Cancelled'
#             })
    
#     with col3:
#         if st.session_state.alert_sent:
#             st.success("✅ Alert has been sent successfully!")

# else:
#     st.success("✅ NO ALERT - Continue Monitoring")
#     st.info("Current conditions are within safe parameters. System continues routine monitoring.")

# # ----------------------------------
# # ALERT HISTORY
# # ----------------------------------
# if st.session_state.alert_history:
#     st.subheader("📜 Alert History")
#     history_df = pd.DataFrame(st.session_state.alert_history)
#     st.dataframe(history_df, use_container_width=True)

# # ----------------------------------
# # NDMA EXPLANATION
# # ----------------------------------
# st.subheader("NDMA-Based Explanation & Action Plan")

# if risk_level == "HIGH":
#     st.write("""
#     **Flood risk is HIGH** due to intense rainfall, high soil saturation,
#     and low-lying terrain.

#     **NDMA Recommended Actions:**
#     - ⚠️ Activate emergency response teams
#     - 🏠 Prepare evacuation shelters
#     - 🚰 Continuous drainage monitoring
#     - 📢 Issue early public alerts
#     """)
# elif risk_level == "MEDIUM":
#     st.write("""
#     **Flood risk is MODERATE**.

#     **Recommended Actions:**
#     - 📊 Monitor rainfall trends
#     - 👥 Keep emergency teams on standby
#     - 🔧 Inspect drainage infrastructure
#     """)
# else:
#     st.write("""
#     **Flood risk is LOW**.

#     **Recommended Actions:**
#     - 👀 Routine monitoring
#     - ✅ No immediate emergency actions required
#     """)

# # ----------------------------------
# # MAP VISUALIZATION WITH HEATMAP
# # ----------------------------------
# st.subheader("🗺️ Flood Risk Visualization")

# # Generate heatmap data
# low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
# heatmap_df = generate_flood_heatmap(rainfall, soil_moisture, elevation_risk, risk_prob, low_zones)

# # Create tabs for different visualizations
# tab1, tab2, tab3 = st.tabs(["🌡️ Risk Heatmap", "📍 Flood-Prone Zones", "🏗️ Critical Infrastructure"])

# with tab1:
#     st.markdown("### Real-Time Flood Risk Heatmap")
#     st.caption("Based on: Rainfall, Soil Moisture, and Elevation Risk")
    
#     # Heatmap layer
#     heatmap_layer = pdk.Layer(
#         "HeatmapLayer",
#         data=heatmap_df,
#         get_position="[lon, lat]",
#         get_weight="weight",
#         radiusPixels=60,
#         intensity=1,
#         threshold=0.05,
#         opacity=0.8,
#     )
    
#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[heatmap_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#                 pitch=0,
#             ),
#             map_style="mapbox://styles/mapbox/dark-v10",
#         )
#     )
    
#     # Legend
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown("🟢 **Low Risk** - Safe zones")
#     with col2:
#         st.markdown("🟡 **Medium Risk** - Monitor closely")
#     with col3:
#         st.markdown("🔴 **High Risk** - Immediate action")

# with tab2:
#     st.markdown("### Low-Lying Flood-Prone Zones")

#     low_zones["lat"] = low_zones.geometry.centroid.y
#     low_zones["lon"] = low_zones.geometry.centroid.x
#     low_zones["color"] = [color] * len(low_zones)

#     low_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=low_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=120,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[low_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#         )
#     )

# with tab3:
#     st.markdown("### Critical Infrastructure at Risk")

#     impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
#     impact_zones["lat"] = impact_zones.geometry.centroid.y
#     impact_zones["lon"] = impact_zones.geometry.centroid.x
#     impact_zones["color"] = [color] * len(impact_zones)

#     infra_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=impact_zones,
#         get_position="[lon, lat]",
#         get_color="color",
#         get_radius=180,
#         pickable=True,
#     )

#     st.pydeck_chart(
#         pdk.Deck(
#             layers=[infra_layer],
#             initial_view_state=pdk.ViewState(
#                 latitude=19.0760,
#                 longitude=72.8777,
#                 zoom=10,
#             ),
#             tooltip={"text": "{name}"} if "name" in impact_zones.columns else None,
#         )
#     )

# # ----------------------------------
# # INFRASTRUCTURE SUMMARY
# # ----------------------------------
# st.subheader("🚨 Critical Infrastructure Summary")

# possible_cols = ["infrastructure_type", "infra_type", "amenity", "type", "category", "layer", "name", "source"]
# infra_col = next((c for c in possible_cols if c in impact_zones.columns), None)

# if infra_col:
#     infra_counts = impact_zones[infra_col].value_counts()
#     st.table(infra_counts)
#     total = int(infra_counts.sum())
# else:
#     total = len(impact_zones)

# st.metric("Total Critical Infrastructure Affected", total)

# # ----------------------------------
# # AUTO-REFRESH LOGIC
# # ----------------------------------
# if data_mode == "Real-Time API" and auto_refresh:
#     time.sleep(300)
#     st.rerun()

# # ----------------------------------
# # FOOTER
# # ----------------------------------
# st.markdown("---")
# st.caption(
#     "RainGuardAI | ML + Geospatial Intelligence + Gemini AI + Reinforcement Learning | Real-Time Edition"
# )

from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st
import requests
from datetime import datetime
import time
import numpy as np

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="RainGuardAI",
    layout="wide",
)

st.title("🌊 RainGuardAI - Autonomous Flood Alert System")
st.markdown(
    "AI-driven flood prediction with **LIVE** weather data, **Gemini AI Insights**, geospatial risk analysis, "
    "reinforcement learning alert optimization, and NDMA-based explanations."
)

# ----------------------------------
# PATHS
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RL_MODEL_PATH = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
LOW_ZONES_PATH = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"

# ----------------------------------
# INITIALIZE SESSION STATE
# ----------------------------------
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'show_confirmation' not in st.session_state:
    st.session_state.show_confirmation = False
if 'gemini_summary' not in st.session_state:
    st.session_state.gemini_summary = None

# ----------------------------------
# GEMINI AI SUMMARIZATION FUNCTION
# ----------------------------------

def generate_gemini_summary(risk_level, risk_prob, rainfall, soil_moisture, elevation_risk, 
                           alert_decision, location="Mumbai"):
    """
    Generate AI-powered summary using Google Gemini API
    Get free API key from: https://makersuite.google.com/app/apikey
    """
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        gemini_api_key = st.session_state.get('gemini_api_key', '')
        if not gemini_api_key:
            return None, "Please enter Gemini API key in sidebar"
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create detailed prompt
        prompt = f"""
You are an expert flood risk analyst for RainGuardAI system. Analyze the following flood risk data and provide a comprehensive summary.

**CURRENT SITUATION:**
- Location: {location}
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Risk Level: {risk_level}
- Risk Probability: {risk_prob:.2%}

**ENVIRONMENTAL FACTORS:**
- Rainfall Intensity: {rainfall:.1f} mm/hr
- Soil Saturation: {soil_moisture:.2%}
- Elevation Risk Factor: {elevation_risk:.2f}
- Alert Status: {"ALERT RECOMMENDED" if alert_decision == 1 else "NO ALERT - MONITORING"}

**ANALYSIS REQUIRED:**
Please provide a professional flood risk assessment including:

1. **Situation Overview** (2-3 sentences)
   - What is happening right now?
   - How severe is the situation?

2. **Key Risk Factors** (bullet points)
   - Which factors are most concerning?
   - What makes this situation risky/safe?

3. **Immediate Recommendations** (3-5 actions)
   - What should authorities do NOW?
   - What should citizens do?

4. **Next 6-12 Hours Outlook**
   - What should we watch for?
   - When should we reassess?

5. **Confidence Assessment**
   - How confident are we in this prediction?
   - What could change the situation?

Format your response in clear sections with emojis for readability. Be specific, actionable, and professional.
Keep the tone serious but not alarmist. Focus on safety and preparedness.
"""
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text, None
        
    except ImportError:
        return None, "Install Google Generative AI: pip install google-generativeai"
    except Exception as e:
        return None, f"Gemini API Error: {str(e)}"


# ----------------------------------
# MOBILE NOTIFICATION FUNCTIONS
# ----------------------------------

def send_sms_twilio(phone_number, message, account_sid, auth_token, from_number):
    """Send SMS via Twilio API"""
    try:
        from twilio.rest import Client
        
        client = Client(account_sid, auth_token)
        
        sms = client.messages.create(
            body=message,
            from_=from_number,
            to=phone_number
        )
        
        return True, f"✅ SMS sent! SID: {sms.sid}"
    except ImportError:
        return False, "❌ Install Twilio: pip install twilio"
    except Exception as e:
        return False, f"❌ SMS failed: {str(e)}"


def send_whatsapp_twilio(phone_number, message, account_sid, auth_token, from_number):
    """Send WhatsApp via Twilio API"""
    try:
        from twilio.rest import Client
        
        client = Client(account_sid, auth_token)
        
        whatsapp = client.messages.create(
            body=message,
            from_=f'whatsapp:{from_number}',
            to=f'whatsapp:{phone_number}'
        )
        
        return True, f"✅ WhatsApp sent! SID: {whatsapp.sid}"
    except ImportError:
        return False, "❌ Install Twilio: pip install twilio"
    except Exception as e:
        return False, f"❌ WhatsApp failed: {str(e)}"


def send_email_alert(to_email, subject, message, smtp_server, smtp_port, from_email, password):
    """Send Email Alert via SMTP"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        
        return True, "✅ Email sent successfully!"
    except Exception as e:
        return False, f"❌ Email failed: {str(e)}"


def send_telegram_alert(bot_token, chat_id, message):
    """Send alert via Telegram Bot"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        
        return True, "✅ Telegram alert sent!"
    except Exception as e:
        return False, f"❌ Telegram failed: {str(e)}"


def generate_alert_message(risk_level, risk_prob, rainfall, soil_moisture, location="Mumbai"):
    """Generate detailed alert message based on risk level"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if risk_level == "HIGH":
        message = f"""
🚨 URGENT FLOOD ALERT - HIGH RISK 🚨

Location: {location}
Time: {timestamp}
Risk Level: {risk_level}
Risk Probability: {risk_prob:.2%}

CURRENT CONDITIONS:
• Rainfall: {rainfall:.1f} mm/hr
• Soil Saturation: {soil_moisture:.2%}

⚠️ IMMEDIATE ACTIONS REQUIRED:
1. Evacuate low-lying areas immediately
2. Move to higher ground
3. Activate emergency response teams
4. Prepare evacuation shelters
5. Monitor drainage systems continuously
6. Issue public alerts via all channels

NDMA Guidelines:
- Do NOT attempt to cross flooded areas
- Keep emergency contacts ready
- Listen to local authorities
- Stock emergency supplies (food, water, medicine)

Stay Safe! Follow official instructions.

RainGuardAI Autonomous Flood Alert System
        """
    
    elif risk_level == "MEDIUM":
        message = f"""
⚠️ FLOOD ALERT - MODERATE RISK ⚠️

Location: {location}
Time: {timestamp}
Risk Level: {risk_level}
Risk Probability: {risk_prob:.2%}

CURRENT CONDITIONS:
• Rainfall: {rainfall:.1f} mm/hr
• Soil Saturation: {soil_moisture:.2%}

📋 RECOMMENDED ACTIONS:
1. Monitor weather updates closely
2. Keep emergency teams on standby
3. Inspect drainage infrastructure
4. Prepare emergency supplies
5. Avoid low-lying areas if possible

NDMA Guidelines:
- Stay informed about weather conditions
- Keep emergency kit ready
- Avoid unnecessary travel
- Check on vulnerable neighbors

Stay Alert!

RainGuardAI Autonomous Flood Alert System
        """
    
    else:  # LOW
        message = f"""
✅ FLOOD MONITORING - LOW RISK ✅

Location: {location}
Time: {timestamp}
Risk Level: {risk_level}
Risk Probability: {risk_prob:.2%}

CURRENT CONDITIONS:
• Rainfall: {rainfall:.1f} mm/hr
• Soil Saturation: {soil_moisture:.2%}

Status: Routine monitoring in progress
No immediate emergency actions required

Continue normal activities.

RainGuardAI Autonomous Flood Alert System
        """
    
    return message.strip()


# ----------------------------------
# REAL-TIME DATA FETCHING FUNCTIONS
# ----------------------------------

@st.cache_data(ttl=300)
def fetch_openweather_data(lat, lon, api_key):
    """Fetch real-time weather data from OpenWeatherMap API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        rainfall = data.get('rain', {}).get('1h', 0)
        humidity = data.get('main', {}).get('humidity', 50) / 100
        temp = data.get('main', {}).get('temp', 25)
        
        return {
            'rainfall': rainfall,
            'humidity': humidity,
            'temp': temp,
            'description': data.get('weather', [{}])[0].get('description', 'N/A'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error(f"Error fetching OpenWeatherMap data: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_weatherapi_data(lat, lon, api_key):
    """Fetch real-time weather data from WeatherAPI.com"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        rainfall = current.get('precip_mm', 0)
        humidity = current.get('humidity', 50) / 100
        temp = current.get('temp_c', 25)
        
        return {
            'rainfall': rainfall,
            'humidity': humidity,
            'temp': temp,
            'description': current.get('condition', {}).get('text', 'N/A'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error(f"Error fetching WeatherAPI data: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_open_meteo_data(lat, lon):
    """Fetch real-time weather data from Open-Meteo (FREE, no API key needed!)"""
    urls = [
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto",
        f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto",
    ]

    last_error = None
    for url in urls:
        for attempt in range(3):  # Retry up to 3 times per URL
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                current = data.get('current', {})
                rainfall = current.get('rain', 0) + current.get('precipitation', 0)
                humidity = current.get('relative_humidity_2m', 50) / 100
                temp = current.get('temperature_2m', 25)

                return {
                    'rainfall': rainfall,
                    'humidity': humidity,
                    'temp': temp,
                    'description': 'Real-time data',
                    'timestamp': current.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                }
            except requests.exceptions.Timeout:
                last_error = f"Timeout on attempt {attempt + 1}"
                time.sleep(2)  # Wait before retrying
                continue
            except Exception as e:
                last_error = str(e)
                break  # Non-timeout error, try next URL

    # All attempts failed — return simulated fallback data with a warning
    st.warning(
        f"⚠️ Could not reach Open-Meteo API ({last_error}). "
        "Using estimated values. Check your internet connection or switch to Manual Input mode."
    )
    return {
        'rainfall': 0.0,
        'humidity': 0.60,
        'temp': 28.0,
        'description': '⚠️ Estimated (API unavailable)',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def estimate_soil_moisture(humidity, rainfall, temp):
    """Estimate soil saturation based on weather conditions"""
    base_moisture = humidity * 0.5
    rain_contribution = min(rainfall / 50, 0.4)
    temp_factor = max(0, (30 - temp) / 30) * 0.1
    
    soil_moisture = min(base_moisture + rain_contribution + temp_factor, 1.0)
    return soil_moisture


def get_elevation_risk(lat, lon):
    """Get elevation data from Open-Elevation API (FREE)"""
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        elevation = data['results'][0]['elevation']
        
        if elevation < 5:
            risk = 0.9
        elif elevation < 10:
            risk = 0.7
        elif elevation < 20:
            risk = 0.5
        elif elevation < 50:
            risk = 0.3
        else:
            risk = 0.1
            
        return risk, elevation
    except Exception as e:
        st.warning(f"Could not fetch elevation data: {e}")
        return 0.7, None


# ----------------------------------
# ANN PROXY FUNCTION
# ----------------------------------
def ann_risk_proxy(rainfall, soil_moisture, elevation_risk):
    risk = (
        0.5 * (rainfall / 300) +
        0.3 * soil_moisture +
        0.2 * elevation_risk
    )
    return min(max(risk, 0), 1)


# ----------------------------------
# COLOR LOGIC
# ----------------------------------
def risk_color(risk):
    if risk >= 0.75:
        return [255, 0, 0]      # Red
    elif risk >= 0.5:
        return [255, 255, 0]    # Yellow
    else:
        return [0, 200, 0]      # Green


# ----------------------------------
# LOAD RL MODEL
# ----------------------------------
rl_model = joblib.load(RL_MODEL_PATH)

# ----------------------------------
# SIDEBAR - DATA SOURCE SELECTION
# ----------------------------------
st.sidebar.header("⚙️ Configuration")

data_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Real-Time API", "Manual Input"],
    help="Choose between live weather data or manual sliders"
)

if data_mode == "Real-Time API":
    st.sidebar.subheader("API Settings")
    
    latitude = st.sidebar.number_input("Latitude", value=19.0760, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.4f")
    
    api_source = st.sidebar.selectbox(
        "Weather Data Provider",
        ["Open-Meteo (Free, No Key)", "OpenWeatherMap", "WeatherAPI.com"],
        help="Open-Meteo requires no API key!"
    )
    
    api_key = ""
    if api_source != "Open-Meteo (Free, No Key)":
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            help=f"Get your free API key from {api_source}"
        )
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
    if auto_refresh:
        st.sidebar.info("⏱️ Auto-refresh enabled")
    
    refresh_button = st.sidebar.button("🔄 Refresh Data Now")
    
    # Fetch real-time data
    if api_source == "Open-Meteo (Free, No Key)":
        weather_data = fetch_open_meteo_data(latitude, longitude)
    elif api_source == "OpenWeatherMap" and api_key:
        weather_data = fetch_openweather_data(latitude, longitude, api_key)
    elif api_source == "WeatherAPI.com" and api_key:
        weather_data = fetch_weatherapi_data(latitude, longitude, api_key)
    else:
        weather_data = None
        if api_source != "Open-Meteo (Free, No Key)":
            st.sidebar.warning("⚠️ Please enter your API key")
    
    if weather_data:
        st.sidebar.success(f"✅ Live data updated: {weather_data['timestamp']}")
        
        st.sidebar.metric("🌧️ Rainfall (mm/hr)", f"{weather_data['rainfall']:.1f}")
        st.sidebar.metric("💧 Humidity", f"{weather_data['humidity']*100:.0f}%")
        st.sidebar.metric("🌡️ Temperature", f"{weather_data['temp']:.1f}°C")
        st.sidebar.caption(f"Conditions: {weather_data['description']}")
        
        soil_moisture = estimate_soil_moisture(
            weather_data['humidity'],
            weather_data['rainfall'],
            weather_data['temp']
        )
        
        elevation_risk, elevation = get_elevation_risk(latitude, longitude)
        if elevation is not None:
            st.sidebar.metric("📏 Elevation", f"{elevation:.1f}m")
        
        rainfall = weather_data['rainfall']
        
    else:
        st.sidebar.error("❌ Could not fetch real-time data. Using default values.")
        rainfall = 120
        soil_moisture = 0.65
        elevation_risk = 0.70

else:
    st.sidebar.subheader("Manual Environmental Inputs")
    rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 300, 120)
    soil_moisture = st.sidebar.slider("Soil Saturation Index", 0.0, 1.0, 0.65)
    elevation_risk = st.sidebar.slider("Elevation Risk Factor", 0.0, 1.0, 0.70)

# ----------------------------------
# GEMINI AI SETTINGS
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Gemini AI Insights")

enable_gemini = st.sidebar.checkbox("Enable AI Summary", value=False)

if enable_gemini:
    st.sidebar.info("Get free API key from Google AI Studio")
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="https://makersuite.google.com/app/apikey"
    )
    st.session_state['gemini_api_key'] = gemini_api_key
    
    if st.sidebar.button("🔄 Generate AI Summary"):
        st.session_state.gemini_summary = "generating"

# ----------------------------------
# NOTIFICATION SETTINGS
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📱 Alert Notification Settings")

notification_enabled = st.sidebar.checkbox("Enable Mobile Alerts", value=False)

if notification_enabled:
    notification_method = st.sidebar.selectbox(
        "Notification Method",
        ["Telegram (Easiest)", "SMS (Twilio)", "WhatsApp (Twilio)", "Email (SMTP)"]
    )
    
    if notification_method == "Telegram (Easiest)":
        st.sidebar.info("Get Bot Token from @BotFather on Telegram")
        telegram_bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
        telegram_chat_id = st.sidebar.text_input("Your Chat ID", help="Get from @userinfobot")
        
    elif notification_method in ["SMS (Twilio)", "WhatsApp (Twilio)"]:
        st.sidebar.info("Sign up at twilio.com/try-twilio")
        twilio_account_sid = st.sidebar.text_input("Twilio Account SID", type="password")
        twilio_auth_token = st.sidebar.text_input("Twilio Auth Token", type="password")
        twilio_from_number = st.sidebar.text_input("From Number (e.g., +1234567890)")
        user_phone_number = st.sidebar.text_input("Your Phone Number (e.g., +919876543210)")
        
    elif notification_method == "Email (SMTP)":
        st.sidebar.info("Use Gmail App Password (not regular password)")
        smtp_server = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.sidebar.number_input("SMTP Port", value=587)
        from_email = st.sidebar.text_input("From Email")
        email_password = st.sidebar.text_input("Email App Password", type="password")
        to_email = st.sidebar.text_input("Alert Recipient Email")

# ----------------------------------
# FLOOD RISK PREDICTION
# ----------------------------------
risk_prob = ann_risk_proxy(rainfall, soil_moisture, elevation_risk)
color = risk_color(risk_prob)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("🌧️ Rainfall", f"{rainfall:.1f} mm/hr")
with col_b:
    st.metric("💧 Soil Saturation", f"{soil_moisture:.2f}")
with col_c:
    st.metric("📍 Elevation Risk", f"{elevation_risk:.2f}")

st.subheader("Flood Risk Prediction")
st.metric("Flood Risk Probability", f"{risk_prob:.2f}")

if risk_prob >= 0.75:
    risk_level = "HIGH"
    st.error(f"⚠️ **Risk Level:** `{risk_level}`")
elif risk_prob >= 0.50:
    risk_level = "MEDIUM"
    st.warning(f"⚡ **Risk Level:** `{risk_level}`")
else:
    risk_level = "LOW"
    st.success(f"✅ **Risk Level:** `{risk_level}`")

# ----------------------------------
# SMART RL ALERT DECISION
# ----------------------------------
X_rl = pd.DataFrame({"risk_score": [risk_prob]})
rl_prediction = rl_model.predict(X_rl)[0]

st.subheader("Alert Decision (Reinforcement Learning)")

# Override RL based on risk level for safety
if risk_level == "HIGH":
    alert_decision = 1
    alert_reasoning = "HIGH risk detected - Alert REQUIRED by safety protocol"
elif risk_level == "MEDIUM":
    alert_decision = rl_prediction
    alert_reasoning = f"MEDIUM risk - RL Model suggests: {'ALERT' if rl_prediction == 1 else 'MONITOR'}"
else:
    alert_decision = 0
    alert_reasoning = "LOW risk - No alert required, continue monitoring"

st.info(f"🤖 **Decision Logic:** {alert_reasoning}")

# ----------------------------------
# GEMINI AI SUMMARY
# ----------------------------------
if enable_gemini and st.session_state.gemini_summary == "generating":
    st.subheader("🤖 AI-Powered Risk Analysis (Gemini)")
    
    with st.spinner("Generating AI insights..."):
        summary, error = generate_gemini_summary(
            risk_level, risk_prob, rainfall, soil_moisture, 
            elevation_risk, alert_decision
        )
        
        if summary:
            st.session_state.gemini_summary = summary
            st.markdown(summary)
        else:
            st.error(error)
            st.session_state.gemini_summary = None

elif enable_gemini and st.session_state.gemini_summary and st.session_state.gemini_summary != "generating":
    st.subheader("🤖 AI-Powered Risk Analysis (Gemini)")
    st.markdown(st.session_state.gemini_summary)

# ----------------------------------
# ALERT CONFIRMATION & SENDING
# ----------------------------------
if alert_decision == 1:
    st.error("🚨 ALERT RECOMMENDED - Immediate Response Required")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("✅ CONFIRM & SEND ALERT", type="primary", use_container_width=True):
            st.session_state.show_confirmation = True
            
            alert_msg = generate_alert_message(risk_level, risk_prob, rainfall, soil_moisture)
            
            if notification_enabled:
                with st.spinner("Sending alert..."):
                    success = False
                    
                    if notification_method == "Telegram (Easiest)" and telegram_bot_token and telegram_chat_id:
                        success, msg = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_msg)
                        
                    elif notification_method == "SMS (Twilio)" and twilio_account_sid and twilio_auth_token:
                        success, msg = send_sms_twilio(user_phone_number, alert_msg, 
                                                       twilio_account_sid, twilio_auth_token, twilio_from_number)
                        
                    elif notification_method == "WhatsApp (Twilio)" and twilio_account_sid and twilio_auth_token:
                        success, msg = send_whatsapp_twilio(user_phone_number, alert_msg,
                                                            twilio_account_sid, twilio_auth_token, twilio_from_number)
                        
                    elif notification_method == "Email (SMTP)" and from_email and email_password and to_email:
                        success, msg = send_email_alert(to_email, 
                                                        f"🚨 FLOOD ALERT - {risk_level} RISK",
                                                        alert_msg, smtp_server, smtp_port, 
                                                        from_email, email_password)
                    
                    if success:
                        st.success(msg)
                        st.session_state.alert_sent = True
                        st.session_state.alert_history.append({
                            'timestamp': datetime.now(),
                            'risk_level': risk_level,
                            'risk_prob': risk_prob,
                            'method': notification_method,
                            'status': 'Sent'
                        })
                    else:
                        st.error(msg)
            else:
                st.warning("⚠️ Notification not enabled. Enable in sidebar to send alerts.")
                st.session_state.alert_sent = True
                st.session_state.alert_history.append({
                    'timestamp': datetime.now(),
                    'risk_level': risk_level,
                    'risk_prob': risk_prob,
                    'method': 'None',
                    'status': 'Confirmed (Not Sent)'
                })
            
            st.text_area("Alert Message Sent:", alert_msg, height=400)
    
    with col2:
        if st.button("❌ CANCEL ALERT", use_container_width=True):
            st.session_state.show_confirmation = False
            st.info("Alert cancelled by user")
            st.session_state.alert_history.append({
                'timestamp': datetime.now(),
                'risk_level': risk_level,
                'risk_prob': risk_prob,
                'method': 'None',
                'status': 'Cancelled'
            })
    
    with col3:
        if st.session_state.alert_sent:
            st.success("✅ Alert has been sent successfully!")

else:
    st.success("✅ NO ALERT - Continue Monitoring")
    st.info("Current conditions are within safe parameters. System continues routine monitoring.")

# ----------------------------------
# ALERT HISTORY
# ----------------------------------
if st.session_state.alert_history:
    st.subheader("📜 Alert History")
    history_df = pd.DataFrame(st.session_state.alert_history)
    st.dataframe(history_df, use_container_width=True)

# ----------------------------------
# NDMA EXPLANATION
# ----------------------------------
st.subheader("NDMA-Based Explanation & Action Plan")

if risk_level == "HIGH":
    st.write("""
    **Flood risk is HIGH** due to intense rainfall, high soil saturation,
    and low-lying terrain.

    **NDMA Recommended Actions:**
    - ⚠️ Activate emergency response teams
    - 🏠 Prepare evacuation shelters
    - 🚰 Continuous drainage monitoring
    - 📢 Issue early public alerts
    """)
elif risk_level == "MEDIUM":
    st.write("""
    **Flood risk is MODERATE**.

    **Recommended Actions:**
    - 📊 Monitor rainfall trends
    - 👥 Keep emergency teams on standby
    - 🔧 Inspect drainage infrastructure
    """)
else:
    st.write("""
    **Flood risk is LOW**.

    **Recommended Actions:**
    - 👀 Routine monitoring
    - ✅ No immediate emergency actions required
    """)

# ----------------------------------
# MAP VISUALIZATION (WITHOUT HEATMAP)
# ----------------------------------
st.subheader("🗺️ Flood Risk Visualization")

low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["📍 Flood-Prone Zones", "🏗️ Critical Infrastructure"])

with tab1:
    st.markdown("### Low-Lying Flood-Prone Zones")

    low_zones["lat"] = low_zones.geometry.centroid.y
    low_zones["lon"] = low_zones.geometry.centroid.x
    low_zones["color"] = [color] * len(low_zones)

    low_layer = pdk.Layer(
        "ScatterplotLayer",
        data=low_zones,
        get_position="[lon, lat]",
        get_color="color",
        get_radius=120,
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[low_layer],
            initial_view_state=pdk.ViewState(
                latitude=19.0760,
                longitude=72.8777,
                zoom=10,
            ),
        )
    )

with tab2:
    st.markdown("### Critical Infrastructure at Risk")

    impact_zones = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
    impact_zones["lat"] = impact_zones.geometry.centroid.y
    impact_zones["lon"] = impact_zones.geometry.centroid.x
    impact_zones["color"] = [color] * len(impact_zones)

    infra_layer = pdk.Layer(
        "ScatterplotLayer",
        data=impact_zones,
        get_position="[lon, lat]",
        get_color="color",
        get_radius=180,
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[infra_layer],
            initial_view_state=pdk.ViewState(
                latitude=19.0760,
                longitude=72.8777,
                zoom=10,
            ),
            tooltip={"text": "{name}"} if "name" in impact_zones.columns else None,
        )
    )

# ----------------------------------
# INFRASTRUCTURE SUMMARY
# ----------------------------------
st.subheader("🚨 Critical Infrastructure Summary")

possible_cols = ["infrastructure_type", "infra_type", "amenity", "type", "category", "layer", "name", "source"]
infra_col = next((c for c in possible_cols if c in impact_zones.columns), None)

if infra_col:
    infra_counts = impact_zones[infra_col].value_counts()
    st.table(infra_counts)
    total = int(infra_counts.sum())
else:
    total = len(impact_zones)

st.metric("Total Critical Infrastructure Affected", total)

# ----------------------------------
# AUTO-REFRESH LOGIC
# ----------------------------------
if data_mode == "Real-Time API" and auto_refresh:
    time.sleep(300)
    st.rerun()

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption(
    "RainGuardAI | ML + Geospatial Intelligence + Gemini AI + Reinforcement Learning | Real-Time Edition"
)