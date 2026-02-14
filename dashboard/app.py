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
import json

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="RainGuardAI", layout="wide", page_icon="🌊")

st.title("🌊 RainGuardAI — Autonomous Flood Early Warning System")
st.markdown(
    "AI-driven flood prediction • **Live weather data** • **Zone-wise population analysis** • "
    "**Critical infrastructure mapping** • **Gemini AI summary**"
)

# ----------------------------------
# PATHS
# ----------------------------------
BASE_DIR          = Path(__file__).resolve().parents[1]
RL_MODEL_PATH     = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
LOW_ZONES_PATH    = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"
POPULATION_PATH   = BASE_DIR / "data" / "raw" / "Demographics" / "mumbai_wardwise_population_data.csv"

# ----------------------------------
# SESSION STATE
# ----------------------------------
for key, val in [
    ('alert_sent', False), ('alert_history', []),
    ('show_confirmation', False), ('gemini_summary', None),
    ('ai_full_summary', None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# HELPER — risk zone from flood probability
# ============================================================
def classify_zone(risk_prob):
    if risk_prob >= 0.75:
        return "RED"
    elif risk_prob >= 0.50:
        return "ORANGE"
    else:
        return "GREEN"

def zone_color_rgb(zone):
    return {"RED": [220, 38, 38, 190],
            "ORANGE": [249, 115, 22, 180],
            "GREEN": [34, 197, 94, 170]}.get(zone, [100, 100, 100, 150])

# ============================================================
# POPULATION ZONE ANALYSIS
# ============================================================
@st.cache_data
def load_population():
    return pd.read_csv(POPULATION_PATH)

def compute_population_zones(risk_prob, pop_df):
    df = pop_df.copy()
    df['pop_density'] = df['Population_2025'] / df['Area_in_Sq_km']
    max_density = df['pop_density'].max()
    df['density_factor'] = df['pop_density'] / max_density
    df['ward_risk'] = (0.80 * risk_prob) + (0.20 * df['density_factor'] * risk_prob)
    df['ward_risk'] = df['ward_risk'].clip(0, 1)
    df['Zone'] = df['ward_risk'].apply(classify_zone)
    return df

WARD_COORDS = {
    'A':   (18.9400, 72.8370), 'B':   (18.9550, 72.8390),
    'C':   (18.9640, 72.8280), 'D':   (18.9700, 72.8180),
    'E':   (18.9780, 72.8140), 'F/S': (19.0050, 72.8250),
    'F/N': (19.0280, 72.8380), 'G/S': (19.0100, 72.8450),
    'G/N': (19.0350, 72.8500), 'H/E': (19.0550, 72.8680),
    'H/W': (19.0700, 72.8380), 'K/E': (19.0900, 72.8950),
    'K/W': (19.1000, 72.8500), 'P/S': (19.1150, 72.8600),
    'P/N': (19.1500, 72.8700), 'R/S': (19.1300, 72.8800),
    'R/C': (19.1700, 72.9100), 'R/N': (19.1900, 72.8650),
    'L':   (19.0950, 72.9250), 'M/E': (19.1200, 72.9400),
    'M/W': (19.0750, 72.9150), 'N':   (19.1450, 72.9300),
    'S':   (19.1050, 72.9550), 'T':   (19.1750, 72.9500),
}

def build_ward_map_data(zone_df):
    rows = []
    for _, row in zone_df.iterrows():
        ward = row['Ward']
        if ward in WARD_COORDS:
            lat, lon = WARD_COORDS[ward]
            rows.append({
                'Ward': ward, 'lat': lat, 'lon': lon,
                'Population_2025': int(row['Population_2025']),
                'Zone': row['Zone'],
                'color': zone_color_rgb(row['Zone']),
                'Ward_Category': row['Ward_Category'],
                'risk_score': round(row['ward_risk'], 3),
            })
    return pd.DataFrame(rows)

# ============================================================
# INFRASTRUCTURE ZONE ANALYSIS
# ============================================================
@st.cache_data
def load_infrastructure():
    gdf = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
    gdf['lat'] = gdf.geometry.centroid.y
    gdf['lon'] = gdf.geometry.centroid.x
    return gdf

def assign_infra_zones(infra_gdf, risk_prob):
    df = infra_gdf.copy()
    weights = {'Healthcare': 1.10, 'Shelter': 1.05,
               'Police': 1.00, 'Fire_Station': 1.00, 'Waterways': 0.70}
    df['infra_risk'] = df['infrastructure_type'].map(
        lambda t: min(risk_prob * weights.get(t, 1.0), 1.0))
    df['Zone'] = df['infra_risk'].apply(classify_zone)
    df['color'] = df['Zone'].apply(lambda z: zone_color_rgb(z))
    return df

# ============================================================
# GENAI SUMMARY
# ============================================================
def generate_gemini_full_summary(risk_level, risk_prob, rainfall, soil_moisture,
                                  elevation_risk, alert_decision, zone_summary,
                                  infra_summary, gemini_api_key):
    try:
        import google.generativeai as genai
        if not gemini_api_key:
            return None, "Please enter Gemini API key in sidebar."
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
You are RainGuardAI — an expert flood emergency analyst for Mumbai, Maharashtra, India.
Generate a comprehensive, professional flood situation report based on the data below.

=== FLOOD PREDICTION ===
Risk Level: {risk_level}
Risk Probability: {risk_prob:.2%}
Rainfall: {rainfall:.1f} mm/hr
Soil Saturation: {soil_moisture:.2%}
Elevation Risk: {elevation_risk:.2f}
Alert Decision: {"SEND ALERT - Immediate Action Required" if alert_decision else "Continue Monitoring"}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== POPULATION ZONE SUMMARY ===
RED (HIGH RISK) Zone:
  - Wards: {zone_summary['red']['wards']}
  - Population at risk: {zone_summary['red']['population']:,}
  - Ward names: {zone_summary['red']['ward_names']}

ORANGE (MEDIUM RISK) Zone:
  - Wards: {zone_summary['orange']['wards']}
  - Population: {zone_summary['orange']['population']:,}
  - Ward names: {zone_summary['orange']['ward_names']}

GREEN (LOW RISK) Zone:
  - Wards: {zone_summary['green']['wards']}
  - Population: {zone_summary['green']['population']:,}
  - Ward names: {zone_summary['green']['ward_names']}

=== CRITICAL INFRASTRUCTURE IMPACT ===
RED Zone: {json.dumps(infra_summary['red'])}
ORANGE Zone: {json.dumps(infra_summary['orange'])}
GREEN Zone: {json.dumps(infra_summary['green'])}

Write a professional emergency situation report with these sections:

## SITUATION OVERVIEW
(3-4 sentences: what is happening, severity, affected area)

## POPULATION AT RISK
(Specific ward names, population numbers per zone, total at risk)

## CRITICAL INFRASTRUCTURE STATUS
(Which hospitals, shelters, police stations need immediate attention, zone by zone)

## IMMEDIATE ACTIONS REQUIRED
(Numbered list: 5-7 specific actions for authorities in the next 2 hours)

## 6-HOUR OUTLOOK
(What to expect, thresholds to watch, when to escalate)

## NDMA COMPLIANCE NOTES
(How this aligns with NDMA flood response guidelines)

Be specific with numbers. Use ward names. Be urgent but professional.
"""
        response = model.generate_content(prompt)
        return response.text, None
    except ImportError:
        return None, "Install: pip install google-generativeai"
    except Exception as e:
        return None, f"Gemini Error: {str(e)}"

# ============================================================
# NOTIFICATIONS
# ============================================================
def send_telegram_alert(bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        resp.raise_for_status()
        return True, "Telegram alert sent!"
    except Exception as e:
        return False, f"Telegram failed: {e}"

def send_email_alert(to_email, subject, message, smtp_server, smtp_port, from_email, password):
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart()
        msg['From'] = from_email; msg['To'] = to_email; msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls(); server.login(from_email, password)
        server.send_message(msg); server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, f"Email failed: {e}"

# ============================================================
# WEATHER DATA
# ============================================================
@st.cache_data(ttl=300)
def fetch_open_meteo_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        d = resp.json()
        cur = d.get('current', {})
        return {
            'rainfall':    cur.get('rain', 0) + cur.get('precipitation', 0),
            'humidity':    cur.get('relative_humidity_2m', 60) / 100,
            'temp':        cur.get('temperature_2m', 28),
            'description': 'Live data',
            'timestamp':   cur.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    except Exception:
        return {'rainfall': 0.0, 'humidity': 0.60, 'temp': 28.0,
                'description': 'Estimated', 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def estimate_soil_moisture(humidity, rainfall, temp):
    return min(humidity * 0.5 + min(rainfall / 50, 0.4) + max(0, (30 - temp) / 30) * 0.1, 1.0)

def get_elevation_risk(lat, lon):
    try:
        resp = requests.get(f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}", timeout=10)
        elev = resp.json()['results'][0]['elevation']
        risk = 0.9 if elev < 5 else 0.7 if elev < 10 else 0.5 if elev < 20 else 0.3 if elev < 50 else 0.1
        return risk, elev
    except Exception:
        return 0.7, None

# ------------------------------------------------------------------
# REAL ANN MODEL — loads the trained keras model + saved scaler
# 4 features: Intensity, soil_moisture_index, elevation_risk, humidity
# Falls back to formula proxy if model files are not found
# ------------------------------------------------------------------
ANN_MODEL_PATH  = BASE_DIR / "model" / "ann_flood_model.keras"
ANN_SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

# Feature order must match FEATURE_COLS in train_ann.py exactly
ANN_FEATURE_COLS = ["Intensity", "soil_moisture_index", "elevation_risk", "humidity"]

@st.cache_resource
def load_ann_model():
    """Load trained ANN model and scaler. Returns (model, scaler) or (None, None)."""
    try:
        from tensorflow.keras.models import load_model as keras_load
        if ANN_MODEL_PATH.exists() and ANN_SCALER_PATH.exists():
            ann   = keras_load(str(ANN_MODEL_PATH))
            sc    = joblib.load(str(ANN_SCALER_PATH))
            return ann, sc
        else:
            return None, None
    except Exception:
        return None, None

def ann_risk_proxy(rainfall, soil_moisture, elevation_risk, humidity=70.0):
    """
    Predict flood risk probability.
    Uses the real trained ANN (4 features) when available;
    falls back to weighted formula when model files are missing.
    
    Parameters
    ----------
    rainfall       : mm/hr  (= Intensity in training data)
    soil_moisture  : 0.0 – 1.0
    elevation_risk : 0.0 – 1.0
    humidity       : relative humidity 0–100 (default 70 for manual input)
    """
    ann_model, ann_scaler = load_ann_model()

    if ann_model is not None and ann_scaler is not None:
        try:
            X = np.array([[rainfall, soil_moisture, elevation_risk, humidity]])
            X_scaled = ann_scaler.transform(X)
            prob = float(ann_model.predict(X_scaled, verbose=0)[0][0])
            return min(max(prob, 0.0), 1.0)
        except Exception:
            pass  # fall through to formula proxy

    # Formula fallback (used when model/scaler.pkl not yet generated)
    return min(max(0.5 * (rainfall / 300) + 0.25 * soil_moisture +
                   0.15 * elevation_risk + 0.10 * (humidity / 100), 0), 1)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_rl_model():
    return joblib.load(RL_MODEL_PATH)

rl_model = load_rl_model()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuration")

data_mode = st.sidebar.radio("Data Source", ["Real-Time API", "Manual Input"])

if data_mode == "Real-Time API":
    st.sidebar.subheader("Location")
    latitude  = st.sidebar.number_input("Latitude",  value=19.0760, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.4f")
    st.sidebar.button("Refresh Now")
    weather_data = fetch_open_meteo_data(latitude, longitude)
    if weather_data:
        st.sidebar.success(f"Updated: {weather_data['timestamp']}")
        st.sidebar.metric("Rainfall", f"{weather_data['rainfall']:.1f} mm/hr")
        st.sidebar.metric("Humidity",  f"{weather_data['humidity']*100:.0f}%")
        soil_moisture  = estimate_soil_moisture(weather_data['humidity'], weather_data['rainfall'], weather_data['temp'])
        elevation_risk, elev = get_elevation_risk(latitude, longitude)
        if elev:
            st.sidebar.metric("Elevation", f"{elev:.1f}m")
        rainfall = weather_data['rainfall']
        # humidity as 0-100 for the ANN 4th feature
        humidity = weather_data['humidity'] * 100
    else:
        rainfall = 120; soil_moisture = 0.65; elevation_risk = 0.70; humidity = 70.0
else:
    st.sidebar.subheader("Manual Inputs")
    rainfall       = st.sidebar.slider("Rainfall (mm/hr)", 0, 300, 120)
    soil_moisture  = st.sidebar.slider("Soil Saturation",  0.0, 1.0, 0.65)
    elevation_risk = st.sidebar.slider("Elevation Risk",   0.0, 1.0, 0.70)
    humidity       = st.sidebar.slider("Humidity (%)",     0,   100, 70)

st.sidebar.markdown("---")
st.sidebar.subheader("Gemini AI")
enable_gemini  = st.sidebar.checkbox("Enable Gemini Summary")
gemini_api_key = ""
if enable_gemini:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    st.session_state['gemini_api_key'] = gemini_api_key

st.sidebar.markdown("---")
st.sidebar.subheader("Notifications")
notify_enabled = st.sidebar.checkbox("Enable Alerts")
notify_method = None
tg_token = tg_chat_id = ""
smtp_srv = "smtp.gmail.com"; smtp_port = 587
from_em = em_pass = to_em = ""
if notify_enabled:
    notify_method = st.sidebar.selectbox("Method", ["Telegram", "Email (SMTP)"])
    if notify_method == "Telegram":
        tg_token   = st.sidebar.text_input("Bot Token",  type="password")
        tg_chat_id = st.sidebar.text_input("Chat ID")
    elif notify_method == "Email (SMTP)":
        smtp_srv  = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.sidebar.number_input("Port", value=587)
        from_em   = st.sidebar.text_input("From Email")
        em_pass   = st.sidebar.text_input("Password", type="password")
        to_em     = st.sidebar.text_input("To Email")

# ============================================================
# CORE RISK COMPUTATION
# ============================================================
risk_prob  = ann_risk_proxy(rainfall, soil_moisture, elevation_risk, humidity)
risk_level = "HIGH" if risk_prob >= 0.75 else "MEDIUM" if risk_prob >= 0.50 else "LOW"

X_rl = pd.DataFrame({"risk_score": [risk_prob]})
rl_pred = rl_model.predict(X_rl)[0]
if risk_level == "HIGH":
    alert_decision = 1
    alert_reasoning = "HIGH risk — alert REQUIRED by safety protocol"
elif risk_level == "MEDIUM":
    alert_decision = rl_pred
    alert_reasoning = f"MEDIUM risk — RL suggests: {'ALERT' if rl_pred else 'MONITOR'}"
else:
    alert_decision = 0
    alert_reasoning = "LOW risk — continue monitoring"

# ============================================================
# TOP METRICS BAR
# ============================================================
st.markdown("---")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rainfall",        f"{rainfall:.1f} mm/hr")
m2.metric("Humidity",        f"{humidity:.0f}%")
m3.metric("Soil Saturation", f"{soil_moisture:.2f}")
m4.metric("Elevation Risk",  f"{elevation_risk:.2f}")
m5.metric("Flood Probability", f"{risk_prob:.2%}")
color_emoji = "🔴" if risk_level=="HIGH" else "🟠" if risk_level=="MEDIUM" else "🟢"
m6.metric("Risk Level", f"{color_emoji} {risk_level}")

if risk_level == "HIGH":
    st.error(f"CRITICAL FLOOD ALERT — Risk probability {risk_prob:.1%}. Immediate action required.")
elif risk_level == "MEDIUM":
    st.warning(f"ELEVATED FLOOD RISK — Risk probability {risk_prob:.1%}. Prepare response teams.")
else:
    st.success(f"LOW FLOOD RISK — Risk probability {risk_prob:.1%}. Routine monitoring.")

st.info(f"RL Alert Decision: {alert_reasoning}")
st.markdown("---")

# ============================================================
# LOAD AND COMPUTE ALL DATA
# ============================================================
pop_df    = load_population()
zone_df   = compute_population_zones(risk_prob, pop_df)
infra_gdf = load_infrastructure()
infra_df  = assign_infra_zones(infra_gdf, risk_prob)
ward_map  = build_ward_map_data(zone_df)

red_wards    = zone_df[zone_df['Zone'] == 'RED']
orange_wards = zone_df[zone_df['Zone'] == 'ORANGE']
green_wards  = zone_df[zone_df['Zone'] == 'GREEN']

red_infra    = infra_df[infra_df['Zone'] == 'RED']
orange_infra = infra_df[infra_df['Zone'] == 'ORANGE']
green_infra  = infra_df[infra_df['Zone'] == 'GREEN']

def infra_counts(df):
    if df.empty: return {}
    return df['infrastructure_type'].value_counts().to_dict()

zone_summary = {
    'red':    {'wards': len(red_wards),    'population': int(red_wards['Population_2025'].sum()),    'ward_names': ', '.join(red_wards['Ward'].tolist()) or 'None'},
    'orange': {'wards': len(orange_wards), 'population': int(orange_wards['Population_2025'].sum()), 'ward_names': ', '.join(orange_wards['Ward'].tolist()) or 'None'},
    'green':  {'wards': len(green_wards),  'population': int(green_wards['Population_2025'].sum()),  'ward_names': ', '.join(green_wards['Ward'].tolist()) or 'None'},
}
infra_summary = {
    'red':    infra_counts(red_infra),
    'orange': infra_counts(orange_infra),
    'green':  infra_counts(green_infra),
}

total_pop     = int(zone_df['Population_2025'].sum())
total_at_risk = zone_summary['red']['population'] + zone_summary['orange']['population']

infra_icons = {'Healthcare': 'Hospital', 'Shelter': 'Shelter', 'Police': 'Police Station', 'Fire_Station': 'Fire Station', 'Waterways': 'Waterway'}

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Population Zone Analysis",
    "Critical Infrastructure",
    "Combined Zone Map",
    "AI Situation Report"
])

# ============================================================
# TAB 1 — POPULATION ZONE ANALYSIS
# ============================================================
with tab1:
    st.subheader("Population Zone Analysis — Ward-wise Risk Classification")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mumbai Population", f"{total_pop:,}")
    c2.metric("RED Zone Population",    f"{zone_summary['red']['population']:,}",    f"{zone_summary['red']['wards']} wards")
    c3.metric("ORANGE Zone Population", f"{zone_summary['orange']['population']:,}", f"{zone_summary['orange']['wards']} wards")
    c4.metric("GREEN Zone Population",  f"{zone_summary['green']['population']:,}",  f"{zone_summary['green']['wards']} wards")

    st.markdown("---")
    col_map, col_summary = st.columns([3, 2])

    with col_map:
        st.markdown(f"#### Ward Map — {risk_level} Risk Scenario")
        st.caption("Each circle = one ward. Size = population. Color = flood risk zone.")

        ward_layer = pdk.Layer(
            "ScatterplotLayer",
            data=ward_map,
            get_position="[lon, lat]",
            get_color="color",
            get_radius="Population_2025 / 20",
            radius_scale=1,
            radius_min_pixels=12,
            radius_max_pixels=60,
            pickable=True,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[ward_layer],
            initial_view_state=pdk.ViewState(latitude=19.08, longitude=72.88, zoom=10, pitch=15),
            tooltip={"html": "<b>Ward {Ward}</b><br/>Zone: {Zone}<br/>Population: {Population_2025}<br/>Category: {Ward_Category}<br/>Risk Score: {risk_score}",
                     "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}},
            map_style="mapbox://styles/mapbox/dark-v10",
        ))

        l1, l2, l3 = st.columns(3)
        l1.markdown("🔴 **RED** — High Risk (>=75%)")
        l2.markdown("🟠 **ORANGE** — Medium Risk (50-75%)")
        l3.markdown("🟢 **GREEN** — Low Risk (<50%)")

    with col_summary:
        st.markdown("#### Ward-wise Summary Table")

        def style_zone(val):
            return {"RED": "background-color:#FFE5E5;color:#C00000;font-weight:bold",
                    "ORANGE": "background-color:#FFF3E0;color:#E65100;font-weight:bold",
                    "GREEN":  "background-color:#E8F5E9;color:#1B5E20;font-weight:bold"}.get(val, "")

        display_df = zone_df[['Ward', 'Ward_Category', 'Population_2025', 'Zone']].copy()
        display_df.columns = ['Ward', 'Category', 'Population (2025)', 'Risk Zone']
        display_df['Population (2025)'] = display_df['Population (2025)'].apply(lambda x: f"{x:,}")
        st.dataframe(display_df.style.applymap(style_zone, subset=['Risk Zone']), height=450, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Zone Breakdown by Ward")
    zc1, zc2, zc3 = st.columns(3)

    with zc1:
        st.markdown("<div style='background:#FFE5E5;border-left:6px solid #C00000;padding:12px;border-radius:8px'><b style='color:#C00000'>🔴 HIGH RISK ZONE</b></div>", unsafe_allow_html=True)
        st.write("")
        if not red_wards.empty:
            for _, r in red_wards.iterrows():
                st.markdown(f"**Ward {r['Ward']}** — {r['Ward_Category']}")
                st.write(f"Population: {r['Population_2025']:,}")
                st.write(f"Area: {r['Area_in_Sq_km']} km²")
                st.markdown("---")
        else:
            st.success("No wards in HIGH RISK zone.")

    with zc2:
        st.markdown("<div style='background:#FFF3E0;border-left:6px solid #E65100;padding:12px;border-radius:8px'><b style='color:#E65100'>🟠 MEDIUM RISK ZONE</b></div>", unsafe_allow_html=True)
        st.write("")
        if not orange_wards.empty:
            for _, r in orange_wards.iterrows():
                st.markdown(f"**Ward {r['Ward']}** — {r['Ward_Category']}")
                st.write(f"Population: {r['Population_2025']:,}")
                st.write(f"Area: {r['Area_in_Sq_km']} km²")
                st.markdown("---")
        else:
            st.success("No wards in MEDIUM RISK zone.")

    with zc3:
        st.markdown("<div style='background:#E8F5E9;border-left:6px solid #1B5E20;padding:12px;border-radius:8px'><b style='color:#1B5E20'>🟢 LOW RISK ZONE</b></div>", unsafe_allow_html=True)
        st.write("")
        if not green_wards.empty:
            for _, r in green_wards.iterrows():
                st.markdown(f"**Ward {r['Ward']}** — {r['Ward_Category']}")
                st.write(f"Population: {r['Population_2025']:,}")
                st.write(f"Area: {r['Area_in_Sq_km']} km²")
                st.markdown("---")
        else:
            st.info("All wards in higher risk zones.")

# ============================================================
# TAB 2 — CRITICAL INFRASTRUCTURE
# ============================================================
with tab2:
    st.subheader("Critical Infrastructure Risk — Zone-wise Classification")

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Total Infrastructure Assets", len(infra_df))
    i2.metric("RED Zone Assets",    len(red_infra))
    i3.metric("ORANGE Zone Assets", len(orange_infra))
    i4.metric("GREEN Zone Assets",  len(green_infra))

    st.markdown("---")
    col_imap, col_isummary = st.columns([3, 2])

    with col_imap:
        st.markdown("#### Infrastructure Risk Map")
        st.caption("Each point = one critical asset. Color = flood risk zone.")

        infra_map_df = infra_df[['lat', 'lon', 'infrastructure_type', 'Zone', 'color', 'area_m2']].dropna(subset=['lat', 'lon']).copy()
        infra_map_df['color'] = infra_map_df['color'].tolist()

        infra_layer = pdk.Layer(
            "ScatterplotLayer",
            data=infra_map_df,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=200,
            radius_min_pixels=8,
            radius_max_pixels=25,
            pickable=True,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[infra_layer],
            initial_view_state=pdk.ViewState(latitude=18.97, longitude=72.82, zoom=11, pitch=20),
            tooltip={"html": "<b>{infrastructure_type}</b><br/>Zone: {Zone}<br/>Area: {area_m2:.0f} m²",
                     "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}},
            map_style="mapbox://styles/mapbox/dark-v10",
        ))

        l1, l2, l3 = st.columns(3)
        l1.markdown("🔴 **RED** — Immediate support needed")
        l2.markdown("🟠 **ORANGE** — Prepare standby")
        l3.markdown("🟢 **GREEN** — Monitor")

    with col_isummary:
        st.markdown("#### Asset Risk Summary Table")

        def style_infra_zone(val):
            return {"RED": "background-color:#FFE5E5;color:#C00000;font-weight:bold",
                    "ORANGE": "background-color:#FFF3E0;color:#E65100;font-weight:bold",
                    "GREEN": "background-color:#E8F5E9;color:#1B5E20;font-weight:bold"}.get(val, "")

        infra_display = infra_df[['infrastructure_type', 'Zone', 'area_m2']].copy()
        infra_display.columns = ['Infrastructure Type', 'Risk Zone', 'Area (m²)']
        infra_display['Area (m²)'] = infra_display['Area (m²)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(infra_display.style.applymap(style_infra_zone, subset=['Risk Zone']), height=400, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Infrastructure Breakdown by Zone")
    ic1, ic2, ic3 = st.columns(3)

    def render_infra_card(df_zone, title_html):
        st.markdown(title_html, unsafe_allow_html=True)
        st.write("")
        if not df_zone.empty:
            for infra_type, cnt in df_zone['infrastructure_type'].value_counts().items():
                label = infra_icons.get(infra_type, infra_type)
                st.write(f"**{label}**: {cnt} assets")
            st.write(f"**Total**: {len(df_zone)} assets")
        else:
            st.success("No assets in this zone.")

    with ic1:
        render_infra_card(red_infra,    "<div style='background:#FFE5E5;border-left:6px solid #C00000;padding:12px;border-radius:8px'><b style='color:#C00000'>🔴 HIGH RISK</b></div>")
    with ic2:
        render_infra_card(orange_infra, "<div style='background:#FFF3E0;border-left:6px solid #E65100;padding:12px;border-radius:8px'><b style='color:#E65100'>🟠 MEDIUM RISK</b></div>")
    with ic3:
        render_infra_card(green_infra,  "<div style='background:#E8F5E9;border-left:6px solid #1B5E20;padding:12px;border-radius:8px'><b style='color:#1B5E20'>🟢 LOW RISK</b></div>")

    st.markdown("---")
    if alert_decision == 1:
        st.error("SEND ALERT — Immediate Response Required")
        if st.button("CONFIRM AND SEND ALERT", type="primary"):
            msg_text = (f"FLOOD ALERT - {risk_level} RISK\n"
                       f"Risk: {risk_prob:.1%}\n"
                       f"Red zone: {zone_summary['red']['population']:,} people\n"
                       f"Critical assets at risk: {len(red_infra)}\n"
                       f"Action: Immediate evacuation")
            if notify_enabled and notify_method == "Telegram" and tg_token and tg_chat_id:
                ok, msg = send_telegram_alert(tg_token, tg_chat_id, msg_text)
                st.success(msg) if ok else st.error(msg)
            elif notify_enabled and notify_method == "Email (SMTP)" and from_em and em_pass and to_em:
                ok, msg = send_email_alert(to_em, f"FLOOD ALERT - {risk_level}", msg_text, smtp_srv, smtp_port, from_em, em_pass)
                st.success(msg) if ok else st.error(msg)
            else:
                st.warning("Enable notifications in sidebar to send alerts.")
    else:
        st.success("NO ALERT — Continue Monitoring")

# ============================================================
# TAB 3 — COMBINED ZONE MAP
# ============================================================
with tab3:
    st.subheader("Combined Zone Map — Population + Infrastructure + Flood Zones")
    st.caption("Large circles = wards (population). Small circles = infrastructure. Shaded area = GIS low-lying flood zones.")

    combined_col, legend_col = st.columns([4, 1])

    with combined_col:
        ward_layer_combined = pdk.Layer(
            "ScatterplotLayer",
            data=ward_map,
            get_position="[lon, lat]",
            get_color="color",
            get_radius="Population_2025 / 15",
            radius_scale=1, radius_min_pixels=15, radius_max_pixels=70,
            pickable=True, opacity=0.6,
        )

        infra_map_df2 = infra_df[['lat','lon','infrastructure_type','Zone','color','area_m2']].dropna(subset=['lat','lon']).copy()
        infra_map_df2['color'] = infra_map_df2['color'].tolist()

        infra_layer_combined = pdk.Layer(
            "ScatterplotLayer",
            data=infra_map_df2,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=300, radius_min_pixels=10, radius_max_pixels=20,
            pickable=True, opacity=0.95,
        )

        flood_color = ([220, 38, 38, 80]  if risk_level == "HIGH" else
                       [249, 115, 22, 70] if risk_level == "MEDIUM" else
                       [34, 197, 94, 60])

        layers = [ward_layer_combined, infra_layer_combined]
        try:
            low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
            low_zones_json = json.loads(low_zones.to_json())
            geojson_layer = pdk.Layer(
                "GeoJsonLayer",
                data=low_zones_json,
                get_fill_color=flood_color,
                get_line_color=[255, 255, 255, 100],
                line_width_min_pixels=1, opacity=0.5, pickable=False,
            )
            layers = [geojson_layer, ward_layer_combined, infra_layer_combined]
        except Exception:
            pass

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=19.05, longitude=72.88, zoom=10, pitch=25),
            tooltip={"html": "<b>Ward: {Ward}</b><br/>{infrastructure_type}<br/>Zone: {Zone}<br/>Pop: {Population_2025}",
                     "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}},
            map_style="mapbox://styles/mapbox/dark-v10",
        ))

    with legend_col:
        st.markdown("#### Legend")
        st.markdown("**🔴 RED ZONE**")
        st.caption("High flood risk. Immediate evacuation.")
        st.markdown("**🟠 ORANGE ZONE**")
        st.caption("Medium risk. Standby alert.")
        st.markdown("**🟢 GREEN ZONE**")
        st.caption("Low risk. Routine monitoring.")
        st.markdown("---")
        st.caption("Large circles = wards")
        st.caption("Small circles = infrastructure")
        st.caption("Shaded area = GIS flood zone")
        st.markdown("---")
        st.markdown("**Population at Risk**")
        pop_risk_pct = (total_at_risk / total_pop * 100) if total_pop else 0
        st.metric("", f"{total_at_risk:,}", f"{pop_risk_pct:.1f}% of total")

    st.markdown("---")
    st.markdown("#### Situation Snapshot")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Total Wards",           24)
    s2.metric("High Risk Wards",        zone_summary['red']['wards'])
    s3.metric("Medium Risk Wards",      zone_summary['orange']['wards'])
    s4.metric("Low Risk Wards",         zone_summary['green']['wards'])
    s5.metric("Infra at HIGH Risk",     len(red_infra))
    s6.metric("Population at Risk",     f"{total_at_risk:,}")

# ============================================================
# TAB 4 — AI SITUATION REPORT
# ============================================================
with tab4:
    st.subheader("AI-Generated Situation Report")
    st.caption("Gemini AI reads ALL zone data — flood probability, ward populations, infrastructure status — and writes a professional emergency report.")

    # Automated summary (no API key needed)
    st.markdown("#### Automated Summary")
    auto_col1, auto_col2 = st.columns(2)

    flood_bg     = "#FFE5E5" if risk_level=="HIGH" else "#FFF3E0" if risk_level=="MEDIUM" else "#E8F5E9"
    flood_border = "#C00000" if risk_level=="HIGH" else "#E65100" if risk_level=="MEDIUM" else "#1B5E20"

    with auto_col1:
        st.markdown(f"""
        <div style="background:{flood_bg};border-left:6px solid {flood_border};padding:16px;border-radius:8px">
        <h4 style="color:{flood_border}">Flood Risk: {risk_level}</h4>
        <p>Probability: <strong>{risk_prob:.1%}</strong></p>
        <p>Rainfall: <strong>{rainfall:.1f} mm/hr</strong></p>
        <p>Soil Saturation: <strong>{soil_moisture:.1%}</strong></p>
        <p>Alert: <strong>{"SEND ALERT" if alert_decision else "MONITOR"}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with auto_col2:
        st.markdown(f"""
        <div style="background:#E3F2FD;border-left:6px solid #1565C0;padding:16px;border-radius:8px">
        <h4 style="color:#1565C0">Population Impact</h4>
        <p>🔴 <strong>{zone_summary['red']['wards']} wards</strong> in HIGH risk — 
           <strong>{zone_summary['red']['population']:,}</strong> people</p>
        <p>🟠 <strong>{zone_summary['orange']['wards']} wards</strong> in MEDIUM risk — 
           <strong>{zone_summary['orange']['population']:,}</strong> people</p>
        <p>🟢 <strong>{zone_summary['green']['wards']} wards</strong> in LOW risk — 
           <strong>{zone_summary['green']['population']:,}</strong> people</p>
        <p>Total at risk: <strong>{total_at_risk:,}</strong> ({total_at_risk/total_pop*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#F3E5F5;border-left:6px solid #6A1B9A;padding:16px;border-radius:8px;margin-top:12px">
    <h4 style="color:#6A1B9A">Critical Infrastructure Status</h4>
    <p>🔴 <strong>{len(red_infra)}</strong> assets HIGH risk: 
       {", ".join(f"{v} {infra_icons.get(k,k)}" for k,v in infra_counts(red_infra).items()) or "None"}</p>
    <p>🟠 <strong>{len(orange_infra)}</strong> assets MEDIUM risk: 
       {", ".join(f"{v} {infra_icons.get(k,k)}" for k,v in infra_counts(orange_infra).items()) or "None"}</p>
    <p>🟢 <strong>{len(green_infra)}</strong> assets LOW risk: 
       {", ".join(f"{v} {infra_icons.get(k,k)}" for k,v in infra_counts(green_infra).items()) or "None"}</p>
    </div>
    """, unsafe_allow_html=True)

    # Gemini section
    st.markdown("---")
    st.markdown("#### Gemini AI Comprehensive Report")

    if not enable_gemini:
        st.info("Enable Gemini AI in the sidebar and enter your API key to generate a full situation report.")
    else:
        if st.button("Generate AI Situation Report", type="primary"):
            with st.spinner("Gemini is analysing all zone data and writing the report..."):
                summary_text, err = generate_gemini_full_summary(
                    risk_level, risk_prob, rainfall, soil_moisture, elevation_risk,
                    alert_decision, zone_summary, infra_summary,
                    st.session_state.get('gemini_api_key', '')
                )
                if summary_text:
                    st.session_state['ai_full_summary'] = summary_text
                else:
                    st.error(err)

        if st.session_state.get('ai_full_summary'):
            st.markdown(st.session_state['ai_full_summary'])
            st.download_button(
                "Download Report",
                st.session_state['ai_full_summary'],
                file_name=f"RainGuardAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    if st.session_state.alert_history:
        st.markdown("---")
        st.subheader("Alert History")
        st.dataframe(pd.DataFrame(st.session_state.alert_history), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"RainGuardAI  |  Risk: {risk_level} ({risk_prob:.1%})  |  "
    f"Population at risk: {total_at_risk:,}  |  "
    f"Updated: {datetime.now().strftime('%H:%M:%S')}  |  "
    "ML + GIS + GenAI + RL"
)