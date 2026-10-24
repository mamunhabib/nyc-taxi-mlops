import os
import time
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (NYC Taxi Amber & Dark Aesthetics)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f59e0b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #9ca3af;
        letter-spacing: 0.05em;
    }
    .badge-live {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Popular NYC Taxi Zones Lookup Table (ID -> Name)
TAXI_ZONES = {
    "43": "Central Park (Manhattan)",
    "236": "Upper East Side North (Manhattan)",
    "237": "Upper East Side South (Manhattan)",
    "161": "Midtown Center (Manhattan)",
    "162": "Midtown East (Manhattan)",
    "230": "Times Sq / Theatre District (Manhattan)",
    "142": "Lincoln Square East (Manhattan)",
    "74": "East Harlem North (Manhattan)",
    "75": "East Harlem South (Manhattan)",
    "41": "Central Harlem (Manhattan)",
    "42": "Central Harlem North (Manhattan)",
    "166": "Morningside Heights (Manhattan)",
    "244": "Washington Heights South (Manhattan)",
    "132": "JFK Airport (Queens)",
    "138": "LaGuardia Airport (Queens)",
    "7": "Astoria (Queens)",
    "95": "Forest Hills (Queens)",
    "80": "East Williamsburg (Brooklyn)",
    "255": "Williamsburg North (Brooklyn)",
    "256": "Williamsburg South (Brooklyn)",
    "61": "Crown Heights North (Brooklyn)",
    "33": "Brooklyn Heights (Brooklyn)",
}

# Centroid coordinates (lat, lon) for NYC taxi zones to calculate realistic road distances
ZONE_COORDS = {
    "43": (40.7812, -73.9665),   # Central Park
    "236": (40.7765, -73.9535),  # Upper East Side North
    "237": (40.7705, -73.9590),  # Upper East Side South
    "161": (40.7565, -73.9800),  # Midtown Center
    "162": (40.7540, -73.9740),  # Midtown East
    "230": (40.7580, -73.9855),  # Times Sq / Theatre District
    "142": (40.7738, -73.9820),  # Lincoln Square East
    "74": (40.7980, -73.9350),   # East Harlem North
    "75": (40.7910, -73.9440),   # East Harlem South
    "41": (40.8060, -73.9490),   # Central Harlem
    "42": (40.8180, -73.9400),   # Central Harlem North
    "166": (40.8100, -73.9620),  # Morningside Heights
    "244": (40.8400, -73.9390),  # Washington Heights South
    "132": (40.6413, -73.7781),  # JFK Airport
    "138": (40.7769, -73.8740),  # LaGuardia Airport
    "7": (40.7640, -73.9230),    # Astoria
    "95": (40.7180, -73.8450),   # Forest Hills
    "80": (40.7120, -73.9350),   # East Williamsburg
    "255": (40.7190, -73.9570),  # Williamsburg North
    "256": (40.7090, -73.9580),  # Williamsburg South
    "61": (40.6720, -73.9430),   # Crown Heights North
    "33": (40.6960, -73.9930),   # Brooklyn Heights
}


def estimate_distance(pu_id: str, do_id: str) -> float:
    """Calculate realistic NYC road distance (miles) between taxi zones."""
    if pu_id == do_id:
        return 1.1
    c1 = ZONE_COORDS.get(str(pu_id))
    c2 = ZONE_COORDS.get(str(do_id))
    if not c1 or not c2:
        return 3.0
    lat1, lon1 = c1
    lat2, lon2 = c2
    # Approximate Manhattan street grid / highway distance in miles
    dlat = abs(lat1 - lat2) * 69.0
    dlon = abs(lon1 - lon2) * 52.0
    import math
    euclid = math.hypot(dlat, dlon)
    return round(max(0.5, euclid * 1.32), 1)


# Resolve backend URL (defaults to localhost for local test, or internal cluster DNS in K8s)
BACKEND_URL = os.getenv("SERVING_URL", "http://localhost:8085").rstrip("/")


def check_backend_health() -> tuple[bool, str]:
    """Verify connectivity to FastAPI serving backend."""
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if res.status_code == 200:
            data = res.json()
            return True, data.get("model_version", "champion")
    except Exception:
        pass
    return False, "offline"


# Header
st.markdown('<div class="main-header">🚕 NYC Taxi Trip Duration Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production Model Inference powered by FastAPI, MLflow Champion Registry, and Evidently AI</div>',
    unsafe_allow_html=True,
)

# Backend status check
is_healthy, model_tag = check_backend_health()
col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    if is_healthy:
        st.markdown(
            f'Status: <span class="badge-live">● CONNECTED</span> &nbsp; Model: <b>{model_tag}</b> &nbsp; Endpoint: <code>{BACKEND_URL}</code>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'Status: <span class="badge-offline">● BACKEND OFFLINE</span> &nbsp; Target: <code>{BACKEND_URL}</code> (FastAPI must be running)',
            unsafe_allow_html=True,
        )

st.divider()

# Sidebar: Quick Presets & Platform Links
with st.sidebar:
    st.header("⚡ Quick Preset Trips")
    if st.button("✈️ JFK Airport → Times Square"):
        st.session_state["pu_sel"] = "132"
        st.session_state["do_sel"] = "230"
        st.session_state["dist_val"] = 17.8
        st.rerun()
    if st.button("🌳 Central Park → Midtown East"):
        st.session_state["pu_sel"] = "43"
        st.session_state["do_sel"] = "162"
        st.session_state["dist_val"] = 2.4
        st.rerun()
    if st.button("🏙️ East Harlem → Williamsburg"):
        st.session_state["pu_sel"] = "74"
        st.session_state["do_sel"] = "255"
        st.session_state["dist_val"] = 7.4
        st.rerun()
    if st.button("✈️ LaGuardia Airport → Upper East Side"):
        st.session_state["pu_sel"] = "138"
        st.session_state["do_sel"] = "236"
        st.session_state["dist_val"] = 5.5
        st.rerun()

    st.divider()
    st.header("🔗 MLOps Platform Observability")
    st.markdown("- [Kibana Telemetry & Logs (8084)](http://localhost:8084)")
    st.markdown("- [Evidently Drift Monitoring (8086)](http://localhost:8086)")
    st.markdown("- [MLflow Model Registry (8087)](http://localhost:8087)")
    st.markdown("- [MinIO Object Storage (8088)](http://localhost:8088)")
    st.markdown("- [FastAPI Swagger Docs (8085)](http://localhost:8085/docs)")


# Form Inputs & Reactive State
zone_keys = list(TAXI_ZONES.keys())

if "pu_sel" not in st.session_state:
    st.session_state["pu_sel"] = "43"
if "do_sel" not in st.session_state:
    st.session_state["do_sel"] = "161"
if "dist_val" not in st.session_state:
    st.session_state["dist_val"] = estimate_distance(st.session_state["pu_sel"], st.session_state["do_sel"])


def on_route_change():
    """Automatically recalculate estimated trip distance when pickup/dropoff zones change."""
    pu = st.session_state.get("pu_sel", "43")
    do = st.session_state.get("do_sel", "161")
    st.session_state["dist_val"] = estimate_distance(pu, do)


col_pu, col_do, col_dist = st.columns(3)

with col_pu:
    selected_pu = st.selectbox(
        "📍 Pick-up Location",
        options=zone_keys,
        key="pu_sel",
        on_change=on_route_change,
        format_func=lambda x: f"[{x}] {TAXI_ZONES.get(x, 'Zone ' + str(x))}",
    )

with col_do:
    selected_do = st.selectbox(
        "🏁 Drop-off Location",
        options=zone_keys,
        key="do_sel",
        on_change=on_route_change,
        format_func=lambda x: f"[{x}] {TAXI_ZONES.get(x, 'Zone ' + str(x))}",
    )

with col_dist:
    trip_distance = st.slider(
        "🛣️ Trip Distance (miles)",
        min_value=0.1,
        max_value=35.0,
        step=0.1,
        key="dist_val",
        help="Automatically calculated for the chosen route. You can also adjust manually.",
    )

st.write("")
predict_clicked = st.button("🚀 Calculate Estimated Trip Duration", type="primary", use_container_width=True)

if predict_clicked:
    payload = {
        "PULocationID": selected_pu,
        "DOLocationID": selected_do,
        "trip_distance": float(trip_distance),
    }

    with st.spinner("Querying Champion inference service..."):
        start_time = time.time()
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
            latency_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                result = resp.json()
                duration = result.get("trip_duration", result.get("predicted_duration_minutes", 0.0))
                model_used = result.get("model_version", "champion")

                # Realistic NYC Taxi Fare estimate ($3 base + $1.75/mi + $0.50/min)
                fare_estimate = 3.00 + (1.75 * trip_distance) + (0.50 * duration)

                st.success("✅ Prediction Successfully Generated!")

                # Metric Cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimated Duration</div>
                            <div class="metric-value">{duration:.1f} <span style="font-size: 1.2rem; color: #9ca3af;">mins</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimated Fare</div>
                            <div class="metric-value">${fare_estimate:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Inference Latency</div>
                            <div class="metric-value">{latency_ms:.0f} <span style="font-size: 1.2rem; color: #9ca3af;">ms</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Model Deployed</div>
                            <div class="metric-value" style="font-size: 1.4rem; padding-top: 8px;">{model_used}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.caption(f"Payload dispatched to `{BACKEND_URL}/predict`: `{payload}`")
            else:
                st.error(f"Inference error ({resp.status_code}): {resp.text}")
        except Exception as err:
            st.error(f"Failed to connect to FastAPI Serving at `{BACKEND_URL}`. Details: {err}")
            st.info("Tip: Ensure the serving container is running with `kubectl apply -f k8s/apps/serving.yaml` or `./venv/bin/uvicorn serving.app:app --port 8085`.")
