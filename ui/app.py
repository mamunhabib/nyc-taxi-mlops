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
    '<div class="sub-header">Production Model Inference powered by FastAPI, XGBoost (@champion), and MLflow</div>',
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
        st.session_state["dist_sel"] = 16.5
    if st.button("🌳 Central Park → Midtown East"):
        st.session_state["pu_sel"] = "43"
        st.session_state["do_sel"] = "162"
        st.session_state["dist_sel"] = 2.4
    if st.button("🏙️ East Harlem → Williamsburg"):
        st.session_state["pu_sel"] = "74"
        st.session_state["do_sel"] = "255"
        st.session_state["dist_sel"] = 7.8
    if st.button("✈️ LaGuardia Airport → Upper East Side"):
        st.session_state["pu_sel"] = "138"
        st.session_state["do_sel"] = "236"
        st.session_state["dist_sel"] = 8.2

    st.divider()
    st.header("🔗 MLOps Platform Observability")
    st.markdown("- [Evidently Drift Monitoring (8086)](http://localhost:8086)")
    st.markdown("- [MLflow Model Registry (8087)](http://localhost:8087)")
    st.markdown("- [MinIO Object Storage (8088)](http://localhost:8088)")
    st.markdown("- [FastAPI Swagger Docs (8085)](http://localhost:8085/docs)")


# Form Inputs
zone_keys = list(TAXI_ZONES.keys())

default_pu = st.session_state.get("pu_sel", "43")
default_do = st.session_state.get("do_sel", "161")
default_dist = float(st.session_state.get("dist_sel", 2.5))

col_pu, col_do, col_dist = st.columns(3)

with col_pu:
    pu_index = zone_keys.index(default_pu) if default_pu in zone_keys else 0
    selected_pu = st.selectbox(
        "📍 Pick-up Location",
        options=zone_keys,
        index=pu_index,
        format_func=lambda x: f"[{x}] {TAXI_ZONES.get(x, 'Zone ' + str(x))}",
    )

with col_do:
    do_index = zone_keys.index(default_do) if default_do in zone_keys else 0
    selected_do = st.selectbox(
        "🏁 Drop-off Location",
        options=zone_keys,
        index=do_index,
        format_func=lambda x: f"[{x}] {TAXI_ZONES.get(x, 'Zone ' + str(x))}",
    )

with col_dist:
    trip_distance = st.slider(
        "🛣️ Trip Distance (miles)",
        min_value=0.1,
        max_value=35.0,
        value=default_dist,
        step=0.1,
    )

st.write("")
predict_clicked = st.button("🚀 Calculate Estimated Trip Duration", type="primary", use_container_width=True)

if predict_clicked:
    payload = {
        "PULocationID": selected_pu,
        "DOLocationID": selected_do,
        "trip_distance": float(trip_distance),
    }

    with st.spinner("Querying Champion XGBoost inference service..."):
        start_time = time.time()
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
            latency_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                result = resp.json()
                duration = result["predicted_duration_minutes"]
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
