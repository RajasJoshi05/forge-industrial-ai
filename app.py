import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from industrial_ai.maintenance.predictor import (  # noqa: E402
    PredictiveMaintenancePredictor,
)
from industrial_ai.monitoring.fleet_analytics import (  # noqa: E402
    build_fleet_snapshot,
    summarize_fleet,
)
from industrial_ai.monitoring.prediction_history import (  # noqa: E402
    record_prediction,
)
from industrial_ai.monitoring.sensor_health import (  # noqa: E402
    monitor_sensor_health,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_features.csv"
HISTORY_PATH = PROJECT_ROOT / "artifacts" / "reports" / "prediction_history.csv"

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="FORGE | Industrial Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B1119;
        --panel: #121D2A;
        --panel-soft: #172536;
        --border: #26384A;
        --text: #EAF1F7;
        --muted: #91A5B8;
        --teal: #36D6C3;
        --amber: #FFBE69;
        --red: #FF727C;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, #152739 0%, transparent 42%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 25, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #0D1722;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        letter-spacing: -0.035em;
        font-weight: 600 !important;
    }

    h1, h2, h3, p, label, li {
        font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    }

    p, label, li {
        color: #C7D3DF;
    }

    code, .mono {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #172536, #111C29);
        border: 1px solid var(--border);
        padding: 18px 20px;
        border-radius: 12px;
        min-height: 125px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.075em;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.65rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 12px;
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] {
        background: var(--panel);
    }

    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 600;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--teal);
        color: #071B1B;
        border: 1px solid var(--teal);
    }

    .stButton > button:hover {
        border-color: var(--teal);
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] > div {
        background-color: #172536;
        border-color: var(--border);
        border-radius: 8px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    .forge-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        padding: 23px 26px;
        margin-bottom: 20px;
        border: 1px solid #2B4558;
        border-radius: 14px;
        background: linear-gradient(115deg, #172B3B, #111C29 70%);
    }

    .forge-brand {
        color: #F1F6FA;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.09em;
    }

    .forge-subtitle {
        color: #91A5B8;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.09em;
        margin-top: 5px;
    }

    .forge-status {
        display: inline-block;
        padding: 8px 12px;
        border: 1px solid #287B72;
        border-radius: 6px;
        background: #12352F;
        color: #62E5D3;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }

    .section-label {
        color: #8EA5BA;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 8px 0 14px 0;
    }

    .status-normal {
        color: #58E0CC;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
    }

    .status-warning {
        color: #FFBE69;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
    }

    .footer-note {
        color: #71869A;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        border-top: 1px solid var(--border);
        padding-top: 16px;
        margin-top: 30px;
    }

    @media (max-width: 700px) {
        .forge-header {
            align-items: flex-start;
            flex-direction: column;
            padding: 18px;
        }

        .forge-brand {
            font-size: 1.25rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# DATA ACCESS
# ---------------------------------------------------------


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)

    required_columns = {
        "machine_id",
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
        "failure_within_next_30min",
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    return data


@st.cache_resource
def load_predictor() -> PredictiveMaintenancePredictor:
    return PredictiveMaintenancePredictor()


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="forge-header">
        <div>
            <div class="forge-brand">
                FORGE <span style="color:#36D6C3">//</span> INDUSTRIAL AI
            </div>
            <div class="forge-subtitle">
                CONDITION MONITORING / PREDICTIVE MAINTENANCE / CONTROL CENTER
            </div>
        </div>
        <div class="forge-status">● APPLICATION ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "DEVELOPMENT ENVIRONMENT · Synthetic historical telemetry · Experimental model predictions"
)

if not DATA_PATH.is_file():
    st.error(f"Processed dataset not found: {DATA_PATH}")
    st.stop()

try:
    data = load_data(str(DATA_PATH))
    predictor = load_predictor()
except Exception as exc:
    st.error(f"System initialization failed: {exc}")
    st.stop()

machines = sorted(data["machine_id"].dropna().unique().tolist())

if not machines:
    st.error("No machines were found in the dataset.")
    st.stop()

# ---------------------------------------------------------
# SENSOR HEALTH COMPUTATION
# ---------------------------------------------------------

try:
    sensor_health_rows, sensor_health_summary = monitor_sensor_health(data)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Sensor-health monitoring failed: {exc}")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 6px 0 18px 0;">
        <div style="font-family:'IBM Plex Mono',monospace;
                    font-size:0.72rem;color:#36D6C3;
                    letter-spacing:0.13em;">FORGE / NAVIGATION</div>
        <div style="font-size:1.3rem;font-weight:700;
                    color:#EAF1F7;margin-top:7px;">Asset Monitor</div>
        <div style="font-size:0.8rem;color:#91A5B8;margin-top:4px;">
            Industrial intelligence suite
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_machine = st.sidebar.selectbox("MACHINE ASSET", machines)

st.sidebar.divider()
st.sidebar.markdown("**SYSTEM CONTEXT**")
st.sidebar.caption("Model: Balanced Logistic Regression")
st.sidebar.caption(f"Fleet size: {len(machines)} assets")
st.sidebar.caption(f"Dataset rows: {len(data):,}")

st.sidebar.divider()
st.sidebar.markdown("**DATA SOURCE**")
st.sidebar.warning("Synthetic data — not live telemetry")

if st.sidebar.button("Refresh dataset view", use_container_width=True):
    load_data.clear()
    st.rerun()

# ---------------------------------------------------------
# FLEET ANALYTICS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-label">01 / Fleet intelligence</div>',
    unsafe_allow_html=True,
)

try:
    fleet_snapshot = build_fleet_snapshot(data, predictor)
    fleet_summary = summarize_fleet(fleet_snapshot)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Fleet analytics could not be generated: {exc}")
    st.stop()

with st.container(border=True):
    fleet_a, fleet_b, fleet_c, fleet_d = st.columns(4)

    fleet_a.metric("MACHINES", fleet_summary["machines"])
    fleet_b.metric("AI WARNINGS", fleet_summary["warnings"])
    fleet_c.metric("AI NORMAL", fleet_summary["normal"])
    fleet_d.metric(
        "SYNTHETIC POSITIVE LABELS",
        fleet_summary["synthetic_positive_labels"],
    )

    st.caption(
        "Each row represents the latest available observation per machine. "
        "These readings come from historical synthetic data."
    )

    st.dataframe(
        fleet_snapshot.sort_values("warning_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "machine_id": st.column_config.TextColumn("MACHINE"),
            "timestamp": st.column_config.DatetimeColumn(
                "OBSERVATION TIME",
                format="YYYY-MM-DD HH:mm:ss",
            ),
            "temperature_c": st.column_config.NumberColumn("TEMP °C", format="%.2f"),
            "vibration_mm_s": st.column_config.NumberColumn("VIBRATION", format="%.2f"),
            "rotational_speed_rpm": st.column_config.NumberColumn("SPEED RPM", format="%.0f"),
            "warning_probability": st.column_config.NumberColumn("WARNING SCORE", format="%.4f"),
            "prediction": st.column_config.TextColumn("AI ASSESSMENT"),
            "actual_synthetic_label": st.column_config.NumberColumn("SYNTHETIC LABEL", format="%d"),
        },
    )

# ---------------------------------------------------------
# INDIVIDUAL MACHINE
# ---------------------------------------------------------

machine_data = data.loc[data["machine_id"] == selected_machine].copy()
machine_data = machine_data.sort_values("timestamp")

if machine_data.empty:
    st.warning("No observations are available for this asset.")
    st.stop()

latest = machine_data.iloc[-1]

observation = {feature: latest[feature] for feature in predictor.features}

try:
    prediction = predictor.predict(observation)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

st.markdown(
    '<div class="section-label">02 / Asset overview</div>',
    unsafe_allow_html=True,
)

history = load_history()
recorded_events = len(history)

fleet_col, machine_col, score_col, event_col = st.columns(4)

fleet_col.metric("FLEET SIZE", f"{len(machines):02d}")
machine_col.metric("SELECTED ASSET", selected_machine)
score_col.metric(
    "WARNING SCORE",
    f"{prediction['warning_probability']:.2%}",
)
event_col.metric("RECORDED EVENTS", f"{recorded_events:,}")

# ---------------------------------------------------------
# MACHINE CONDITION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-label">03 / Condition assessment</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    status_col, info_col = st.columns([1.4, 1])

    with status_col:
        if prediction["prediction"] == "WARNING":
            st.markdown(
                '<div class="status-warning">● WARNING / REVIEW REQUIRED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-normal">● NORMAL / BELOW THRESHOLD</div>',
                unsafe_allow_html=True,
            )

        st.write(prediction["interpretation"])
        st.caption("An experimental model assessment, not a confirmed equipment diagnosis.")

    with info_col:
        st.markdown("**MODEL INFORMATION**")
        st.write(f"Model: `{prediction['model_name']}`")
        st.write(f"Threshold: `{prediction['threshold']:.2f}`")
        st.write(f"Target: `{prediction['target']}`")
        st.write(f"Dataset type: `{prediction['dataset_type']}`")

st.markdown(
    '<div class="section-label">04 / Latest sensor telemetry</div>',
    unsafe_allow_html=True,
)

temp_col, vibration_col, rpm_col = st.columns(3)

temp_col.metric("TEMPERATURE", f"{latest['temperature_c']:.2f} °C")
vibration_col.metric("VIBRATION", f"{latest['vibration_mm_s']:.2f} mm/s")
rpm_col.metric("ROTATIONAL SPEED", f"{latest['rotational_speed_rpm']:.0f} RPM")

st.caption(
    f"Observation timestamp (UTC): {latest['timestamp']} · "
    f"{len(machine_data):,} observations available"
)

# ---------------------------------------------------------
# SENSOR HEALTH MONITOR
# ---------------------------------------------------------

st.markdown(
    '<div class="section-label">05 / Sensor health monitoring</div>',
    unsafe_allow_html=True,
)

machine_health = sensor_health_rows.loc[sensor_health_rows["machine_id"] == selected_machine].copy()

machine_health_summary = sensor_health_summary.loc[
    sensor_health_summary["machine_id"] == selected_machine
].copy()

with st.container(border=True):
    issue_rows = int((machine_health["sensor_health_issue_count"] > 0).sum())
    total_rows = len(machine_health)

    missing_total = int(machine_health_summary["missing_count"].sum())
    frozen_total = int(machine_health_summary["frozen_count"].sum())
    limits_total = int(machine_health_summary["outside_limits_count"].sum())
    drift_total = int(machine_health_summary["drift_count"].sum())

    health_a, health_b, health_c, health_d = st.columns(4)
    health_a.metric("ROWS WITH ISSUES", f"{issue_rows:,}")
    health_b.metric("MISSING READINGS", f"{missing_total:,}")
    health_c.metric("FROZEN READINGS", f"{frozen_total:,}")
    health_d.metric("DRIFT FLAGS", f"{drift_total:,}")

    if limits_total:
        st.warning(f"{limits_total:,} reading(s) fell outside configured engineering limits.")
    else:
        limits_configured = (
            machine_health_summary["engineering_limits_configured"].astype(bool).all()
        )
        if limits_configured:
            st.success("No readings outside configured engineering limits were detected.")
        else:
            st.info(
                "Engineering limits are not configured for every sensor. "
                "No limit violations can be concluded from this check."
            )

    drift_configured = machine_health_summary["drift_threshold_configured"].astype(bool).all()

    if not drift_configured:
        st.info(
            "Drift thresholds are not configured for every sensor. "
            "Drift monitoring is therefore incomplete."
        )

    if issue_rows:
        st.warning(
            f"{issue_rows:,} observation(s) have one or more sensor-health flags. "
            "Inspect the details below."
        )
    else:
        st.success(
            "No missing-reading or frozen-sensor flags were detected in this asset's "
            "available history. Configured limit and drift checks must be considered "
            "separately."
        )

    st.caption(
        f"Sensor-health observations checked: {total_rows:,}. "
        "These are retrospective checks on synthetic historical data, not live alerts."
    )

    st.markdown("**SENSOR CHANNEL SUMMARY**")

    st.dataframe(
        machine_health_summary,
        use_container_width=True,
        hide_index=True,
    )

    flagged_rows = machine_health.loc[machine_health["sensor_health_issue_count"] > 0].copy()

    with st.expander("View flagged sensor observations"):
        if flagged_rows.empty:
            st.info("No row-level sensor-health flags were produced for this asset.")
        else:
            st.dataframe(
                flagged_rows,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("View all sensor-health observations"):
        st.dataframe(
            machine_health,
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------
# SENSOR ANALYTICS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-label">06 / Sensor analytics</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    sensor_options = {
        "Temperature / °C": "temperature_c",
        "Vibration / mm/s": "vibration_mm_s",
        "Rotational speed / RPM": "rotational_speed_rpm",
    }

    chart_col, stats_col = st.columns([2.2, 1])

    with chart_col:
        selected_sensor_label = st.selectbox(
            "TELEMETRY CHANNEL",
            list(sensor_options.keys()),
        )
        selected_sensor = sensor_options[selected_sensor_label]

        chart_data = machine_data.set_index("timestamp")[[selected_sensor]].rename(
            columns={selected_sensor: selected_sensor_label}
        )

        st.line_chart(chart_data, height=320)

    with stats_col:
        series = machine_data[selected_sensor].dropna()

        st.markdown("**CHANNEL SUMMARY**")

        if not series.empty:
            st.metric("LATEST", f"{series.iloc[-1]:,.2f}")
            st.metric("MEAN", f"{series.mean():,.2f}")
            st.metric("MINIMUM", f"{series.min():,.2f}")
            st.metric("MAXIMUM", f"{series.max():,.2f}")

        st.caption("Descriptive statistics only; operating safety limits have not been configured.")

# ---------------------------------------------------------
# PREDICTION EVENT LOGGING
# ---------------------------------------------------------

st.markdown(
    '<div class="section-label">07 / Prediction event journal</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.write(
        "Save the current assessment to persistent history. "
        "Recording is manual to prevent duplicate events on page reruns."
    )

    if st.button(
        "＋ Record current assessment",
        type="primary",
        use_container_width=True,
    ):
        try:
            event = record_prediction(
                machine_id=str(latest["machine_id"]),
                observation_timestamp=str(latest["timestamp"]),
                prediction=prediction,
                history_path=HISTORY_PATH,
            )

            st.success("Prediction event saved.")
            st.code(event["event_id"], language=None)
            st.caption(f"Recorded at {event['recorded_at_utc']} UTC")
            st.rerun()

        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Unable to record prediction: {exc}")

history = load_history()

if history.empty:
    st.info("No prediction events have been recorded yet.")
else:
    required_history_columns = {
        "event_id",
        "recorded_at_utc",
        "machine_id",
        "observation_timestamp",
        "model_name",
        "warning_probability",
        "threshold",
        "prediction",
        "dataset_type",
    }

    if not required_history_columns.issubset(history.columns):
        st.error(
            "Prediction history has an unexpected schema. "
            "Inspect the history CSV before continuing."
        )
    else:
        selected_history = history.loc[history["machine_id"] == selected_machine].copy()

        journal_a, journal_b, journal_c = st.columns(3)

        journal_a.metric("ALL EVENTS", len(history))
        journal_b.metric("SELECTED ASSET EVENTS", len(selected_history))
        journal_c.metric(
            "RECORDED WARNING EVENTS",
            int((history["prediction"] == "WARNING").sum()),
        )

        if not selected_history.empty:
            selected_history = selected_history.sort_values("recorded_at_utc", ascending=False)

            visible_columns = [
                "recorded_at_utc",
                "observation_timestamp",
                "machine_id",
                "prediction",
                "warning_probability",
                "threshold",
                "model_name",
                "dataset_type",
                "event_id",
            ]

            st.dataframe(
                selected_history[visible_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "recorded_at_utc": st.column_config.TextColumn("RECORDED AT (UTC)"),
                    "observation_timestamp": st.column_config.TextColumn("OBSERVATION TIME"),
                    "machine_id": st.column_config.TextColumn("ASSET"),
                    "prediction": st.column_config.TextColumn("AI DECISION"),
                    "warning_probability": st.column_config.NumberColumn(
                        "WARNING SCORE", format="%.4f"
                    ),
                    "threshold": st.column_config.NumberColumn("THRESHOLD", format="%.2f"),
                    "event_id": st.column_config.TextColumn("EVENT ID", width="medium"),
                },
            )
        else:
            st.info(f"No prediction events recorded for {selected_machine}.")

        with st.expander("View fleet-wide event history"):
            st.dataframe(
                history.sort_values("recorded_at_utc", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

# ---------------------------------------------------------
# TECHNICAL DETAILS
# ---------------------------------------------------------

with st.expander("Technical details & data provenance"):
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("**MODEL**")
        st.write(f"Model: {prediction['model_name']}")
        st.write(f"Required features: {len(predictor.features)}")
        st.write(f"Target: {prediction['target']}")
        st.write(f"Threshold: {prediction['threshold']}")
        st.write(f"Dataset type: {prediction['dataset_type']}")

    with detail_col2:
        st.markdown("**DATA & STORAGE**")
        st.write(f"Dataset rows: {len(data):,}")
        st.write(f"Machine assets: {len(machines)}")
        st.write(f"Dataset path: `{DATA_PATH}`")
        st.write(f"History path: `{HISTORY_PATH}`")
        st.write("Sensor-health checks: enabled")
        st.write("Sensor-health settings: see channel summary above")

    st.warning(
        "This prototype uses synthetic historical observations. "
        "It does not ingest live sensors or provide safety-certified "
        "machine diagnostics."
    )

st.markdown(
    """
    <div class="footer-note">
        FORGE INDUSTRIAL AI / PREDICTIVE MAINTENANCE
        <br><br>
        EXPERIMENTAL SYSTEM · SYNTHETIC DATA · NOT FOR SAFETY-CRITICAL USE
    </div>
    """,
    unsafe_allow_html=True,
)
