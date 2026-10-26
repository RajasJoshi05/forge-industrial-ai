import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# MACHINE-SPECIFIC CONFIGURATION STORAGE
# ---------------------------------------------------------

CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "machine_limits.json"
)

ASSET_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "config" / "asset_registry.json"
)


def load_machine_limits() -> dict:
    """Load saved engineering limits for individual machines."""
    if not CONFIG_PATH.exists():
        return {}

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        return saved if isinstance(saved, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_machine_limits(settings: dict) -> None:
    """Persist machine-specific engineering limits to JSON."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = CONFIG_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    temporary_path.replace(CONFIG_PATH)



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
    SENSORS,
    monitor_sensor_health,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_features.csv"
HISTORY_PATH = PROJECT_ROOT / "artifacts" / "reports" / "prediction_history.csv"
ANOMALY_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "industrial_anomaly_predictions.csv"
)


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


ASSET_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "config" / "asset_registry.json"
)


TELEMETRY_STORAGE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "uploaded_telemetry.csv"
)


def save_uploaded_telemetry(
    telemetry: pd.DataFrame,
    machine_id: str,
) -> int:
    """Append validated telemetry for one registered machine to disk."""
    required_columns = {
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
    }

    missing = required_columns - set(telemetry.columns)
    if missing:
        raise ValueError(
            f"Telemetry is missing columns: {sorted(missing)}"
        )

    if telemetry.empty:
        raise ValueError("Cannot save an empty telemetry upload.")

    records = telemetry.copy()
    records["machine_id"] = machine_id

    records["timestamp"] = pd.to_datetime(
        records["timestamp"],
        utc=True,
        errors="raise",
    )

    TELEMETRY_STORAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TELEMETRY_STORAGE_PATH.exists():
        existing = pd.read_csv(TELEMETRY_STORAGE_PATH)
        combined = pd.concat(
            [existing, records],
            ignore_index=True,
        )
    else:
        combined = records

    combined.to_csv(TELEMETRY_STORAGE_PATH, index=False)

    return len(records)






def load_asset_registry() -> dict:
    """Load registered industrial assets."""
    if not ASSET_REGISTRY_PATH.is_file():
        return {}

    try:
        with ASSET_REGISTRY_PATH.open("r", encoding="utf-8") as file:
            registry = json.load(file)

        return registry if isinstance(registry, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_asset_registry(registry: dict) -> None:
    """Save registered industrial assets."""
    ASSET_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ASSET_REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, ensure_ascii=False)








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


@st.cache_data
def load_uploaded_telemetry() -> pd.DataFrame:
    """Load saved telemetry uploaded for registered industrial assets."""
    required_columns = {
        "machine_id",
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
    }

    if not TELEMETRY_STORAGE_PATH.is_file():
        return pd.DataFrame(columns=sorted(required_columns))

    telemetry = pd.read_csv(TELEMETRY_STORAGE_PATH)

    missing = required_columns - set(telemetry.columns)
    if missing:
        raise ValueError(
            f"Uploaded telemetry is missing columns: {sorted(missing)}"
        )

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in (
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
    ):
        telemetry[column] = pd.to_numeric(
            telemetry[column],
            errors="coerce",
        )

    telemetry = telemetry.dropna(
        subset=[
            "machine_id",
            "timestamp",
            "temperature_c",
            "vibration_mm_s",
            "rotational_speed_rpm",
        ]
    )

    return telemetry






@st.cache_data
def load_anomaly_predictions(path: str) -> pd.DataFrame:
    """Load and validate saved anomaly-detection predictions."""
    predictions = pd.read_csv(path)

    required_columns = {
        "machine_id",
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "rotational_speed_rpm",
        "anomaly_score",
        "is_anomaly",
        "partition",
    }

    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Anomaly predictions are missing columns: {sorted(missing)}")

    if predictions.empty:
        raise ValueError("Anomaly predictions file contains no rows.")

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"],
        utc=True,
        errors="raise",
    )

    anomaly_values = (
        predictions["is_anomaly"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if anomaly_values.isna().any():
        raise ValueError("Anomaly predictions contain invalid is_anomaly values.")

    predictions["is_anomaly"] = anomaly_values
    predictions["anomaly_score"] = pd.to_numeric(
        predictions["anomaly_score"],
        errors="raise",
    )

    return predictions


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
# ASSET REGISTRY
# ---------------------------------------------------------

asset_registry = load_asset_registry()

with st.expander("ASSET REGISTRY", expanded=False):
    st.caption("Registered assets and their telemetry readiness.")

    if not asset_registry:
        st.info("No custom assets have been registered yet.")
    else:
        registry_rows = []

        for asset in asset_registry.values():
            registry_rows.append(
                {
                    "Machine ID": asset.get("machine_id", "Unknown"),
                    "Name": asset.get("name", "Unnamed asset"),
                    "Machine Type": asset.get("machine_type", "Not specified"),
                    "Location": asset.get("location") or "Not specified",
                    "Status": asset.get(
                        "status",
                        "Registered — awaiting telemetry",
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(registry_rows),
            use_container_width=True,
            hide_index=True,
        )






# ---------------------------------------------------------
# TELEMETRY CSV IMPORT
# ---------------------------------------------------------

with st.expander("TELEMETRY IMPORT", expanded=False):
    st.caption(
        "Upload sensor readings for a registered machine. "
        "Validated readings are saved separately from the "
        "synthetic historical dataset."
    )

    asset_registry = load_asset_registry()

    if not asset_registry:
        st.info("Register a machine before importing telemetry.")

    else:
        asset_options = {
            f"{asset.get('machine_id', asset_id)} — "
            f"{asset.get('name', 'Unnamed asset')}": asset_id
            for asset_id, asset in asset_registry.items()
        }

        selected_asset_label = st.selectbox(
            "Registered asset",
            list(asset_options.keys()),
            key="telemetry_import_asset",
        )

        uploaded_telemetry = st.file_uploader(
            "Upload telemetry CSV",
            type=["csv"],
            key="telemetry_csv_upload",
            help=(
                "Required columns: timestamp, temperature_c, "
                "vibration_mm_s, rotational_speed_rpm."
            ),
        )

        if uploaded_telemetry is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_telemetry)

                required_columns = {
                    "timestamp",
                    "temperature_c",
                    "vibration_mm_s",
                    "rotational_speed_rpm",
                }

                missing_columns = required_columns - set(
                    uploaded_data.columns
                )

                if missing_columns:
                    st.error(
                        "CSV is missing required columns: "
                        f"{sorted(missing_columns)}"
                    )

                elif uploaded_data.empty:
                    st.error("The uploaded CSV contains no data.")

                else:
                    uploaded_data["timestamp"] = pd.to_datetime(
                        uploaded_data["timestamp"],
                        utc=True,
                        errors="coerce",
                    )

                    numeric_columns = [
                        "temperature_c",
                        "vibration_mm_s",
                        "rotational_speed_rpm",
                    ]

                    for column in numeric_columns:
                        uploaded_data[column] = pd.to_numeric(
                            uploaded_data[column],
                            errors="coerce",
                        )

                    invalid_rows = uploaded_data[
                        uploaded_data["timestamp"].isna()
                        | uploaded_data[numeric_columns]
                        .isna()
                        .any(axis=1)
                    ]

                    if not invalid_rows.empty:
                        st.error(
                            f"{len(invalid_rows)} row(s) contain invalid "
                            "timestamps or sensor readings. Correct the "
                            "CSV and upload it again."
                        )

                    else:
                        selected_machine_id = asset_options[
                            selected_asset_label
                        ]

                        try:
                            saved_count = save_uploaded_telemetry(
                                uploaded_data,
                                selected_machine_id,
                            )

                            st.success(
                                f"Validated and saved {saved_count:,} "
                                f"readings for {selected_machine_id}."
                            )

                            st.dataframe(
                                uploaded_data.head(20),
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.caption(
                                "Telemetry is stored separately from "
                                "the synthetic dataset. Predictions "
                                "have not been generated."
                            )

                        except (OSError, ValueError, pd.errors.ParserError) as exc:
                            st.error(
                                f"Could not save telemetry: {exc}"
                            )

            except (
                ValueError,
                pd.errors.ParserError,
                UnicodeDecodeError,
            ) as exc:
                st.error(f"Could not read the uploaded CSV: {exc}")


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

# Load the original historical dataset and predictive-maintenance model.
try:
    data = load_data(str(DATA_PATH))
    predictor = load_predictor()
except Exception as exc:
    st.error(f"System initialization failed: {exc}")
    st.stop()

# Load the registered industrial assets.
asset_registry = load_asset_registry()

registered_machine_ids = [
    str(asset.get("machine_id", asset_id))
    for asset_id, asset in asset_registry.items()
]

# Load telemetry uploaded by users.
try:
    uploaded_telemetry = load_uploaded_telemetry()
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Uploaded telemetry could not be loaded: {exc}")
    st.stop()

uploaded_machine_ids = (
    uploaded_telemetry["machine_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

# Build a unified machine list without modifying the training dataset.
historical_machine_ids = (
    data["machine_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

machines = sorted(
    set(historical_machine_ids)
    | set(registered_machine_ids)
    | set(uploaded_machine_ids)
)

if not machines:
    st.error(
        "No machines were found in the historical dataset, "
        "asset registry, or uploaded telemetry."
    )
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




with st.sidebar.expander("＋ Register a machine", expanded=False):
    st.caption("Add an asset to the FORGE registry.")

    with st.form("register_machine_form", clear_on_submit=True):
        new_machine_id = st.text_input(
            "Machine ID",
            placeholder="e.g. PRESS-001",
        ).strip().upper()

        new_machine_name = st.text_input(
            "Display name",
            placeholder="e.g. Hydraulic Press 01",
        ).strip()

        new_machine_type = st.selectbox(
            "Machine type",
            [
                "General industrial equipment",
                "Motor",
                "Pump",
                "Compressor",
                "Conveyor",
                "Hydraulic press",
                "CNC machine",
                "Robotic arm",
                "Other",
            ],
        )

        new_machine_location = st.text_input(
            "Location / production line",
            placeholder="e.g. Line A",
        ).strip()

        register_submitted = st.form_submit_button(
            "Register asset",
            use_container_width=True,
        )

    if register_submitted:
        if not new_machine_id:
            st.error("Enter a machine ID.")
        elif not new_machine_name:
            st.error("Enter a display name.")
        elif new_machine_id in machines:
            st.error(
                "This machine ID already exists in the telemetry dataset."
            )
        else:
            registry = load_asset_registry()

            if new_machine_id in registry:
                st.error("This machine ID is already registered.")
            else:
                registry[new_machine_id] = {
                    "machine_id": new_machine_id,
                    "name": new_machine_name,
                    "machine_type": new_machine_type,
                    "location": new_machine_location,
                    "status": "Registered — awaiting telemetry",
                }

                try:
                    save_asset_registry(registry)
                    st.success(f"Registered {new_machine_id}.")
                    st.rerun()
                except OSError as exc:
                    st.error(
                        f"Could not save the asset registry: {exc}"
                    )








st.sidebar.divider()
st.sidebar.markdown("**SENSOR CONFIGURATION**")
st.sidebar.caption(
    "Configure review thresholds for the selected asset. "
    "Use verified machine specifications."
)

sensor_labels = {
    "temperature_c": "Temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "rotational_speed_rpm": "Rotational speed (RPM)",
}

all_machine_limits = load_machine_limits()
machine_settings = all_machine_limits.get(selected_machine, {})

with st.sidebar.expander("Engineering limits", expanded=False):
    configured_limits = {}

    for sensor in SENSORS:
        st.markdown(f"**{sensor_labels[sensor]}**")

        saved_bounds = machine_settings.get(sensor)
        enabled_key = f"{selected_machine}_{sensor}_limits_enabled"
        lower_key = f"{selected_machine}_{sensor}_lower"
        upper_key = f"{selected_machine}_{sensor}_upper"

        if enabled_key not in st.session_state:
            st.session_state[enabled_key] = saved_bounds is not None

        if saved_bounds is not None:
            st.session_state.setdefault(lower_key, float(saved_bounds[0]))
            st.session_state.setdefault(upper_key, float(saved_bounds[1]))

        enabled = st.checkbox(
            f"Enable limits for {sensor_labels[sensor]}",
            key=enabled_key,
        )

        if enabled:
            lower = st.number_input(
                f"Lower limit — {sensor_labels[sensor]}",
                key=lower_key,
                value=0.0,
            )
            upper = st.number_input(
                f"Upper limit — {sensor_labels[sensor]}",
                key=upper_key,
                value=100.0,
            )

            if lower >= upper:
                st.error("The upper limit must be greater than the lower limit.")
                configured_limits[sensor] = None
            else:
                configured_limits[sensor] = (lower, upper)
        else:
            configured_limits[sensor] = None

    if st.button("Save machine limits", use_container_width=True):
        if any(
            st.session_state.get(f"{selected_machine}_{sensor}_limits_enabled")
            and configured_limits.get(sensor) is None
            for sensor in SENSORS
        ):
            st.error("Fix invalid limits before saving.")
        else:
            all_machine_limits[selected_machine] = {
                sensor: (
                    list(configured_limits[sensor])
                    if configured_limits[sensor] is not None
                    else None
                )
                for sensor in SENSORS
            }

            try:
                save_machine_limits(all_machine_limits)
                st.success(f"Limits saved for {selected_machine}.")
            except OSError as exc:
                st.error(f"Could not save machine limits: {exc}")


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
# SENSOR HEALTH COMPUTATION
# ---------------------------------------------------------

all_machine_limits = load_machine_limits()

try:
    health_results = []
    health_summaries = []

    for machine_id in machines:
        machine_data_for_health = data.loc[
            data["machine_id"] == machine_id
        ].copy()

        # Skip registered machines that do not have telemetry yet.
        if machine_data_for_health.empty:
            continue

        saved_limits = all_machine_limits.get(machine_id, {})

        # Use the current sidebar configuration for the selected machine.
        if machine_id == selected_machine:
            saved_limits = {
                sensor: (
                    list(configured_limits[sensor])
                    if configured_limits.get(sensor) is not None
                    else None
                )
                for sensor in SENSORS
            }

        machine_engineering_limits = {
            sensor: (
                tuple(saved_limits[sensor])
                if saved_limits.get(sensor) is not None
                else None
            )
            for sensor in SENSORS
        }

        checked_rows, checked_summary = monitor_sensor_health(
            machine_data_for_health,
            engineering_limits=machine_engineering_limits,
        )

        if not checked_rows.empty:
            health_results.append(checked_rows)

        if not checked_summary.empty:
            health_summaries.append(checked_summary)

    if health_results and health_summaries:
        sensor_health_rows = pd.concat(
            health_results,
            ignore_index=True,
        )

        sensor_health_summary = pd.concat(
            health_summaries,
            ignore_index=True,
        )
    else:
        sensor_health_rows = pd.DataFrame()
        sensor_health_summary = pd.DataFrame()

except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Sensor-health monitoring failed: {exc}")
    st.stop()





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


# SENSOR ANALYTICS
st.markdown(
    '<div class="section-label">04 / Sensor analytics</div>',
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

        chart_data = machine_data[
            ["timestamp", selected_sensor]
        ].copy()

        chart_data["timestamp"] = pd.to_datetime(
            chart_data["timestamp"],
            utc=True,
            errors="coerce",
        )
        chart_data[selected_sensor] = pd.to_numeric(
            chart_data[selected_sensor],
            errors="coerce",
        )

        chart_data = (
            chart_data
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_data["timestamp"],
                y=chart_data[selected_sensor],
                mode="lines",
                name=selected_sensor_label,
                line={"color": "#36D6C3", "width": 2},
                connectgaps=False,
            )
        )

        active_limits = configured_limits.get(selected_sensor)

        if active_limits is not None:
            lower_limit, upper_limit = active_limits

            fig.add_hline(
                y=lower_limit,
                line_dash="dash",
                line_color="#FFBE69",
                annotation_text=f"Lower limit: {lower_limit:g}",
            )

            fig.add_hline(
                y=upper_limit,
                line_dash="dash",
                line_color="#FF727C",
                annotation_text=f"Upper limit: {upper_limit:g}",
            )

            abnormal = chart_data.loc[
                chart_data[selected_sensor].notna()
                & (
                    (chart_data[selected_sensor] < lower_limit)
                    | (chart_data[selected_sensor] > upper_limit)
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=abnormal["timestamp"],
                    y=abnormal[selected_sensor],
                    mode="markers",
                    name="Outside configured limits",
                    marker={
                        "color": "#FF454F",
                        "size": 7,
                        "symbol": "circle",
                    },
                )
            )

        fig.update_layout(
            height=380,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={
                "family": "IBM Plex Sans, sans-serif",
                "color": "#EAF1F7",
            },
            margin={"l": 20, "r": 20, "t": 45, "b": 35},
            legend={"orientation": "h", "y": 1.12},
            xaxis_title="Timestamp",
            yaxis_title=selected_sensor_label,
            hovermode="x unified",
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with stats_col:
        series = pd.to_numeric(
            machine_data[selected_sensor],
            errors="coerce",
        ).dropna()

        st.markdown("**CHANNEL SUMMARY**")

        if not series.empty:
            st.metric(
                "Latest reading",
                f"{series.iloc[-1]:,.2f}",
            )

            st.metric(
                "Average",
                f"{series.mean():,.2f}",
            )

            st.metric(
                "Minimum",
                f"{series.min():,.2f}",
            )

            st.metric(
                "Maximum",
                f"{series.max():,.2f}",
            )

            st.metric(
                "Valid readings",
                f"{len(series):,}",
            )
        else:
            st.warning("No valid readings are available for this sensor.")

        if active_limits is not None and not series.empty:
            lower_limit, upper_limit = active_limits

            outside_count = int(
                ((series < lower_limit) | (series > upper_limit)).sum()
            )

            st.metric(
                "Outside configured limits",
                f"{outside_count:,}",
            )

            if outside_count:
                st.warning(
                    "Some readings exceed the configured limits. "
                    "Verify these thresholds against machine-specific "
                    "engineering specifications."
                )
            else:
                st.success("All valid readings are within configured limits.")

        else:
            st.caption(
                "Configure lower and upper limits in the sidebar to "
                "highlight readings outside your selected range."
            )


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
# ANOMALY DETECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-label">08 / Anomaly detection</div>',
    unsafe_allow_html=True,
)

if not ANOMALY_PREDICTIONS_PATH.is_file():
    st.warning(
        "Saved anomaly predictions were not found. "
        "Run the anomaly-training pipeline to generate them."
    )
else:
    try:
        anomaly_data = load_anomaly_predictions(str(ANOMALY_PREDICTIONS_PATH))
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Anomaly monitoring could not be loaded: {exc}")
    else:
        total_observations = len(anomaly_data)
        total_anomalies = int(anomaly_data["is_anomaly"].sum())
        overall_rate = total_anomalies / total_observations if total_observations else 0.0

        holdout_data = anomaly_data.loc[anomaly_data["partition"] == "holdout"]
        holdout_anomalies = int(holdout_data["is_anomaly"].sum())

        with st.container(border=True):
            anomaly_a, anomaly_b, anomaly_c, anomaly_d = st.columns(4)

            anomaly_a.metric(
                "OBSERVATIONS SCORED",
                f"{total_observations:,}",
            )
            anomaly_b.metric(
                "ANOMALIES FLAGGED",
                f"{total_anomalies:,}",
            )
            anomaly_c.metric(
                "OVERALL ANOMALY RATE",
                f"{overall_rate:.2%}",
            )
            anomaly_d.metric(
                "HOLDOUT ANOMALIES",
                f"{holdout_anomalies:,}",
            )

            st.caption(
                "Results are from the saved Isolation Forest predictions. "
                "An anomaly is an unusual sensor pattern, not a confirmed failure."
            )

        machine_anomalies = anomaly_data.loc[anomaly_data["machine_id"] == selected_machine].copy()

        machine_anomalies = machine_anomalies.sort_values("timestamp")

        st.markdown("**SELECTED ASSET / ANOMALY REVIEW**")

        if machine_anomalies.empty:
            st.info("No anomaly observations are available for this asset.")
        else:
            flagged_machine_rows = machine_anomalies.loc[machine_anomalies["is_anomaly"]].copy()

            machine_a, machine_b, machine_c = st.columns(3)
            machine_a.metric(
                "ASSET OBSERVATIONS",
                f"{len(machine_anomalies):,}",
            )
            machine_b.metric(
                "ASSET ANOMALIES",
                f"{len(flagged_machine_rows):,}",
            )
            machine_c.metric(
                "ASSET ANOMALY RATE",
                f"{machine_anomalies['is_anomaly'].mean():.2%}",
            )

            sensor_options = {
                "Temperature / °C": "temperature_c",
                "Vibration / mm/s": "vibration_mm_s",
                "Rotational speed / RPM": "rotational_speed_rpm",
            }

            selected_anomaly_sensor_label = st.selectbox(
                "ANOMALY SENSOR CHANNEL",
                list(sensor_options.keys()),
                key="anomaly_sensor_channel",
            )
            selected_anomaly_sensor = sensor_options[selected_anomaly_sensor_label]

            chart_data = machine_anomalies.tail(500).set_index("timestamp")
            st.line_chart(
                chart_data[[selected_anomaly_sensor]],
                height=300,
            )
            st.caption(
                "Trend shows the latest 500 observations for the selected asset. "
                "Use the flagged-observations table below to identify anomalies."
            )

            with st.expander("View flagged observations", expanded=True):
                if flagged_machine_rows.empty:
                    st.success("No observations were flagged for this asset.")
                else:
                    visible_columns = [
                        "timestamp",
                        "machine_id",
                        "temperature_c",
                        "vibration_mm_s",
                        "rotational_speed_rpm",
                        "anomaly_score",
                        "partition",
                    ]

                    st.dataframe(
                        flagged_machine_rows[visible_columns].sort_values(
                            "timestamp",
                            ascending=False,
                        ),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn(
                                "OBSERVATION TIME (UTC)",
                                format="YYYY-MM-DD HH:mm:ss",
                            ),
                            "machine_id": st.column_config.TextColumn("ASSET"),
                            "temperature_c": st.column_config.NumberColumn(
                                "TEMPERATURE °C",
                                format="%.2f",
                            ),
                            "vibration_mm_s": st.column_config.NumberColumn(
                                "VIBRATION mm/s",
                                format="%.3f",
                            ),
                            "rotational_speed_rpm": st.column_config.NumberColumn(
                                "SPEED RPM",
                                format="%.2f",
                            ),
                            "anomaly_score": st.column_config.NumberColumn(
                                "ANOMALY SCORE",
                                format="%.5f",
                            ),
                            "partition": st.column_config.TextColumn("DATA PARTITION"),
                        },
                    )

            with st.expander("View all observations for this asset"):
                st.dataframe(
                    machine_anomalies.sort_values(
                        "timestamp",
                        ascending=False,
                    ),
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
