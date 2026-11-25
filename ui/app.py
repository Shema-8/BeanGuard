
import os
import json
import time
from datetime import datetime

import requests
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="BeanGuard Dashboard",
    page_icon="🌱",
    layout="wide",
)

# Render API URL
API = os.getenv(
    "API_URL",
    "https://beanguard.onrender.com"
).rstrip("/")

REQUEST_TIMEOUT = 60


# --------------------------------------------------
# API HELPER FUNCTIONS
# --------------------------------------------------

def api(method, endpoint, **kwargs):
    """Send a request to the BeanGuard API."""
    url = f"{API}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        return response

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to BeanGuard API: {error}")
        return None


def get_error_message(response, default="No data available."):
    """Safely extract an error message from an API response."""
    try:
        data = response.json()

        if isinstance(data, dict):
            return data.get("detail", default)

        return default

    except (ValueError, AttributeError):
        if response is not None:
            return response.text or (
                f"Server returned HTTP {response.status_code}"
            )

        return default


def get_json(endpoint, default=None):
    """GET an endpoint and return its JSON response."""
    if default is None:
        default = {}

    response = api("GET", endpoint)

    if response is None:
        return default

    if response.ok:
        try:
            return response.json()
        except ValueError:
            return default

    return default


def display_api_error(response):
    """Display an API error safely."""
    if response is not None:
        st.error(get_error_message(response))


def show_metric(label, value):
    """Display a metric safely."""
    st.metric(label=label, value=value)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌱 BeanGuard")
st.subheader("Bean Leaf Disease Detection and Monitoring")

st.write(
    "Upload a bean leaf image to detect whether it is healthy "
    "or affected by angular leaf spot or bean rust."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌱 BeanGuard")
st.sidebar.caption("MLOps Monitoring Dashboard")

st.sidebar.write("**API connection**")

health_response = api("GET", "/health")

if health_response is not None and health_response.ok:
    st.sidebar.success("API is online")
else:
    st.sidebar.error("API is unavailable")

if st.sidebar.button("Refresh API status"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("BeanGuard | Bean Leaf Disease Classifier")


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

diagnose_tab, insights_tab, upload_tab, monitoring_tab = st.tabs(
    [
        "🔍 Diagnose",
        "📊 Data insights",
        "📤 Upload & Retrain",
        "📈 Monitoring",
    ]
)


# ==================================================
# TAB 1: DIAGNOSE
# ==================================================

with diagnose_tab:

    st.header("Bean Leaf Disease Diagnosis")

    st.write(
        "Upload a bean leaf image. BeanGuard will analyze it "
        "using the trained MobileNetV2 model."
    )

    uploaded_image = st.file_uploader(
        "Choose a bean leaf image",
        type=["jpg", "jpeg", "png"],
        key="diagnosis_image",
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded bean leaf image",
            use_column_width=True,
        )

        if st.button("Analyze image", type="primary"):

            with st.spinner("Analyzing image..."):

                try:
                    uploaded_image.seek(0)

                    files = {
                        "file": (
                            uploaded_image.name,
                            uploaded_image.getvalue(),
                            uploaded_image.type,
                        )
                    }

                    response = api(
                        "POST",
                        "/predict",
                        files=files,
                    )

                    if response is not None and response.ok:

                        result = response.json()

                        st.success("Analysis completed successfully!")

                        prediction = result.get(
                            "prediction",
                            result.get("class", "Unknown"),
                        )

                        confidence = result.get(
                            "confidence",
                            None,
                        )

                        st.subheader("Prediction")

                        st.write(f"**Detected class:** {prediction}")

                        if confidence is not None:

                            try:
                                confidence_value = float(confidence)

                                if confidence_value > 1:
                                    confidence_value /= 100

                                st.metric(
                                    "Confidence",
                                    f"{confidence_value * 100:.2f}%",
                                )

                                st.progress(
                                    max(
                                        0.0,
                                        min(1.0, confidence_value),
                                    )
                                )

                            except (ValueError, TypeError):
                                st.write(
                                    f"**Confidence:** {confidence}"
                                )

                        st.subheader("Full API response")

                        st.json(result)

                    else:
                        display_api_error(response)

                except Exception as error:
                    st.error(f"Prediction failed: {error}")


# ==================================================
# TAB 2: DATA INSIGHTS
# ==================================================

with insights_tab:

    st.header("Dataset Insights")

    st.write(
        "Explore information about the bean leaf dataset "
        "used by the BeanGuard model."
    )

    if st.button("Load data insights"):

        with st.spinner("Loading dataset information..."):

            insights = get_json("/insights", default=None)

            if insights:

                st.success("Dataset information loaded.")

                st.json(insights)

                # Display dictionary values as metrics
                if isinstance(insights, dict):

                    metric_items = list(insights.items())

                    metric_items = [
                        item
                        for item in metric_items
                        if isinstance(
                            item[1],
                            (int, float, str),
                        )
                    ]

                    if metric_items:

                        columns = st.columns(
                            min(4, len(metric_items))
                        )

                        for index, (key, value) in enumerate(
                            metric_items[:4]
                        ):

                            with columns[index]:
                                show_metric(
                                    key.replace("_", " ").title(),
                                    value,
                                )

            else:
                st.info(
                    "No dataset insights are currently available."
                )


# ==================================================
# TAB 3: UPLOAD AND RETRAIN
# ==================================================

with upload_tab:

    st.header("Upload New Bean Leaf Images")

    st.write(
        "Upload correctly labelled bean leaf images to add "
        "new data for the retraining process."
    )

    st.warning(
        "Only use real, correctly labelled bean leaf images "
        "for model training. Do not use unverified or AI-generated "
        "images as training data."
    )

    selected_label = st.selectbox(
        "Select the correct disease class",
        [
            "angular_leaf_spot",
            "bean_rust",
            "healthy",
        ],
    )

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="training_images",
    )

    if uploaded_files:

        st.write(
            f"**Selected images:** {len(uploaded_files)}"
        )

        for file in uploaded_files:

            st.image(
                file,
                caption=file.name,
                use_column_width=True,
            )

        if st.button("Upload labelled images", type="primary"):

            with st.spinner("Uploading images..."):

                try:
                    files = []

                    for file in uploaded_files:

                        file.seek(0)

                        files.append(
                            (
                                "files",
                                (
                                    file.name,
                                    file.getvalue(),
                                    file.type,
                                ),
                            )
                        )

                    response = api(
                        "POST",
                        "/upload",
                        files=files,
                        data={
                            "label": selected_label,
                        },
                    )

                    if response is not None and response.ok:

                        st.success(
                            "Images uploaded successfully!"
                        )

                        try:
                            st.json(response.json())
                        except ValueError:
                            st.write(response.text)

                    else:
                        display_api_error(response)

                except Exception as error:
                    st.error(f"Upload failed: {error}")

    st.divider()

    st.subheader("Model Retraining")

    st.write(
        "After uploading new labelled images, you can request "
        "the API to start the retraining process."
    )

    if st.button("Start model retraining"):

        with st.spinner("Sending retraining request..."):

            response = api("POST", "/retrain")

            if response is not None and response.ok:

                st.success(
                    "Retraining request submitted successfully."
                )

                try:
                    st.json(response.json())
                except ValueError:
                    st.write(response.text)

            else:
                display_api_error(response)

    st.divider()

    st.subheader("Retraining Status")

    if st.button("Check retraining status"):

        status = get_json(
            "/retrain/status",
            default=None,
        )

        if status is not None:

            st.json(status)

        else:
            st.info(
                "Retraining status is not currently available."
            )


# ==================================================
# TAB 4: MONITORING
# ==================================================

with monitoring_tab:

    st.header("Model Monitoring")

    st.write(
        "Monitor the API, model performance information, "
        "and recent prediction activity."
    )

    # ----------------------------------------------
    # API STATUS
    # ----------------------------------------------

    st.subheader("API Status")

    status = get_json("/status", default=None)

    if status:

        st.success("Status information retrieved.")

        st.json(status)

    else:

        st.warning(
            "API status information is unavailable."
        )

    st.divider()

    # ----------------------------------------------
    # MODEL METRICS
    # ----------------------------------------------

    st.subheader("Model Metrics")

    metrics = get_json("/metrics", default=None)

    if metrics:

        st.json(metrics)

        if isinstance(metrics, dict):

            numeric_metrics = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }

            if numeric_metrics:

                metric_columns = st.columns(
                    min(4, len(numeric_metrics))
                )

                for index, (key, value) in enumerate(
                    list(numeric_metrics.items())[:4]
                ):

                    with metric_columns[index]:

                        show_metric(
                            key.replace("_", " ").title(),
                            value,
                        )

    else:

        st.info(
            "Model metrics are not currently available."
        )

    st.divider()

    # ----------------------------------------------
    # RECENT PREDICTIONS
    # ----------------------------------------------

    st.subheader("Recent Predictions")

    prediction_limit = st.number_input(
        "Number of recent predictions",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    if st.button("Load recent predictions"):

        predictions = get_json(
            f"/predictions/recent?limit={int(prediction_limit)}",
            default=None,
        )

        if predictions:

            st.success("Recent predictions loaded.")

            if isinstance(predictions, list):

                dataframe = pd.DataFrame(predictions)

                if not dataframe.empty:
                    st.dataframe(
                        dataframe,
                        use_container_width=True,
                    )
                else:
                    st.info("No prediction records found.")

            elif isinstance(predictions, dict):

                records = (
                    predictions.get("predictions")
                    or predictions.get("data")
                    or predictions.get("results")
                )

                if isinstance(records, list):

                    dataframe = pd.DataFrame(records)

                    if not dataframe.empty:
                        st.dataframe(
                            dataframe,
                            use_container_width=True,
                        )
                    else:
                        st.info("No prediction records found.")

                else:
                    st.json(predictions)

            else:
                st.write(predictions)

        else:

            st.info(
                "No recent prediction records are available."
            )

    st.divider()

    # ----------------------------------------------
    # REFRESH
    # ----------------------------------------------

    if st.button("Refresh monitoring information"):

        st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "BeanGuard | MobileNetV2 Bean Leaf Disease Classification "
    "| Angular Leaf Spot, Bean Rust, and Healthy"
)
