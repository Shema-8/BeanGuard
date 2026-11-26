
import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BeanGuard | Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = os.getenv(
    "API_URL",
    "https://beanguard.onrender.com"
).rstrip("/")

TIMEOUT = 90


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background: #f5f7f4;
    color: #17251b;
    font-family: 'Segoe UI', sans-serif;
}

/* Main content spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063c27 0%, #0b5634 100%);
}

section[data-testid="stSidebar"] * {
    color: #ffffff;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25);
}

/* Headings */
h1, h2, h3 {
    color: #173d27;
    font-weight: 700;
}

p {
    color: #526457;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #16833d, #36a852);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 45px;
    font-weight: 650;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #0c642e;
    color: white;
    border: none;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #f8fbf7;
    border: 2px dashed #a9cdb0;
    border-radius: 15px;
    padding: 15px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1ebe2;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(20,60,30,0.04);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: #eaf1e9;
    border-radius: 10px;
    padding: 12px 20px;
    color: #244b31;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #218b42 !important;
    color: white !important;
}

/* Cards */
.bg-card {
    background: white;
    border: 1px solid #e2ebe2;
    border-radius: 17px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(20,60,30,0.04);
}

.hero {
    background: linear-gradient(100deg, #073c25, #267a39);
    color: white;
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 43px;
    margin-bottom: 5px;
}

.hero p {
    color: #e1f1e2;
    font-size: 17px;
}

.small-label {
    color: #6b806f;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-online {
    color: #9af2a6;
    font-weight: 700;
}

.result-healthy {
    background: #e7f6e8;
    border-left: 5px solid #269a46;
    padding: 18px;
    border-radius: 12px;
}

.result-disease {
    background: #fff0e9;
    border-left: 5px solid #e47b38;
    padding: 18px;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #829184;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# API FUNCTIONS
# =========================================================

def api_request(method, endpoint, **kwargs):
    try:
        return requests.request(
            method,
            f"{API}{endpoint}",
            timeout=TIMEOUT,
            **kwargs
        )
    except requests.exceptions.RequestException as error:
        st.error(f"API connection failed: {error}")
        return None


def api_json(endpoint):
    response = api_request("GET", endpoint)

    if response is not None and response.ok:
        try:
            return response.json()
        except ValueError:
            return None

    return None


def show_api_error(response):
    if response is None:
        st.error("Unable to connect to the BeanGuard API.")
        return

    try:
        message = response.json().get(
            "detail",
            response.text
        )
    except (ValueError, AttributeError):
        message = response.text

    st.error(f"Request failed ({response.status_code}): {message}")


def display_result(result):
    prediction = result.get(
        "prediction",
        result.get("class", "Unknown")
    )

    confidence = result.get("confidence")

    prediction_text = str(prediction).replace("_", " ").title()

    if prediction == "healthy":
        st.markdown(
            f"""
            <div class="result-healthy">
                <h2>🌿 {prediction_text}</h2>
                <p>The model classified this bean leaf as healthy.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="result-disease">
                <h2>⚠️ {prediction_text}</h2>
                <p>The model detected a possible bean leaf disease.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if confidence is not None:
        try:
            confidence_value = float(confidence)

            if confidence_value > 1:
                confidence_value /= 100

            confidence_value = max(
                0.0,
                min(1.0, confidence_value)
            )

            st.metric(
                "Model confidence",
                f"{confidence_value * 100:.2f}%"
            )

            st.progress(confidence_value)

        except (ValueError, TypeError):
            st.write(f"Confidence: {confidence}")

    st.caption(
        "This is an AI prediction. Confirm suspected diseases "
        "with appropriate agricultural expertise."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:20px 5px;">
            <div style="font-size:55px;">🌱</div>
            <h1 style="color:white; font-size:29px;">BeanGuard</h1>
            <p style="color:#d2e8d8;">
                Smart Bean Leaf<br>Disease Detector
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🌿 Navigation")
    st.caption("Choose a dashboard section using the tabs.")

    health = api_request("GET", "/health")

    if health is not None and health.ok:
        st.success("● API Online")
        try:
            health_data = health.json()

            if "model_loaded" in health_data:
                st.write(
                    "Model loaded: "
                    + (
                        "Yes"
                        if health_data["model_loaded"]
                        else "No"
                    )
                )

            if "uptime_seconds" in health_data:
                uptime = health_data["uptime_seconds"]
                st.write(f"API uptime: {uptime:.0f} seconds")

        except (ValueError, TypeError):
            pass
    else:
        st.error("● API Offline")

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; padding:15px 5px;">
            <p style="color:#d7eddd;">
                Healthy Beans,<br>
                Food Secure Tomorrow 🌱
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div style="font-size:15px; color:#c6edca;">
            🌱 AGRICULTURAL AI FOR A GREENER FUTURE
        </div>
        <h1>BeanGuard</h1>
        <p>
            Spot bean leaf diseases early.<br>
            Make informed decisions and keep your crops healthy.
        </p>
        <div style="margin-top:20px; color:#d8f1d9;">
            <b>Angular Leaf Spot</b>
            &nbsp; • &nbsp;
            <b>Bean Rust</b>
            &nbsp; • &nbsp;
            <b>Healthy Leaves</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD TABS
# =========================================================

diagnose, insights, upload, monitoring = st.tabs([
    "🔍 Diagnose",
    "📊 Data Insights",
    "📤 Upload & Retrain",
    "📈 Monitoring"
])


# =========================================================
# DIAGNOSE
# =========================================================

with diagnose:

    st.header("Bean Leaf Diagnosis")
    st.write(
        "Upload a clear image of a bean leaf to identify "
        "possible disease symptoms."
    )

    left, right = st.columns([1, 1])

    with left:

        st.markdown(
            '<div class="bg-card"><h3>🌿 Upload a Bean Leaf Photo</h3>',
            unsafe_allow_html=True
        )

        image = st.file_uploader(
            "Choose a bean leaf image",
            type=["jpg", "jpeg", "png"],
            key="diagnosis_image"
        )

        if image is not None:
            st.image(
                image,
                caption="Selected bean leaf",
                use_column_width=True
            )

        diagnose_button = st.button(
            "🔍 Diagnose Leaf",
            type="primary",
            key="diagnose_button"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown(
            '<div class="bg-card"><h3>🧪 Diagnosis Result</h3>',
            unsafe_allow_html=True
        )

        if diagnose_button:

            if image is None:
                st.warning("Please upload a bean leaf image first.")

            else:
                with st.spinner("Analyzing your bean leaf..."):

                    image.seek(0)

                    response = api_request(
                        "POST",
                        "/predict",
                        files={
                            "file": (
                                image.name,
                                image.getvalue(),
                                image.type
                            )
                        }
                    )

                    if response is not None and response.ok:

                        try:
                            result = response.json()

                            st.success("Analysis completed.")
                            display_result(result)

                            with st.expander("View full prediction response"):
                                st.json(result)

                        except ValueError:
                            st.error("The API returned an invalid response.")

                    else:
                        show_api_error(response)

        else:
            st.info(
                "Your diagnosis will appear here after you upload "
                "an image and select Diagnose Leaf."
            )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# DATA INSIGHTS
# =========================================================

with insights:

    st.header("📊 Dataset Insights")

    st.write(
        "Explore information about the bean leaf dataset "
        "used by the classification model."
    )

    if st.button("Load Dataset Insights"):

        with st.spinner("Loading dataset information..."):

            data = api_json("/insights")

            if data is not None:

                st.success("Dataset information loaded.")

                if isinstance(data, dict):

                    metric_items = [
                        (key, value)
                        for key, value in data.items()
                        if isinstance(value, (int, float))
                    ]

                    if metric_items:

                        columns = st.columns(
                            min(4, len(metric_items))
                        )

                        for index, (key, value) in enumerate(
                            metric_items[:4]
                        ):

                            with columns[index]:
                                st.metric(
                                    key.replace("_", " ").title(),
                                    value
                                )

                st.markdown("### Dataset information")
                st.json(data)

            else:
                st.warning(
                    "Dataset insights are not currently available."
                )


# =========================================================
# UPLOAD AND RETRAIN
# =========================================================

with upload:

    st.header("📤 Upload New Bean Leaf Images")

    st.write(
        "Add correctly labelled images to support future model retraining."
    )

    st.warning(
        "Use real bean leaf images with verified labels. "
        "Incorrect labels can reduce model quality."
    )

    st.markdown(
        '<div class="bg-card"><h3>Choose a Disease Category</h3>',
        unsafe_allow_html=True
    )

    selected_label = st.selectbox(
        "Image category",
        [
            "angular_leaf_spot",
            "bean_rust",
            "healthy"
        ]
    )

    training_images = st.file_uploader(
        "Upload labelled bean leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="training_images"
    )

    if training_images:
        st.success(
            f"{len(training_images)} image(s) selected."
        )

        for item in training_images:
            st.write(f"📄 {item.name}")

    if st.button("Upload Images", type="primary"):

        if not training_images:
            st.warning("Select at least one image.")

        else:

            files = []

            for item in training_images:
                item.seek(0)

                files.append(
                    (
                        "files",
                        (
                            item.name,
                            item.getvalue(),
                            item.type
                        )
                    )
                )

            with st.spinner("Uploading images..."):

                response = api_request(
                    "POST",
                    "/upload",
                    files=files,
                    data={"label": selected_label}
                )

                if response is not None and response.ok:

                    st.success("Images uploaded successfully!")

                    try:
                        st.json(response.json())
                    except ValueError:
                        st.write(response.text)

                else:
                    show_api_error(response)

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🔄 Model Retraining")

    st.write(
        "Start retraining after new labelled images have been uploaded."
    )

    if st.button("Start Retraining", type="primary"):

        with st.spinner("Submitting retraining request..."):

            response = api_request("POST", "/retrain")

            if response is not None and response.ok:

                st.success("Retraining request submitted.")

                try:
                    st.json(response.json())
                except ValueError:
                    st.write(response.text)

            else:
                show_api_error(response)

    st.markdown("### Retraining Status")

    if st.button("Check Retraining Status"):

        status = api_json("/retrain/status")

        if status is not None:
            st.json(status)
        else:
            st.warning("Retraining status is unavailable.")


# =========================================================
# MONITORING
# =========================================================

with monitoring:

    st.header("📈 Model Monitoring")

    st.write(
        "Monitor the API service, model metrics, and recent predictions."
    )

    # API STATUS
    st.markdown("### API Status")

    status = api_json("/status")

    if status is not None:
        st.success("API status information retrieved.")
        st.json(status)
    else:
        st.warning("API status information is unavailable.")

    st.divider()

    # MODEL METRICS
    st.markdown("### Model Performance")

    metrics = api_json("/metrics")

    if metrics is not None:

        if isinstance(metrics, dict):

            metric_items = [
                (key, value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            ]

            if metric_items:

                columns = st.columns(
                    min(4, len(metric_items))
                )

                for index, (key, value) in enumerate(
                    metric_items[:4]
                ):

                    with columns[index]:
                        st.metric(
                            key.replace("_", " ").title(),
                            value
                        )

        with st.expander("View complete model metrics"):
            st.json(metrics)

    else:
        st.info("Model metrics are unavailable.")

    st.divider()

    # RECENT PREDICTIONS
    st.markdown("### Recent Predictions")

    limit = st.selectbox(
        "Number of records",
        [5, 10, 20, 50, 100],
        index=1
    )

    if st.button("Load Recent Predictions"):

        records = api_json(
            f"/predictions/recent?limit={limit}"
        )

        if records is not None:

            if isinstance(records, dict):

                records = (
                    records.get("predictions")
                    or records.get("data")
                    or records.get("results")
                    or records
                )

            if isinstance(records, list):

                if records:
                    st.dataframe(
                        pd.DataFrame(records)
                    )
                else:
                    st.info("No prediction records found.")

            elif isinstance(records, dict):
                st.json(records)

            else:
                st.write(records)

        else:
            st.warning("Recent predictions are unavailable.")

    st.divider()

    if st.button("Refresh Dashboard"):
        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <b>BeanGuard</b> · Smart Bean Leaf Disease Detection<br>
        Supporting healthier bean crops through machine learning 🌱
    </div>
    """,
    unsafe_allow_html=True
)
