"""BeanGuard dashboard (Streamlit): Diagnose | Data insights | Upload & Retrain | Monitoring."""
import os
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (APP_NAME, APP_TAGLINE, class_info, color_map, fmt_uptime,  # noqa: E402
                     result_card_html, status_pill_html)

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title=f"{APP_NAME} - bean leaf disease detector", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.stApp { background: #F6F9F4; }
.block-container { padding-top: 1.4rem; max-width: 1250px; }
.hero { background: linear-gradient(135deg, #1B5E20 0%, #388E3C 55%, #66BB6A 100%);
        padding: 28px 34px; border-radius: 20px; color: #fff; margin-bottom: 18px; }
.hero h1 { color: #fff; margin: 0; font-size: 2.1rem; letter-spacing: -0.5px; }
.hero p { color: #E8F5E9; margin: 6px 0 0 0; font-size: 1.02rem; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #E3EBE0; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
.card { background: #fff; border: 1px solid #E3EBE0; border-radius: 16px; padding: 18px 20px;
        margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.result { border-radius: 18px; padding: 22px 26px; color: #fff; margin-bottom: 12px; }
.r-big { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.r-small { font-size: .95rem; opacity: .95; }
.pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600; }
.advice { background: #FFFBEB; border-left: 5px solid #F59E0B; border-radius: 10px; padding: 12px 16px;
          color: #3F3A2B; }
.step { background: #fff; border: 1px solid #E3EBE0; border-radius: 14px; padding: 14px; height: 100%; }
.step b { color: #2E7D32; }
.insight-title { font-weight: 700; color: #1B5E20; }
</style>
""", unsafe_allow_html=True)


def api(method: str, path: str, **kw):
    try:
        return requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 60), **kw)
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API}: {exc}")
        return None


# ----------------------------------------------------------------------------- sidebar
status_resp = api("GET", "/status", timeout=10)
status = status_resp.json() if status_resp is not None and status_resp.ok else {}
classes = status.get("classes", [])

with st.sidebar:
    st.markdown(f"## 🌿 {APP_NAME}")
    st.markdown(status_pill_html(bool(status)), unsafe_allow_html=True)
    if status:
        st.metric("Model version", f"v{status.get('model_version')}" if status.get("model_version") else "none")
        st.metric("API uptime", fmt_uptime(status["uptime_seconds"]))
        st.caption(f"Served by container `{status['host']}`")
        st.markdown("**Classes**")
        for c in classes:
            i = class_info(c)
            st.markdown(f"{i['emoji']} {i['title']}")
    if st.button("🔄 Refresh"):
        st.rerun()

st.markdown(f'<div class="hero"><h1>🌿 {APP_NAME}</h1><p>{APP_TAGLINE}</p></div>', unsafe_allow_html=True)

tab_pred, tab_viz, tab_retrain, tab_mon = st.tabs(
    ["🔍 Diagnose", "📊 Data insights", "♻️ Upload & Retrain", "📡 Monitoring"])

# ----------------------------------------------------------------------------- diagnose
with tab_pred:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Upload a bean leaf photo")
        file = st.file_uploader("JPG or PNG, one leaf in focus works best",
                                type=["jpg", "jpeg", "png", "webp"], key="pred",
                                label_visibility="collapsed")
        if file:
            st.image(file, caption=file.name, use_column_width=True)
        go_btn = st.button("🔬 Diagnose leaf", type="primary", disabled=file is None)
    with right:
        st.subheader("Result")
        if file and go_btn:
            with st.spinner("Analysing the leaf..."):
                r = api("POST", "/predict", files={"file": (file.name, file.getvalue(), file.type)})
            if r is not None and r.ok:
                res = r.json()
                st.markdown(result_card_html(res), unsafe_allow_html=True)
                info = class_info(res["label"])
                if info["advice"] and not res["uncertain"]:
                    st.markdown(f'<div class="advice">💡 <b>What to do:</b> {info["advice"]}</div>',
                                unsafe_allow_html=True)
                top = pd.DataFrame(res["top3"])
                top["name"] = top["label"].map(lambda x: class_info(x)["title"])
                fig = px.bar(top, x="confidence", y="name", orientation="h", range_x=[0, 1],
                             color="label", color_discrete_map=color_map(top["label"]))
                fig.update_layout(height=210, showlegend=False, yaxis_title=None, xaxis_title="confidence",
                                  yaxis={"categoryorder": "total ascending"},
                                  margin=dict(t=10, b=10, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{res['latency_ms']} ms · served by {res['served_by']}")
            elif r is not None:
                st.error(r.json().get("detail", r.text))
        else:
            st.markdown('<div class="card">Upload a photo and press <b>Diagnose leaf</b>. '
                        'The result, confidence and advice appear here.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- data insights
with tab_viz:
    r = api("GET", "/insights", timeout=120)
    if r is not None and r.ok:
        data = r.json()
        cmap = color_map(data["class_counts"])
        counts = pd.DataFrame({"class": list(data["class_counts"]), "images": list(data["class_counts"].values())})
        feats = pd.DataFrame(data["features"])
        k = st.columns(3)
        k[0].metric("Training images", int(counts["images"].sum()))
        k[1].metric("Classes", len(counts))
        k[2].metric("Imbalance ratio", f"{counts['images'].max() / max(counts['images'].min(), 1):.2f}x")

        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(counts, x="class", y="images", color="class", color_discrete_map=cmap,
                               title="Images per class"), use_container_width=True)
        if not feats.empty:
            c2.plotly_chart(px.box(feats, x="class", y="brightness", color="class", color_discrete_map=cmap,
                                   title="Brightness by class"), use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(px.box(feats, x="class", y="saturation", color="class", color_discrete_map=cmap,
                                   title="Colour saturation by class"), use_container_width=True)
            c4.plotly_chart(px.scatter(feats, x="width", y="height", color="class", opacity=0.6,
                                       color_discrete_map=cmap, title="Original image dimensions"),
                            use_container_width=True)
        st.markdown("### What the data tells us")
        for item in data["interpretations"]:
            st.markdown(f'<div class="card"><div class="insight-title">{item["feature"]}</div>'
                        f'{item["finding"]}<br><span style="color:#5B6B5E">{item["implication"]}</span></div>',
                        unsafe_allow_html=True)
    elif r is not None:
        st.warning(r.json().get("detail", "No data available."))

    st.markdown("### Model quality (held-out test set)")
    m = api("GET", "/metrics")
    if m is not None and m.ok and m.json()["model"].get("test"):
        t = m.json()["model"]["test"]
        cols = st.columns(5)
        for col, (name, key) in zip(cols, [("Accuracy", "accuracy"), ("Loss", "loss"),
                                           ("Precision", "precision_macro"), ("Recall", "recall_macro"),
                                           ("F1 score", "f1_macro")]):
            col.metric(name, f"{t[key]:.3f}")
        cm = pd.DataFrame(t["confusion_matrix"], index=t["class_names"], columns=t["class_names"])
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Greens",
                                  labels=dict(x="Predicted", y="Actual"), title="Confusion matrix"),
                        use_container_width=True)
        pc = pd.DataFrame(t["per_class"]).T.loc[t["class_names"], ["precision", "recall", "f1-score", "support"]]
        c2.markdown("**Per-class results**")
        c2.dataframe(pc.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}",
                                      "support": "{:.0f}"}), use_container_width=True)

# ----------------------------------------------------------------------------- retrain
with tab_retrain:
    pending = status.get("pending_uploads", 0)
    thr = status.get("retrain_threshold", 30)

    st.markdown("### How the model keeps learning")
    steps = [
        ("1", "Upload", "Add labelled leaf photos (or a ZIP with one folder per class)."),
        ("2", "Retrain", "The saved model is fine-tuned on old + new images."),
        ("3", "Compare", "Old and new models are scored on the same untouched test set."),
        ("4", "Promote", "The new model goes live only if it is not worse."),
    ]
    for col, (n, title, text) in zip(st.columns(4), steps):
        col.markdown(f'<div class="step"><b>Step {n} · {title}</b><br>{text}</div>', unsafe_allow_html=True)

    st.markdown("")
    st.progress(min(pending / thr, 1.0), text=f"{pending} new labelled images waiting · "
                                              f"automatic retraining at {thr}")
    by_label = status.get("pending_by_label", {})
    st.caption("Waiting per class: " + (", ".join(f"{k}: {v}" for k, v in by_label.items()) or "none"))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("1 · Upload labelled data")
        mode = st.radio("Upload type", ["Images of one class", "ZIP (one folder per class)"], horizontal=True)
        if mode.startswith("Images"):
            label = st.selectbox("Class label", classes, format_func=lambda c: class_info(c)["title"])
            files = st.file_uploader("Select multiple images", accept_multiple_files=True,
                                     type=["jpg", "jpeg", "png", "webp"], key="bulk")
        else:
            label = None
            files = st.file_uploader("ZIP file, e.g. healthy/*.jpg, bean_rust/*.jpg", type=["zip"],
                                     accept_multiple_files=True, key="zip")
        if files and st.button("⬆️ Upload to database", type="primary"):
            payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            r = api("POST", "/upload", files=payload, data={"label": label} if label else {}, timeout=300)
            if r is not None and r.ok:
                res = r.json()
                st.success(f"Accepted {res['accepted']} · rejected {res['rejected_count']} · "
                           f"pending {res['pending_total']}")
                if res["rejected"]:
                    st.dataframe(pd.DataFrame(res["rejected"]))
                if res["auto_retrain_started"]:
                    st.info("Threshold reached: retraining started automatically.")
            elif r is not None:
                st.error(r.json().get("detail", r.text))
    with c2:
        st.subheader("2 · Trigger retraining")
        if st.button("🚀 Retrain model", disabled=pending == 0):
            r = api("POST", "/retrain")
            if r is not None and r.ok:
                st.success("Retraining started.")
            elif r is not None:
                 try:
                     error_message = r.json().get("detail", r.text)
                 except ValueError:
                     error_message = r.text or f"Server returned HTTP {r.status_code}"

                 st.error(error_message)
        rs = api("GET", "/retrain/status")
        if rs is not None and rs.ok:
            stt = rs.json()
            icon = {"running": "⏳", "completed": "✅", "failed": "❌"}.get(stt["state"], "💤")
            st.markdown(f'<div class="card">{icon} <b>{stt["state"].title()}</b><br>{stt.get("message", "")}</div>',
                        unsafe_allow_html=True)
            if stt["state"] == "completed":
                st.dataframe(pd.DataFrame({"": ["Accuracy", "F1 score"],
                                           "Before": [stt.get("old_accuracy"), stt.get("old_f1")],
                                           "After": [stt.get("new_accuracy"), stt.get("new_f1")]}).set_index(""))
            if stt["state"] == "running":
                time.sleep(4)
                st.rerun()

# ----------------------------------------------------------------------------- monitoring
with tab_mon:
    m = api("GET", "/metrics")
    if m is not None and m.ok:
        serving = m.json()["serving"]
        c = st.columns(5)
        c[0].metric("Uptime", fmt_uptime(status.get("uptime_seconds", 0)))
        c[1].metric("Predictions served", serving["total_predictions"])
        c[2].metric("Avg latency", f"{serving['avg_latency_ms']} ms")
        c[3].metric("p95 latency", f"{serving['p95_latency_ms']} ms")
        c[4].metric("Low-confidence share", f"{serving['uncertain_share']:.1%}")
        rec = api("GET", "/predictions/recent", params={"limit": 500})
        if rec is not None and rec.ok and rec.json():
            df = pd.DataFrame(rec.json())
            df["ts"] = pd.to_datetime(df["ts"])
            cm_ = color_map(df["label"].unique())
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(df.sort_values("ts"), x="ts", y="latency_ms",
                                    title="Inference latency (ms)"), use_container_width=True)
            c2.plotly_chart(px.histogram(df, x="confidence", nbins=20,
                                         title="Prediction confidence (drift signal: watch for a left shift)"),
                            use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(px.pie(df, names="label", color="label", color_discrete_map=cm_,
                                   title="Predicted class mix"), use_container_width=True)
            c4.plotly_chart(px.histogram(df, x="host", title="Requests per container"),
                            use_container_width=True)
        st.markdown("### Model versions")
        versions = pd.DataFrame(m.json()["versions"])
        if not versions.empty:
            st.dataframe(versions, use_container_width=True)
            fig = go.Figure(go.Scatter(x=versions["version"], y=versions["f1_macro"], mode="lines+markers",
                                       line=dict(color="#2E7D32", width=3)))
            fig.update_layout(title="F1 score across model versions", xaxis_title="version",
                              yaxis_title="macro F1", xaxis=dict(dtick=1))
            st.plotly_chart(fig, use_container_width=True)
