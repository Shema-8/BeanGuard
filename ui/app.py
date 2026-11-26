
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
