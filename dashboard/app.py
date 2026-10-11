"""VoiceLab Dashboard — Streamlit application."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import requests

import api_client

st.set_page_config(page_title="VoiceLab", page_icon="🎙️", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["New Benchmark", "Runs", "Compare", "Dataset"],
)

# Backend status in sidebar
try:
    health = api_client.health()
    st.sidebar.success(f"API: connected ({health['device']})")
except Exception:
    st.sidebar.error("API: not connected")


# ---------------------------------------------------------------------------
# Page: New Benchmark
# ---------------------------------------------------------------------------

if page == "New Benchmark":
    st.title("New Benchmark")

    try:
        models = api_client.list_models()
        model_names = [m["name"] for m in models]
    except Exception:
        st.error("Could not load models from API. Is the backend running?")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model = st.selectbox("Model", model_names)
    with col2:
        num_samples = st.number_input("Samples", min_value=1, max_value=10000, value=200)
    with col3:
        seed = st.number_input("Seed", min_value=0, value=42)
    with col4:
        device = st.selectbox("Device", ["auto", "mps", "cpu"])

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark... This may take several minutes."):
            try:
                result = api_client.run_benchmark(
                    model=model,
                    num_samples=num_samples,
                    seed=seed,
                    device=device,
                )
                st.success(f"Benchmark complete! Run ID: `{result['run_id']}`")

                # Show summary metrics
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                mcol1.metric("WER", f"{result['wer']:.4f}")
                mcol2.metric("CER", f"{result['cer']:.4f}")
                mcol3.metric("RTF", f"{result['rtf']:.3f}")
                mcol4.metric("Avg Latency", f"{result['avg_latency_s']:.3f}s")
                mcol5.metric("Runtime", f"{result['total_runtime_s']:.1f}s")

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 409:
                    st.warning("A benchmark is already running. Please wait for it to complete.")
                else:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    st.error(f"Benchmark failed: {detail}")
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Page: Runs
# ---------------------------------------------------------------------------

elif page == "Runs":
    st.title("Benchmark Runs")

    try:
        runs = api_client.list_runs()
    except Exception as e:
        st.error(f"Could not load runs: {e}")
        st.stop()

    if not runs:
        st.info("No benchmark runs yet. Create one from the New Benchmark page.")
        st.stop()

    # Build a display dataframe
    df = pd.DataFrame(runs)
    display_cols = [
        "run_id", "status", "model_name", "device", "num_samples",
        "wer", "cer", "rtf", "avg_latency_s", "total_runtime_s",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Shorten run IDs for display
    df_display["run_id_short"] = df_display["run_id"].str[:8]

    # Format floats
    for col in ["wer", "cer", "rtf", "avg_latency_s", "total_runtime_s"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Run detail viewer
    st.subheader("Run Detail")
    run_options = {f"{r['run_id'][:8]} — {r.get('model_name', '?')}": r["run_id"] for r in runs}
    selected_label = st.selectbox("Select a run", list(run_options.keys()))

    if selected_label:
        run_id = run_options[selected_label]

        try:
            detail = api_client.get_run(run_id)
        except Exception as e:
            st.error(f"Could not load run: {e}")
            st.stop()

        # Status badge
        status = detail["status"]
        if status == "FINISHED":
            st.success(f"Status: {status}")
        elif status == "FAILED":
            st.error(f"Status: {status}")
            error_tag = detail.get("tags", {}).get("voicelab.error", "")
            if error_tag:
                st.caption(f"Error: {error_tag}")
        else:
            st.warning(f"Status: {status}")

        # Metrics
        st.subheader("Metrics")
        metrics = detail.get("metrics", {})
        if metrics:
            metric_cols = st.columns(min(len(metrics), 5))
            for i, (key, val) in enumerate(sorted(metrics.items())):
                col = metric_cols[i % len(metric_cols)]
                col.metric(key, f"{val:.4f}" if isinstance(val, float) else str(val))

        # Parameters
        with st.expander("Parameters"):
            params = detail.get("params", {})
            st.json(params)

        # Tags
        with st.expander("Tags"):
            tags = detail.get("tags", {})
            st.json(tags)

        # Predictions
        st.subheader("Predictions")
        try:
            predictions = api_client.get_predictions(run_id)
        except Exception:
            predictions = []

        if predictions:
            pred_df = pd.DataFrame(predictions)

            # Sort controls
            sort_col = st.selectbox(
                "Sort by",
                ["wer", "cer", "latency_s", "sample_id"],
                index=0,
            )
            if sort_col in ["wer", "cer", "latency_s"]:
                pred_df[sort_col] = pd.to_numeric(pred_df[sort_col], errors="coerce")
                pred_df = pred_df.sort_values(sort_col, ascending=False, na_position="last")

            # Filter controls
            show_errors = st.checkbox("Show only errors/excluded", value=False)
            if show_errors:
                pred_df = pred_df[
                    (pred_df["excluded"] == "True") | (pred_df["error"].str.len() > 0)
                ]

            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Download CSV
            try:
                csv_bytes = api_client.get_predictions_csv(run_id)
                st.download_button(
                    "Download Predictions CSV",
                    data=csv_bytes,
                    file_name=f"predictions_{run_id[:8]}.csv",
                    mime="text/csv",
                )
            except Exception:
                pass
        else:
            st.info("No predictions available.")

    # Multi-select for comparison
    st.subheader("Compare Runs")
    compare_ids = st.multiselect(
        "Select runs to compare",
        options=[r["run_id"] for r in runs],
        format_func=lambda x: f"{x[:8]} — {next((r.get('model_name','?') for r in runs if r['run_id']==x), '?')}",
    )
    if len(compare_ids) >= 2:
        if st.button("Compare Selected"):
            st.session_state["compare_ids"] = compare_ids
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Compare
# ---------------------------------------------------------------------------

elif page == "Compare":
    st.title("Compare Benchmark Runs")

    compare_ids = st.session_state.get("compare_ids", [])

    if not compare_ids:
        st.info("Select runs to compare from the Runs page, or enter run IDs below.")
        ids_input = st.text_input("Run IDs (comma-separated)")
        if ids_input:
            compare_ids = [rid.strip() for rid in ids_input.split(",") if rid.strip()]

    if len(compare_ids) < 2:
        st.warning("Select at least 2 runs to compare.")
        st.stop()

    try:
        comparison = api_client.compare_runs(compare_ids)
    except Exception as e:
        st.error(f"Comparison failed: {e}")
        st.stop()

    runs_data = comparison["runs"]

    # Metrics comparison table
    st.subheader("Metrics")
    metric_keys = sorted({k for r in runs_data for k in r.get("metrics", {})})
    rows = []
    for r in runs_data:
        row = {"run_id": r["run_id"][:8], "status": r["status"]}
        row.update({k: r["metrics"].get(k) for k in metric_keys})
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Bar charts for key metrics
    chart_metrics = ["wer", "cer", "rtf", "avg_latency_s"]
    available_charts = [m for m in chart_metrics if m in metric_keys]

    if available_charts:
        st.subheader("Charts")
        chart_cols = st.columns(min(len(available_charts), 2))
        for i, metric in enumerate(available_charts):
            col = chart_cols[i % len(chart_cols)]
            chart_data = pd.DataFrame(
                {
                    "Run": [r["run_id"][:8] for r in runs_data],
                    metric: [r["metrics"].get(metric, 0) for r in runs_data],
                }
            )
            col.bar_chart(chart_data, x="Run", y=metric)

    # Highlight differing params
    st.subheader("Parameters")
    param_keys = sorted({k for r in runs_data for k in r.get("params", {})})
    param_rows = []
    for r in runs_data:
        row = {"run_id": r["run_id"][:8]}
        row.update({k: r["params"].get(k, "") for k in param_keys})
        param_rows.append(row)
    params_df = pd.DataFrame(param_rows)

    # Highlight columns where values differ
    differing = [
        c for c in param_keys if params_df[c].nunique() > 1
    ]
    if differing:
        st.caption(f"Differing parameters: {', '.join(differing)}")

    st.dataframe(params_df, use_container_width=True, hide_index=True)

    # Download comparison CSV
    try:
        csv_bytes = api_client.get_comparison_csv(compare_ids)
        st.download_button(
            "Download Comparison CSV",
            data=csv_bytes,
            file_name="comparison.csv",
            mime="text/csv",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page: Dataset
# ---------------------------------------------------------------------------

elif page == "Dataset":
    st.title("Dataset Info")

    try:
        datasets = api_client.list_datasets()
    except Exception as e:
        st.error(f"Could not load dataset info: {e}")
        st.stop()

    if not datasets:
        st.info("No dataset configured.")
        st.stop()

    ds = datasets[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Name", ds["name"])
        st.metric("Version", ds["version"])
        st.metric("Language", ds["language"])
        st.metric("Split", ds["split"])

    with col2:
        st.metric("Subset Size", ds["num_samples"])
        st.metric("Seed", ds["seed"])
        st.metric(
            "Total Audio Duration",
            f"{ds['total_audio_duration_s']:.1f}s" if ds["total_audio_duration_s"] else "—",
        )
        cached_status = "Cached" if ds["cached"] else "Not cached"
        st.metric("Cache Status", cached_status)

    if ds.get("sample_ids_hash"):
        st.caption(f"Sample IDs hash: `{ds['sample_ids_hash']}`")
