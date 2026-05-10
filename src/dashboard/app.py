import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.classify import _embed, _load_models
from src.pipeline.prioritize import score_priority
from src.pipeline.similarity import build_fused_reports, find_similar, run_temporal_dbscan

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_incidents.csv"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.jsonl"

PRIORITY_COLOR = {"high": "#e74c3c", "medium": "#e67e22", "low": "#27ae60"}
LABEL_COLOR = {
    "authentication_failure": "#3498db",
    "deployment_issue": "#9b59b6",
    "network_issue": "#e67e22",
}


# ---------------------------------------------------------------------------
# Data helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading DistilBERT and classifier...")
def load_models():
    return _load_models()


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    return embeddings, df


@st.cache_data(show_spinner="Running DBSCAN clustering...")
def load_clusters(embeddings_hash: int) -> tuple[list[dict], np.ndarray]:
    embeddings, df = load_dataset()
    cluster_labels = run_temporal_dbscan(embeddings, df["timestamp"])
    reports = build_fused_reports(df, embeddings, cluster_labels)
    return reports, cluster_labels


def get_cluster_details(cluster_id: int, reports: list[dict], df: pd.DataFrame) -> Optional[dict]:
    report = next((r for r in reports if r["cluster_id"] == cluster_id), None)
    if report is None:
        return None
    signals = df[df["id"].isin(report["incident_ids"])].to_dict("records")
    return {**report, "signals": signals}


def log_feedback(cluster_id: int, action: str, reason: Optional[str] = None) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cluster_id": cluster_id,
        "action": action,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with FEEDBACK_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_feedback() -> pd.DataFrame:
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(columns=["cluster_id", "action", "reason", "timestamp"])
    records = [json.loads(line) for line in FEEDBACK_PATH.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["cluster_id", "action", "reason", "timestamp"])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def render_correlation_tab(embeddings: np.ndarray, df: pd.DataFrame) -> None:
    reports, cluster_labels = load_clusters(id(embeddings))

    if not reports:
        st.warning("No correlated clusters found. Try lowering the DBSCAN threshold.")
        return

    n_clusters = len(reports)
    n_noise = int((cluster_labels == -1).sum())
    high_count = sum(1 for r in reports if r["priority"] == "high")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters", n_clusters)
    m2.metric("Singletons", n_noise)
    m3.metric("High priority", high_count)
    m4.metric("Total signals", sum(r["signal_count"] for r in reports))

    st.markdown("---")

    cluster_options = {
        f"Cluster {r['cluster_id']}  |  {r['type'].replace('_', ' ')}  |  {r['priority'].upper()}  |  {r['signal_count']} signals": r["cluster_id"]
        for r in sorted(reports, key=lambda x: (x["priority"] != "high", -x["confidence"]))
    }

    selected_label = st.selectbox("Select incident cluster", list(cluster_options.keys()))
    selected_id = cluster_options[selected_label]
    details = get_cluster_details(selected_id, reports, df)

    if details is None:
        st.error("Cluster details not found.")
        return

    st.markdown("---")
    col_meta, col_chart = st.columns([1, 1], gap="large")

    with col_meta:
        priority = details["priority"]
        badge_color = PRIORITY_COLOR[priority]
        st.markdown(
            f"**Priority** &nbsp;"
            f'<span style="background:{badge_color};color:white;padding:4px 12px;border-radius:4px;font-weight:bold">'
            f'{priority.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Type** &nbsp; `{details['type'].replace('_', ' ').title()}`")
        st.markdown(f"**Confidence** &nbsp; `{details['confidence']:.2f}`")
        st.markdown(f"**Sources** &nbsp; `{', '.join(details['sources'])}`")
        st.markdown(f"**Signals** &nbsp; `{details['signal_count']}`")

    with col_chart:
        member_ids = set(details["incident_ids"])
        plot_df = df.copy()
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["role"] = plot_df["id"].apply(lambda i: "selected cluster" if i in member_ids else "other")
        fig = px.scatter(
            plot_df, x="x", y="y",
            color="role",
            color_discrete_map={"selected cluster": "#e74c3c", "other": "#bdc3c7"},
            symbol="source_type",
            hover_data=["id", "label", "source_type", "priority"],
            opacity=0.7,
            height=280,
            title="Cluster position in embedding space (PCA)",
        )
        fig.update_traces(marker_size=6)
        fig.update_layout(legend_title_text=None, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Related signals")
    for signal in sorted(details["signals"], key=lambda s: s["timestamp"]):
        st.text_area(
            f"{signal['source_type'].upper()}  —  {signal['timestamp']}  —  {signal['label'].replace('_', ' ')}  —  priority: {signal['priority']}",
            value=signal["text"],
            height=90,
            key=f"signal_{signal['id']}",
        )

    st.markdown("---")
    feedback = load_feedback()
    already_logged = not feedback.empty and (feedback["cluster_id"] == selected_id).any()

    if already_logged:
        last = feedback[feedback["cluster_id"] == selected_id].iloc[-1]
        st.info(f"Feedback already logged for this cluster: **{last['action']}**"
                + (f" — reason: {last['reason']}" if last["reason"] else ""))

    col1, col2 = st.columns(2)
    if col1.button("Accept correlation", type="primary", use_container_width=True):
        log_feedback(selected_id, "accept")
        st.success("Feedback logged: accepted")
        st.cache_data.clear()

    with col2:
        with st.expander("Reject correlation"):
            reason = st.selectbox(
                "Reason",
                ["false_positive", "wrong_type", "wrong_priority"],
                key=f"reason_{selected_id}",
            )
            if st.button("Confirm rejection", use_container_width=True):
                log_feedback(selected_id, "reject", reason)
                st.warning(f"Feedback logged: rejected — {reason}")
                st.cache_data.clear()


def render_analyze_tab(embeddings: np.ndarray, df: pd.DataFrame) -> None:
    st.subheader("Analyze a new incident signal")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        text = st.text_area(
            "Incident description",
            placeholder="e.g. Token validation failed for user admin",
            height=130,
        )
        source_type = st.selectbox("Source type", ["ticket", "log", "alert"])
        run = st.button("Analyze", type="primary", use_container_width=True)

    if run and text.strip():
        _, _, clf, label_classes = load_models()
        embedding = _embed(text)
        probs = clf.predict_proba(embedding)[0]
        predicted_idx = int(np.argmax(probs))
        classification = {
            "label": label_classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "all_probs": {label_classes[i]: float(probs[i]) for i in range(len(label_classes))},
        }
        priority = score_priority(text, source_type)

        with col_out:
            st.markdown("**Classification result**")
            c1, c2 = st.columns(2)
            c1.metric("Incident type", classification["label"].replace("_", " ").title())
            c2.metric("Confidence", f"{classification['confidence']:.0%}")

            badge = PRIORITY_COLOR[priority["priority"]]
            st.markdown(
                f"**Priority** &nbsp;"
                f'<span style="background:{badge};color:white;padding:3px 10px;border-radius:4px">'
                f'{priority["priority"].upper()}</span>',
                unsafe_allow_html=True,
            )

            prob_df = pd.DataFrame([
                {"label": k.replace("_", " "), "confidence": v}
                for k, v in classification["all_probs"].items()
            ])
            fig = px.bar(
                prob_df, x="confidence", y="label", orientation="h",
                range_x=[0, 1], height=170, color="label",
                color_discrete_map={k.replace("_", " "): v for k, v in LABEL_COLOR.items()},
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=4, b=0),
                              yaxis_title=None, xaxis_title="confidence")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Similar incidents in dataset** (cosine similarity ≥ 0.80)")
        similar = find_similar(embedding, embeddings, df["id"].tolist(), threshold=0.80)
        similar = sorted(similar, key=lambda x: x["similarity"], reverse=True)[:8]

        if similar:
            sim_scores = {s["id"]: s["similarity"] for s in similar}
            sim_df = df[df["id"].isin(sim_scores)].copy()
            sim_df["similarity"] = sim_df["id"].map(sim_scores)
            sim_df = sim_df.sort_values("similarity", ascending=False)
            st.dataframe(
                sim_df[["id", "source_type", "label", "priority", "similarity", "text"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No similar incidents found above the 0.80 threshold.")

    elif run:
        st.warning("Please enter incident text.")


def render_dataset_tab(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total signals", len(df))
    m2.metric("Incident groups", df["incident_group_id"].nunique())
    m3.metric("Source types", df["source_type"].nunique())

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.bar(
            df["label"].value_counts().reset_index(),
            x="count", y="label", orientation="h", title="By incident type",
            color="label", color_discrete_map=LABEL_COLOR,
        )
        fig.update_layout(showlegend=False, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(df, names="source_type", title="By source type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.bar(
            df["priority"].value_counts().reset_index(),
            x="priority", y="count", title="By priority",
            color="priority", color_discrete_map=PRIORITY_COLOR,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    label_filter = st.multiselect(
        "Filter by label", df["label"].unique().tolist(),
        default=df["label"].unique().tolist(),
    )
    filtered = df[df["label"].isin(label_filter)]
    st.dataframe(
        filtered[["id", "incident_group_id", "source_type", "label", "priority", "timestamp", "text"]],
        use_container_width=True, hide_index=True, height=360,
    )


def render_feedback_tab() -> None:
    st.subheader("Analyst feedback log")
    df = load_feedback()

    if df.empty:
        st.info("No feedback logged yet. Accept or reject clusters from the Correlation tab.")
        return

    accepted = (df["action"] == "accept").sum()
    rejected = (df["action"] == "reject").sum()
    m1, m2 = st.columns(2)
    m1.metric("Accepted", int(accepted))
    m2.metric("Rejected", int(rejected))

    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Clear all feedback"):
        FEEDBACK_PATH.unlink(missing_ok=True)
        st.success("Feedback cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Incident Intelligence System",
        layout="wide",
    )
    st.title("ML-Based Incident Intelligence System")
    st.caption("Prototype · DistilBERT embeddings · Logistic Regression · DBSCAN correlation")

    try:
        embeddings, df = load_dataset()
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e}. Run src/features/embed.py first.")
        return

    load_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation Dashboard", "Analyze Incident", "Dataset", "Feedback Log"
    ])

    with tab1:
        render_correlation_tab(embeddings, df)
    with tab2:
        render_analyze_tab(embeddings, df)
    with tab3:
        render_dataset_tab(df)
    with tab4:
        render_feedback_tab()


if __name__ == "__main__":
    main()
