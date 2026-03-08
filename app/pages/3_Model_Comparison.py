"""Model Comparison page with tabs for each model category."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import DATA_PROCESSED, FIGURES_DIR
from app.components.sidebar import render_sidebar
from app.components.charts import model_comparison_bar

st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="wide")
render_sidebar()

st.title("Model Comparison")

tab_reg, tab_cls, tab_dl, tab_tuned, tab_all = st.tabs([
    "Regression", "Classification", "Deep Learning", "Tuned (Optuna)", "Overall"
])

# --- Regression Tab ---
with tab_reg:
    st.subheader("Regression Models (AQI Value Prediction)")
    try:
        reg_df = pd.read_csv(DATA_PROCESSED / "regression_results.csv", index_col=0)
        st.dataframe(reg_df.round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = model_comparison_bar(reg_df, "R2", "R² Score (higher is better)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = model_comparison_bar(reg_df, "RMSE", "RMSE (lower is better)")
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Best Model**: {reg_df.index[0]} — R²={reg_df.iloc[0]['R2']:.4f}, RMSE={reg_df.iloc[0]['RMSE']:.2f}")

        # Show saved figures
        for fig_name in ["15_regression_comparison.png", "16_predicted_vs_actual.png", "17_residuals.png"]:
            fig_path = FIGURES_DIR / fig_name
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
    except FileNotFoundError:
        st.info("Regression results not available. Run notebook 04 first.")

# --- Classification Tab ---
with tab_cls:
    st.subheader("Classification Models (AQI Category Prediction)")
    try:
        cls_df = pd.read_csv(DATA_PROCESSED / "classification_results.csv", index_col=0)
        st.dataframe(cls_df.round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = model_comparison_bar(cls_df, "Accuracy", "Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = model_comparison_bar(cls_df, "Macro_F1", "Macro F1 Score")
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Best Model**: {cls_df.index[0]} — Accuracy={cls_df.iloc[0]['Accuracy']:.4f}")

        for fig_name in ["19_classification_comparison.png", "20_confusion_matrices.png"]:
            fig_path = FIGURES_DIR / fig_name
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
    except FileNotFoundError:
        st.info("Classification results not available. Run notebook 05 first.")

# --- Deep Learning Tab ---
with tab_dl:
    st.subheader("Deep Learning Models (Time-Series)")
    try:
        dl_df = pd.read_csv(DATA_PROCESSED / "deep_learning_results.csv", index_col=0)
        st.dataframe(dl_df.round(4), use_container_width=True)

        fig = model_comparison_bar(dl_df, "R2", "Deep Learning R² Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Best DL Model**: {dl_df.index[0]} — R²={dl_df.iloc[0]['R2']:.4f}")

        for fig_name in ["21_dl_comparison.png", "22_dl_training_history.png", "23_dl_predicted_vs_actual.png"]:
            fig_path = FIGURES_DIR / fig_name
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
    except FileNotFoundError:
        st.info("Deep learning results not available. Run notebook 06 first.")

# --- Tuned Tab ---
with tab_tuned:
    st.subheader("Optuna-Tuned Models")
    try:
        tuned_df = pd.read_csv(DATA_PROCESSED / "optuna_results.csv", index_col=0)
        st.dataframe(tuned_df.round(4), use_container_width=True)

        fig = model_comparison_bar(tuned_df, "R2", "Tuned Models R² Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Best Tuned Model**: {tuned_df.index[0]} — R²={tuned_df.iloc[0]['R2']:.4f}")

        fig_path = FIGURES_DIR / "25_optuna_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
    except FileNotFoundError:
        st.info("Optuna results not available. Run notebook 07 first.")

# --- Overall Tab ---
with tab_all:
    st.subheader("Overall Comparison: ML vs DL vs Tuned")
    try:
        all_df = pd.read_csv(DATA_PROCESSED / "all_model_comparison.csv", index_col=0)
        st.dataframe(all_df.round(4), use_container_width=True)

        fig = model_comparison_bar(all_df, "R2", "All Models R² Score")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Overall Best**: {all_df.index[0]} — R²={all_df.iloc[0]['R2']:.4f}")

        fig_path = FIGURES_DIR / "30_final_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
    except FileNotFoundError:
        # Build from available pieces
        dfs = []
        try:
            reg = pd.read_csv(DATA_PROCESSED / "regression_results.csv", index_col=0)
            reg.index = ["[ML] " + n for n in reg.index]
            dfs.append(reg[["R2", "RMSE", "MAE"]])
        except FileNotFoundError:
            pass
        try:
            dl = pd.read_csv(DATA_PROCESSED / "deep_learning_results.csv", index_col=0)
            dl.index = ["[DL] " + n for n in dl.index]
            dfs.append(dl[["R2", "RMSE", "MAE"]])
        except FileNotFoundError:
            pass
        if dfs:
            combined = pd.concat(dfs).sort_values("R2", ascending=False)
            st.dataframe(combined.round(4), use_container_width=True)
            fig = model_comparison_bar(combined, "R2", "Available Models R² Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results available yet. Run the notebooks first.")
