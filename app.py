import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from io import BytesIO

st.set_page_config(page_title="Energy Load Forecasting Dashboard", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>⚡ Energy Load Forecasting & Anomaly Detection</h1>
    <p style='text-align: center; font-size: 18px;'>
        Upload your building energy dataset to forecast short-term load, detect anomalies, 
        and generate insights for sustainable energy usage 🌱
    </p>
    <hr>
    """, unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "timestamp" not in df.columns or "load" not in df.columns:
        st.error(" CSV must contain 'timestamp' and 'load' columns")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        df["forecast"] = df["load"].rolling(window=3, min_periods=1).mean().shift(1)
        df["error"] = df["load"] - df["forecast"]
        threshold = df["error"].std() * 2
        anomalies = df[np.abs(df["error"]) > threshold]

        total_points = len(df)
        anomaly_count = len(anomalies)
        efficiency = round(100 - ((anomaly_count / total_points) * 100), 2)

        col1, col2, col3 = st.columns(3)
        col1.markdown(
            f"""
            <div style="background-color:#E8F6F3; padding:20px; border-radius:15px; text-align:center;">
                <h3 style="color:#117A65;">📊 Total Data Points</h3>
                <h2 style="color:#0E6251;">{total_points}</h2>
            </div>
            """, unsafe_allow_html=True
        )
        col2.markdown(
            f"""
            <div style="background-color:#FDEDEC; padding:20px; border-radius:15px; text-align:center;">
                <h3 style="color:#922B21;">⚠️ Anomalies Detected</h3>
                <h2 style="color:#641E16;">{anomaly_count}</h2>
            </div>
            """, unsafe_allow_html=True
        )
        col3.markdown(
            f"""
            <div style="background-color:#FEF9E7; padding:20px; border-radius:15px; text-align:center;">
                <h3 style="color:#9A7D0A;">⚡ Efficiency Score</h3>
                <h2 style="color:#7D6608;">{efficiency}%</h2>
            </div>
            """, unsafe_allow_html=True
        )

        st.subheader("📈 Load Forecasting")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["timestamp"], df["load"], label="Actual Load", color="#2874A6")
        ax.plot(df["timestamp"], df["forecast"], label="Forecast", color="#F39C12", linestyle="dashed")
        ax.set_xlabel("Time")
        ax.set_ylabel("Load (kW)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 🔮 Forecast Summary")
        tomorrow_peak = round(df["forecast"].tail(24).max(), 2)
        tomorrow_avg = round(df["forecast"].tail(24).mean(), 2)

        col1, col2 = st.columns(2)
        col1.markdown(
            f"""
            <div style="background-color:#D6EAF8; padding:20px; border-radius:15px; text-align:center;">
                <h3 style="color:#1B4F72;">Predicted Peak Load</h3>
                <h2 style="color:#154360;">{tomorrow_peak} kW</h2>
            </div>
            """, unsafe_allow_html=True
        )
        col2.markdown(
            f"""
            <div style="background-color:#FCF3CF; padding:20px; border-radius:15px; text-align:center;">
                <h3 style="color:#7D6608;">Predicted Avg Load</h3>
                <h2 style="color:#7D6608;">{tomorrow_avg} kW</h2>
            </div>
            """, unsafe_allow_html=True
        )

        st.subheader("⚠️ Anomaly Detection")
        if not anomalies.empty:
            st.markdown(
                f"<p style='color:red; font-size:18px;'><b>{len(anomalies)}</b> anomalies detected ⚡</p>",
                unsafe_allow_html=True
            )
            st.dataframe(anomalies[["timestamp", "load", "forecast", "error"]].head(10))

            csv_buffer = BytesIO()
            anomalies.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Anomalies Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="anomalies_report.csv",
                mime="text/csv"
            )
        else:
            st.markdown("<p style='color:green; font-size:18px;'>✅ No major anomalies detected</p>", unsafe_allow_html=True)

        st.subheader("🌱 Energy Savings Insight")
        if not anomalies.empty:
            wasted_energy = round(anomalies["error"].abs().sum(), 2)
            trees_saved = int(wasted_energy // 20)  # assume 20 kWh = 1 tree
            st.markdown(
                f"""
                <div style="background-color:#E8F8F5; padding:20px; border-radius:15px; text-align:center;">
                    <h3 style="color:#117864;">Potential Energy Savings</h3>
                    <p style="font-size:18px; color:#1B4F72;">By addressing anomalies, approx.</p>
                    <h2 style="color:#0B5345;">{wasted_energy} kWh</h2>
                    <p style="font-size:18px; color:#1B4F72;">can be saved, equal to planting 🌳 <b>{trees_saved} trees</b></p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.info("🌍 System is energy efficient — No energy wastage detected.")
