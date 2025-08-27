import os
import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def make_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Lags and rolling means
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}"] = df["load_kw"].shift(lag)
    for win in [3, 6, 12, 24]:
        df[f"rollmean_{win}"] = df["load_kw"].shift(1).rolling(win).mean()

    df = df.dropna().reset_index(drop=True)
    return df

def train_and_evaluate(df):
    # Use last 7 days as test
    max_ts = df["timestamp"].max()
    split_point = max_ts - pd.Timedelta(days=7)

    train = df[df["timestamp"] <= split_point].copy()
    test = df[df["timestamp"] > split_point].copy()

    features = [c for c in df.columns if c not in ["timestamp", "building_id", "load_kw"]]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[features], train["load_kw"])

    test["y_pred"] = model.predict(test[features])
    mae = mean_absolute_error(test["load_kw"], test["y_pred"])
    r2 = r2_score(test["load_kw"], test["y_pred"])

    # Anomaly detection on residuals using z-score
    residuals = test["load_kw"] - test["y_pred"]
    mu = residuals.mean()
    sigma = residuals.std(ddof=1) if residuals.std(ddof=1) != 0 else 1.0
    z = (residuals - mu) / sigma
    test["anomaly"] = (np.abs(z) > 2.5).astype(int)

    return model, train, test, features, mae, r2

def forecast_next_24(df_full, model, features):
    df = df_full.copy()
    last_ts = df["timestamp"].max()
    horizon = 24
    future_rows = []

    # We will perform recursive forecasting
    df_work = df.copy()
    for h in range(1, horizon + 1):
        ts = last_ts + pd.Timedelta(hours=h)
        row = {"timestamp": ts, "building_id": df["building_id"].iloc[-1]}

        # Build feature row using latest df_work (which will include appended preds)
        tmp = df_work.copy()
        tmp = make_features(tmp)
        # Keep only last row of tmp for feature creation ref; then override timestamp/hour/day features
        feat = tmp.iloc[-1:].copy()
        feat["timestamp"] = ts
        feat["hour"] = ts.hour
        feat["dayofweek"] = ts.dayofweek
        feat["is_weekend"] = int(ts.dayofweek in [5, 6])

        # Update lag and rolling features based on df_work
        # Recompute lags/rolls by temporarily appending a placeholder; then extract values
        tmp2 = df_work.copy()
        tmp2 = tmp2.sort_values("timestamp").copy()
        # lags need actual load_kw values; for lags we can use existing and predicted appended so far
        for lag in [1, 2, 3, 6, 12, 24]:
            col = f"lag_{lag}"
            # construct by shifting the series
            tmp2[col] = tmp2["load_kw"].shift(lag)
        for win in [3, 6, 12, 24]:
            col = f"rollmean_{win}"
            tmp2[col] = tmp2["load_kw"].shift(1).rolling(win).mean()

        last_feat_row = tmp2.dropna().iloc[-1]

        for col in [c for c in features if c.startswith("lag_") or c.startswith("rollmean_")]:
            feat[col] = last_feat_row[col]

        # Now make prediction
        X_cols = [c for c in features]
        y_pred = model.predict(feat[X_cols])[0]

        row["load_kw"] = y_pred
        future_rows.append(row)

        # Append predicted point to df_work so next step can compute new lags
        df_work = pd.concat([df_work, pd.DataFrame([row])], ignore_index=True)

    future_df = pd.DataFrame(future_rows)
    return future_df

def plot_results(train, test, future_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot test actual vs predicted
    plt.figure()
    plt.plot(test["timestamp"], test["load_kw"], label="Actual")
    plt.plot(test["timestamp"], test["y_pred"], label="Predicted")
    plt.title("Test Set: Actual vs Predicted")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (kW)")
    plt.legend()
    test_plot_path = os.path.join(output_dir, "test_actual_vs_pred.png")
    plt.savefig(test_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot anomalies
    plt.figure()
    plt.plot(test["timestamp"], test["load_kw"], label="Actual")
    anomalies = test[test["anomaly"] == 1]
    if len(anomalies) > 0:
        plt.scatter(anomalies["timestamp"], anomalies["load_kw"], label="Anomaly", marker="x")
    plt.title("Anomaly Detection (Test Residual Z-score > 2.5)")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (kW)")
    plt.legend()
    anom_plot_path = os.path.join(output_dir, "anomalies.png")
    plt.savefig(anom_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 24h forecast
    plt.figure()
    plt.plot(future_df["timestamp"], future_df["load_kw"], label="Forecast 24h")
    plt.title("Next 24 Hours Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (kW)")
    plt.legend()
    fc_plot_path = os.path.join(output_dir, "forecast_next_24h.png")
    plt.savefig(fc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return test_plot_path, anom_plot_path, fc_plot_path

def main():
    parser = argparse.ArgumentParser(description="Short-term Energy Load Forecasting")
    parser.add_argument("--data", type=str, default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default="outputs", help="Directory to save plots")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    feat_df = make_features(df)
    model, train, test, features, mae, r2 = train_and_evaluate(feat_df)

    future_df = forecast_next_24(feat_df[["timestamp", "building_id", "load_kw"]], model, features)
    test_plot, anom_plot, fc_plot = plot_results(train, test, future_df, args.out)

    print("Evaluation on test set:")
    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}")
    print("Saved plots:")
    print(test_plot)
    print(anom_plot)
    print(fc_plot)

if __name__ == "__main__":
    main()
