import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_model(data):
    X = data[["temperature", "humidity"]]
    y = data["load"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2 Score": r2}

def predict_future(data, hours=24):
    model, _, _, _ = train_model(data)

    last_temp = data["temperature"].iloc[-1]
    last_hum = data["humidity"].iloc[-1]

    future_preds = []
    for h in range(1, hours+1):
        pred = model.predict([[last_temp, last_hum]])[0]
        future_preds.append({"Hour": h, "Forecasted_Load": pred})

    return pd.DataFrame(future_preds)
