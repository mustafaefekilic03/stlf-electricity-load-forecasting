import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Dosya yolu
file_path = r"C:\Users\NEW\Desktop\esasveribaslangic\INPUT_TED_Bitirme_STLF.xlsx"
df = pd.read_excel(file_path)

# ============================================================
# ÖN İŞLEME
# ============================================================
df["Tarih"] = pd.to_datetime(df["Tarih"])
df = df.set_index("Tarih")

# 1. Holiday Features   
df["is_holiday"] = (df["ÖzelGün_Adı"] != "Değil").astype(int)
holiday_map = {
    "Kurban_bayramı": "holiday_kurban",
    "Kurban_bayramı_arife": "holiday_kurban_arife",
    "Ramazan_bayramı": "holiday_ramazan",
    "Ramazan_bayramı_arife": "holiday_ramazan_arife",
    "Milli Bayram": "holiday_milli",
    "Seçim Günü": "holiday_secim",
    "Yılbaşı": "holiday_yilbasi",
}
for col in holiday_map.values():
    df[col] = 0
for raw_value, col_name in holiday_map.items():
    df.loc[df["ÖzelGün_Adı"] == raw_value, col_name] = 1

df["holiday_prev1"] = df["is_holiday"].shift(24).fillna(0).astype(int)
df["holiday_next1"] = df["is_holiday"].shift(-24).fillna(0).astype(int)
df["holiday_week"] = df["is_holiday"].rolling(24*7, min_periods=1).max().astype(int)

# 2. Meteoroloji
mugla_temp_cols = [c for c in df.columns if "Hissedilen_Sıcaklık-MUGLA" in c]
denizli_temp_cols = [c for c in df.columns if "Hissedilen_Sıcaklık-DNZ" in c]
aydin_temp_cols = [c for c in df.columns if "Hissedilen_Sıcaklık-AYD" in c]

df["Mugla_Temp"] = df[mugla_temp_cols].mean(axis=1)
df["Denizli_Temp"] = df[denizli_temp_cols].mean(axis=1)
df["Aydin_Temp"] = df[aydin_temp_cols].mean(axis=1)

df["Mugla_Rain"] = df[[c for c in df.columns if c.startswith("Yağış-MUGLA")]].mean(axis=1)
df["Denizli_Rain"] = df[[c for c in df.columns if c.startswith("Yağış-DNZ")]].mean(axis=1)
df["Aydin_Rain"] = df[[c for c in df.columns if c.startswith("Yağış-AYD")]].mean(axis=1)

df["Mugla_Cloud"] = df[[c for c in df.columns if c.startswith("Bulutluluk-MUGLA")]].mean(axis=1)
df["Denizli_Cloud"] = df[[c for c in df.columns if c.startswith("Bulutluluk-DNZ")]].mean(axis=1)
df["Aydin_Cloud"] = df[[c for c in df.columns if c.startswith("Bulutluluk-AYD")]].mean(axis=1)

df["Mugla_DarkFraction"] = df[[c for c in df.columns if "MUGLA" in c and "Dark_Fraction" in c]].mean(axis=1)
df["Denizli_DarkFraction"] = df[[c for c in df.columns if "DNZ" in c and "Dark_Fraction" in c]].mean(axis=1)
df["Aydin_DarkFraction"] = df[[c for c in df.columns if "AYD" in c and "Dark_Fraction" in c]].mean(axis=1)

# 3. Zaman
df["hour"] = df.index.hour
df["month"] = df.index.month
df["day_of_year"] = df.index.dayofyear
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
df["is_weekend"] = df["Haftanın_Günü"].isin([6,7]).astype(int)
df["time_index"] = np.arange(len(df))

# ============================================================
# FİNAL FEATURE MÜHENDİSLİĞİ
# ============================================================
target_col = "ADM_Dağıtılan_Enerji_(MWh)"

df["ADM_Rolling24h_mean"] = df[target_col].rolling(24, min_periods=24).mean().shift(1)
df["ADM_Rolling168h_mean"] = df[target_col].rolling(168, min_periods=168).mean().shift(1)
df["ADM_Rolling24h_std"] = df[target_col].rolling(24, min_periods=24).std().shift(1)

df["Lag24"] = df[target_col].shift(24)
df["Lag48"] = df[target_col].shift(48)
df["Lag72"] = df[target_col].shift(72)
df["Lag168"] = df[target_col].shift(168)

df["Short_Trend"] = df["Lag24"] - df["Lag48"]

roll24 = df[target_col].rolling(24, min_periods=24).mean().shift(1)
roll168 = df[target_col].rolling(168, min_periods=168).mean().shift(1)
df["Trend_Ratio"] = roll24 / roll168

df["Region_Temp"] = (df["Mugla_Temp"] + df["Aydin_Temp"] + df["Denizli_Temp"]) / 3
df["Temp_Deviation"] = abs(df["Region_Temp"] - 18)
df["Temp_Rolling24"] = df["Region_Temp"].rolling(24).mean()

df["day_of_week"] = df.index.dayofweek + 1

df = df.iloc[504:].copy()

feature_cols = [
    "Lag24", "Lag48", "Lag72", "Lag168",
    "Short_Trend", "Trend_Ratio",
    "Temp_Deviation", "Temp_Rolling24",
    "day_of_week",
    "hour_sin", "hour_cos",
    "is_holiday", "is_weekend"
]

print(f"The total number of features: {len(feature_cols)}")

# ============================================================
# MODEL VERİ SETİ
# ============================================================
X = df[feature_cols]
y = df[target_col]


# ============================================================
# FINAL TRAIN & TEST SPLIT
# ============================================================
X_train_final = X.loc[:"2024-12-31 23:00:00"]
y_train_final = y.loc[:"2024-12-31 23:00:00"]
X_test_final  = X.loc["2025-01-01":"2025-08-31"]
y_test_final  = y.loc["2025-01-01":"2025-08-31"]

# ============================================================
# MODEL PARAMETRELERİ
# ============================================================
model_params = {
    "n_estimators": 4000,
    "learning_rate": 0.015,
    "max_depth": 4,
    "min_child_weight": 12,
    "gamma": 0.3,
    "colsample_bytree": 0.5,
    "subsample": 0.55,
    "reg_lambda": 15,
    "reg_alpha": 1,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse"
}

print("\n=== FİNAL MODEL EDUCATION (EARLY STOPPING) ===")

final_model = xgb.XGBRegressor(**model_params)

final_model.fit(
    X_train_final, 
    y_train_final,
    eval_set=[(X_train_final, y_train_final), (X_test_final, y_test_final)],
    verbose=True
)

print(f"\nBest Iteration: {final_model.best_iteration}")
print(f"Best Score: {final_model.best_score}")
# ============================================================
# TIME SERIES CROSS VALIDATION
# ============================================================
# ============================================================
# ADIM 1: CROSS VALIDATION (GÜVEN TESTİ)
# ============================================================
print("\n=== CROSS VALIDATION IS STARTING (5 FOLD) ===")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
fold = 1

for train_index, val_index in tscv.split(X_train_final):
    X_train_fold, X_val_fold = X_train_final.iloc[train_index], X_train_final.iloc[val_index]
    y_train_fold, y_val_fold = y_train_final.iloc[train_index], y_train_final.iloc[val_index]
    
    cv_model = xgb.XGBRegressor(**model_params)
    
    # CV yaparken verbose=False yapıyoruz ki ekran dolmasın
    cv_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
        verbose=False 
    )
    
    preds = cv_model.predict(X_val_fold, iteration_range=(0, cv_model.best_iteration + 1))
    mape = np.mean(np.abs((y_val_fold - preds) / y_val_fold)) * 100
    cv_scores.append(mape)
    
    print(f"Fold {fold} -> MAPE: {mape:.3f}% (Best Iter: {cv_model.best_iteration})")
    fold += 1

print(f"AVERAGE CV MAPE: {np.mean(cv_scores):.3f}%")
print("-" * 50)


# ============================================================
# TAHMİN & METRİKLER
# ============================================================
y_train_pred = final_model.predict(X_train_final, iteration_range=(0, final_model.best_iteration + 1))
y_test_pred = final_model.predict(X_test_final, iteration_range=(0, final_model.best_iteration + 1))

def print_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{label} Results:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  MAPE: {mape:.3f}%")
    return rmse

train_rmse = print_metrics(y_train_final, y_train_pred, "TRAIN (2018-2024)")
test_rmse = print_metrics(y_test_final, y_test_pred, "TEST (2025)")

print(f"\nGAP : {test_rmse - train_rmse:.3f}")

importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\nThe top 10 features:")
print(importances.head(10))

# ============================================================
# 1) GERÇEK VS TAHMİN GRAFİĞİ
# ============================================================
plt.figure(figsize=(16,6))
plt.plot(y_test_final.index, y_test_final.values, label="Real", linewidth=2)
plt.plot(y_test_final.index, y_test_pred, label="Forecast", linewidth=2)
plt.title("Real vs Forecast (Test Era)")
plt.xlabel("Date")
plt.ylabel("MWh")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 4) SCATTER PLOT
# ============================================================
plt.figure(figsize=(6,6))
plt.scatter(y_test_final, y_test_pred, alpha=0.4)
plt.plot(
    [y_test_final.min(), y_test_final.max()],
    [y_test_final.min(), y_test_final.max()],
    color="red", linewidth=2
)
plt.title("Real vs Forecast Scatter Plot")
plt.xlabel("Real")
plt.ylabel("Forecast")
plt.grid()
plt.show()

# ============================================================
# 5) FEATURE IMPORTANCE PLOT
# ============================================================
plt.figure(figsize=(10,6))
importances.head(10).plot(kind="bar", color="teal")
plt.title("The top 10 features")
plt.ylabel("Importance Score")
plt.grid(axis="y")
plt.show()

# ============================================================
# 6) AYLIK MAPE ANALİZİ
# ============================================================
error_df = pd.DataFrame({
    "real": y_test_final,
    "pred": y_test_pred
})
error_df["mape"] = np.abs((error_df["real"] - error_df["pred"]) / error_df["real"]) * 100
error_df["month"] = error_df.index.month

monthly_mape = error_df.groupby("month")["mape"].mean()

plt.figure(figsize=(10,4))
monthly_mape.plot(kind="bar", color="green")
plt.title("Average Monthly MAPE (Test Set)")
plt.ylabel("MAPE (%)")
plt.grid(axis="y")
plt.show()
