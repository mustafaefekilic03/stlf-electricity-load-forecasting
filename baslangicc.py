import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

print("--- OBLIQUE RANDOM FOREST (XGB FE STYLE) V2.0 (REGULARIZED + FAST PI) ---")

# ============================================================
# 1) DOSYAYI OKU
# ============================================================
file_path = r"C:\Users\NEW\Desktop\esasveribaslangicc\INPUT_TED_Bitirme_STLF.xlsx"
df = pd.read_excel(file_path)
df.columns = [c.strip() for c in df.columns]
print("✅ Excel okundu:", df.shape)

# ============================================================
# 2) TARİH INDEX
# ============================================================
df["Tarih"] = pd.to_datetime(df["Tarih"])
df = df.set_index("Tarih").sort_index()

target_col = "ADM_Dağıtılan_Enerji_(MWh)"

# ============================================================
# 3) HOLIDAY FEATURES (XGB İLE AYNI)
# ============================================================
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
df["holiday_week"]  = df["is_holiday"].rolling(24*7, min_periods=1).max().astype(int)

# ============================================================
# 4) METEOROLOJİ (XGB İLE AYNI)
# ============================================================
mugla_temp_cols   = [c for c in df.columns if "Hissedilen_Sıcaklık-MUGLA" in c]
denizli_temp_cols = [c for c in df.columns if "Hissedilen_Sıcaklık-DNZ" in c]
aydin_temp_cols   = [c for c in df.columns if "Hissedilen_Sıcaklık-AYD" in c]

df["Mugla_Temp"]   = df[mugla_temp_cols].mean(axis=1)   if mugla_temp_cols   else 0
df["Denizli_Temp"] = df[denizli_temp_cols].mean(axis=1) if denizli_temp_cols else 0
df["Aydin_Temp"]   = df[aydin_temp_cols].mean(axis=1)   if aydin_temp_cols   else 0

# ============================================================
# 5) ZAMAN FEATURE'LARI
# ============================================================
df["hour"] = df.index.hour
df["month"] = df.index.month
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

df["is_weekend"] = df["Haftanın_Günü"].isin([6,7]).astype(int)  # XGB ile aynı
df["day_of_week"] = df.index.dayofweek + 1  # 1..7

# ============================================================
# 6) FİNAL FEATURE ENGINEERING (XGB İLE AYNI + LEAKAGE YOK)
# ============================================================

# --- Rolling (leakage yok: shift(1)) ---
df["ADM_Rolling24h_mean"]  = df[target_col].rolling(24,  min_periods=24).mean().shift(1)
df["ADM_Rolling168h_mean"] = df[target_col].rolling(168, min_periods=168).mean().shift(1)
df["ADM_Rolling24h_std"]   = df[target_col].rolling(24,  min_periods=24).std().shift(1)
df["ADM_Rolling168h_std"]  = df[target_col].rolling(168, min_periods=168).std().shift(1)

# --- Lags (önce temel lag'ler) ---
df["Lag24"]  = df[target_col].shift(24)
df["Lag48"]  = df[target_col].shift(48)
df["Lag72"]  = df[target_col].shift(72)
df["Lag168"] = df[target_col].shift(168)
df["Lag336"] = df[target_col].shift(336)

# --- Türetilmiş trend feature’ları ---
df["Weekly_Trend"] = df["Lag168"] - df["Lag336"]
df["Short_Trend"]  = df["Lag24"] - df["Lag48"]

roll24  = df[target_col].rolling(24,  min_periods=24).mean().shift(1)
roll168 = df[target_col].rolling(168, min_periods=168).mean().shift(1)
df["Trend_Ratio"] = roll24 / roll168

# --- Sıcaklık feature'ları (önce deviation, sonra interaction) ---
df["Region_Temp"] = (df["Mugla_Temp"] + df["Aydin_Temp"] + df["Denizli_Temp"]) / 3
df["Temp_Deviation"] = np.abs(df["Region_Temp"] - 18)
df["Temp_Rolling24"] = df["Region_Temp"].rolling(24, min_periods=24).mean().shift(1)

# hour_cos daha önce tanımlı olmalı (zaman features kısmında)
df["Temp_Interaction"] = df["Temp_Deviation"] * df["hour_cos"]

# --- 504 kes (XGB’deki gibi) ---
df = df.iloc[504:].copy()

# ============================================================
# 7) FEATURE SET (Genişletildi)
# ============================================================
feature_cols = [
    "Lag24", "Lag48", "Lag72", "Lag168",
    "Lag336", "Weekly_Trend",
    "Short_Trend", "Trend_Ratio",
    "ADM_Rolling24h_std", "ADM_Rolling168h_std",
    "Temp_Deviation", "Temp_Rolling24", "Temp_Interaction",
    "day_of_week",
    "hour_sin", "hour_cos",
    "is_holiday", "is_weekend",
    "holiday_prev1", "holiday_next1", "holiday_week",
    "month_sin", "month_cos"
]

print(f"✅ Feature count: {len(feature_cols)}")

# MODEL DATA
X = df[feature_cols].copy()
y = df[target_col].copy()

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

# ============================================================
# 8) TRAIN / TEST SPLIT (XGB ile aynı mantık)
# ============================================================
X_train = X.loc[:"2024-12-31 23:00:00"]
y_train = y.loc[:"2024-12-31 23:00:00"]
X_test  = X.loc["2025-01-01":"2025-08-31"]
y_test  = y.loc["2025-01-01":"2025-08-31"]

print("Train:", X_train.shape, "Test:", X_test.shape)

# ============================================================
# 9) MODEL (Regularized RF) + DOĞRU LOG TARGET
# ============================================================
rf = RandomForestRegressor(
    n_estimators=900,
    max_depth=18,
    min_samples_split=20,
    min_samples_leaf=8,
    max_features=0.6,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", rf)
])

model = TransformedTargetRegressor(
    regressor=pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

print("\n=== TRAINING OBLIQUE RF (REGULARIZED) ===")
model.fit(X_train, y_train)

# OOB score (fit sonrası erişilebilir)
print("OOB Score:", model.regressor_.named_steps["rf"].oob_score_)

# ============================================================
# 10) METRİKLER
# ============================================================
def calc_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100
    print(f"\n{label} :")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  MAPE: {mape:.3f}%")
    return rmse, mae, mape

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_rmse, train_mae, train_mape = calc_metrics(y_train, y_train_pred, "TRAIN (2018-2024)")
test_rmse,  test_mae,  test_mape  = calc_metrics(y_test,  y_test_pred,  "TEST (2025)")

print(f"\nGAP (RMSE): {test_rmse - train_rmse:.3f}")

# ============================================================
# 11) REAL vs FORECAST
# ============================================================
plt.figure(figsize=(16,6))
plt.plot(y_test.index, y_test.values, label="Real", linewidth=2)
plt.plot(y_test.index, y_test_pred, label="Forecast (RF)", linewidth=2)
plt.title("Real vs Forecast (Test Era) - Regularized RF")
plt.xlabel("Date")
plt.ylabel("MWh")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 12) FAST PERMUTATION IMPORTANCE + YÜZDESEL TOP10
# ============================================================
print("\n📊 Permutation importance (FAST)...")

# hız için sample
sample_n = 1500 if len(X_test) > 1500 else len(X_test)
X_pi = X_test.sample(n=sample_n, random_state=42)
y_pi = y_test.loc[X_pi.index]

perm = permutation_importance(
    model, X_pi, y_pi,
    n_repeats=2,
    random_state=42,
    n_jobs=-1,
    scoring="neg_mean_absolute_error"
)

perm_imp = pd.Series(perm.importances_mean, index=X_pi.columns).sort_values(ascending=False)
top10_raw = perm_imp.head(10)

# yüzdesel normalize (top10 toplamı %100)
den = top10_raw.abs().sum()
top10_pct = (top10_raw.abs() / den) * 100
top10_pct = top10_pct.sort_values(ascending=False)

print("\nTop 10 features (Permutation Importance %) :")
print(top10_pct.round(2).astype(str) + "%")

plt.figure(figsize=(10,6))
top10_pct.sort_values().plot(kind="barh", color="teal")
plt.title("Top 10 Feature Importance (%) - Permutation")
plt.xlabel("Importance سهم (%)")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ DONE.")
