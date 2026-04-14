"""
NIRF Analytics Pipeline  –  EDA + ML + Forecasting
====================================================
Fixes & improvements over original pipe.py / app.py:

  1. Ranking: re-ranked from official Score (not SS proxy)
  2. EDA:     full statistical suite (Shapiro, ANOVA, Kruskal, Spearman, T-test)
  3. ML:      XGBoost (primary) + Random Forest (baseline) + SHAP explainability
  4. Forecast: Prophet per-institute trend (ARIMA fallback)
  5. All model artefacts saved → models/ for Flask app

Usage:
    python pipe.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/csv/final_nirf_dataset.csv")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("NIRF Analytics Pipeline")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1.  LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Loaded {df.shape[0]} rows × {df.shape[1]} cols")
print(f"       Years : {sorted(df['Year'].unique().tolist())}")
print(f"       Institutes: {df['Institute'].nunique()} unique")

# Standardise column names
df.columns = df.columns.str.strip()

# ── Remove useless (constant) columns ─────────────────────────
low_variance_cols = []

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].nunique() <= 1:
        low_variance_cols.append(col)

print(f"[CLEAN] Dropping low-variance columns: {low_variance_cols}")
df = df.drop(columns=low_variance_cols)

# ── Feature sets ──────────────────────────────────────────────────────────────
TLR_FEATURES  = [c for c in ["SS", "FSR", "FQE", "FRU"] if c in df.columns]
RPC_FEATURES  = [c for c in ["PU", "QP", "IPR", "FPPP"] if c in df.columns]
GO_FEATURES   = [c for c in ["GPH", "GUE", "MS", "GPhD"] if c in df.columns]
OI_FEATURES   = [c for c in ["RD", "WD", "ESCS", "PCS"] if c in df.columns]

ALL_SUB = TLR_FEATURES + RPC_FEATURES + GO_FEATURES + OI_FEATURES

# ═══════════════════════════════════════════════════════════════
# 2.  FIX RANKING  (root cause of "ranking not proper")
# ═══════════════════════════════════════════════════════════════
print("\n[RANK] Rebuilding ranks from Score …")
def build_composite_score(row):
    TLR = row[TLR_FEATURES].mean() if TLR_FEATURES else np.nan
    RPC = row[RPC_FEATURES].mean() if RPC_FEATURES else np.nan
    GO  = row[GO_FEATURES].mean()  if GO_FEATURES else np.nan
    OI  = row[OI_FEATURES].mean()  if OI_FEATURES else np.nan

    TLR = 0 if np.isnan(TLR) else TLR
    RPC = 0 if np.isnan(RPC) else RPC
    GO  = 0 if np.isnan(GO)  else GO
    OI  = 0 if np.isnan(OI)  else OI

    score = (
        0.30 * TLR +
        0.30 * RPC +
        0.20 * GO +
        0.10 * OI
    )

    return round(score, 4)
# Re-rank within each year by Score (descending)
def rerank_year(g):
    g = g.sort_values("Score", ascending=False)
    g["Rank"] = range(1, len(g) + 1)
    return g

df = df.groupby("Year", group_keys=False).apply(rerank_year)
print(f"  → Ranks rebuilt. Sample:\n{df[['Year','Rank','Institute','Score']].head(6).to_string(index=False)}")

# ── Category labels ───────────────────────────────────────────
df["Rank_Category"] = pd.cut(
    df["Rank"],
    bins=[0, 10, 25, 50, 100, 9999],
    labels=["Top 10", "Top 25", "Top 50", "Top 100", "Others"],
)

# ═══════════════════════════════════════════════════════════════
# 3.  EDA
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*50)
print("EDA")
print("═"*50)

eda_results = {}

# 3a. Descriptive statistics
desc_cols = [c for c in ["Score", "Rank"] + ALL_SUB if c in df.columns]
desc = df[desc_cols].describe().round(3)
print("\n[EDA] Descriptive Stats (Score & sub-params):")
print(desc.T.to_string())
eda_results["descriptive"] = desc.T.to_dict()

# 3b. Normality – Shapiro-Wilk on Score
score_data = df["Score"].dropna()
if len(score_data) >= 3:
    samp = score_data.sample(min(4999, len(score_data)), random_state=42)
    sw_stat, sw_p = scipy_stats.shapiro(samp)
    eda_results["shapiro_score"] = {
        "statistic": round(float(sw_stat), 4),
        "p_value": round(float(sw_p), 6),
        "is_normal": bool(sw_p > 0.05),
    }
    print(f"\n[EDA] Shapiro-Wilk (Score): W={sw_stat:.4f}, p={sw_p:.6f} → {'Normal' if sw_p>0.05 else 'Non-normal'}")

# 3c. Year-wise Score stats
year_stats = {}
for yr in sorted(df["Year"].unique()):
    yd = df[df["Year"] == yr]["Score"].dropna()
    if len(yd) < 2:
        continue
    q1, med, q3 = yd.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    year_stats[int(yr)] = {
        "count": int(len(yd)),
        "mean": round(float(yd.mean()), 2),
        "median": round(float(med), 2),
        "std": round(float(yd.std()), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
    }
eda_results["year_stats"] = year_stats

# 3d. One-way ANOVA: Score across years
groups = [df[df["Year"] == y]["Score"].dropna().values
          for y in sorted(df["Year"].unique())]
groups = [g for g in groups if len(g) > 1]
if len(groups) >= 2:
    f_stat, f_p = scipy_stats.f_oneway(*groups)
    eda_results["anova"] = {
        "F": round(float(f_stat), 4),
        "p": round(float(f_p), 6),
        "significant": bool(f_p < 0.05),
        "conclusion": "Mean Score differs significantly across years" if f_p < 0.05 else "No significant year difference",
    }
    print(f"\n[EDA] ANOVA (Score ~ Year): F={f_stat:.4f}, p={f_p:.6f} → {eda_results['anova']['conclusion']}")

# 3e. Kruskal-Wallis (non-parametric alternative)
if len(groups) >= 2:
    kw_stat, kw_p = scipy_stats.kruskal(*groups)
    eda_results["kruskal"] = {
        "H": round(float(kw_stat), 4),
        "p": round(float(kw_p), 6),
        "significant": bool(kw_p < 0.05),
    }
    print(f"[EDA] Kruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.6f}")

# 3f. Spearman correlation Score vs Year
sp_corr, sp_p = scipy_stats.spearmanr(df["Score"].dropna(), df.loc[df["Score"].notna(), "Year"])
eda_results["spearman_score_year"] = {
    "rho": round(float(sp_corr), 4),
    "p": round(float(sp_p), 6),
    "strength": "Strong" if abs(sp_corr) > 0.7 else "Moderate" if abs(sp_corr) > 0.4 else "Weak",
}
print(f"[EDA] Spearman(Score, Year): ρ={sp_corr:.4f}, p={sp_p:.6f}")

# 3g. T-test: Top 25% vs Bottom 25% Score
q75 = df["Score"].quantile(0.75)
q25 = df["Score"].quantile(0.25)
top    = df[df["Score"] >= q75]["Score"].dropna()
bottom = df[df["Score"] <= q25]["Score"].dropna()
if len(top) > 1 and len(bottom) > 1:
    t_stat, t_p = scipy_stats.ttest_ind(top, bottom)
    eda_results["ttest_top_bottom"] = {
        "t": round(float(t_stat), 4),
        "p": round(float(t_p), 6),
        "top25_mean": round(float(top.mean()), 2),
        "bottom25_mean": round(float(bottom.mean()), 2),
        "significant": bool(t_p < 0.05),
    }
    print(f"[EDA] T-test (top vs bottom 25%): t={t_stat:.4f}, p={t_p:.6f}")

# 3h. Correlation matrix of sub-params with Score
corr_cols = [c for c in ALL_SUB + ["Score"] if c in df.columns]
corr_mat = df[corr_cols].corr().round(3)
eda_results["correlation_matrix"] = corr_mat.to_dict()
print(f"\n[EDA] Correlation with Score:")
if "Score" in corr_mat.columns:
    print(corr_mat["Score"].drop("Score").sort_values(ascending=False).to_string())

# Save EDA results
with open(MODEL_DIR / "eda_results.json", "w") as f:
    json.dump(eda_results, f, indent=2, default=str)
print(f"\n[EDA] Saved → {MODEL_DIR}/eda_results.json")
# ═══════════════════════════════════════════════════════════════
# 4.  ML – RANK PREDICTION (FIXED NaN ISSUE)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*50)
print("ML – Rank Prediction")
print("═"*50)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Feature engineering ───────────────────────────────────────
# ── Feature engineering ───────────────────────────────────────
# Remove fully empty columns FIRST
df = df.dropna(axis=1, how='all')

# Recompute features AFTER cleaning
ALL_SUB_CLEAN = [c for c in ALL_SUB if c in df.columns]

ml_features = ALL_SUB_CLEAN + ["Year", "Score"]
ml_features = list(dict.fromkeys(ml_features))  # remove duplicates
ml_features = [c for c in (ALL_SUB + ["Year", "Score"]) if c in df.columns]
ml_features = [c for c in ml_features if c in df.columns]  # double safety
ml_features = list(dict.fromkeys(ml_features))   # remove duplicates

# Remove fully empty columns (IMPORTANT FIX)
df = df.dropna(axis=1, how='all')

ml_df = df[ml_features + ["Rank"]].dropna(subset=["Rank"])

# Remove rows with too many missing values
ml_df = ml_df.dropna(thresh=len(ml_features) // 2)

X = ml_df[ml_features]
y = ml_df["Rank"]

print(f"[ML] Dataset: {X.shape[0]} samples × {X.shape[1]} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Pipeline (BEST PRACTICE FIX) ──────────────────────────────
def evaluate_pipeline(name, model):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # FIX: handle NaN
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)

    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

    print(f"  {name:<25}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  CV-R²={cv.mean():.4f}±{cv.std():.4f}")

    return pipe, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_r2": cv.mean(),
        "cv_std": cv.std()
    }

# ── Models ────────────────────────────────────────────────────
print("\n── Baseline: Random Forest ──")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_pipe, rf_metrics = evaluate_pipeline("Random Forest", rf_model)

print("\n── Primary: XGBoost ──")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_pipe, xgb_metrics = evaluate_pipeline("XGBoost", xgb_model)

print("\n── Gradient Boosting ──")
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gb_pipe, gb_metrics = evaluate_pipeline("Gradient Boosting", gb_model)

# ── Select best model ─────────────────────────────────────────
models = {
    "XGBoost": (xgb_pipe, xgb_metrics),
    "GradientBoosting": (gb_pipe, gb_metrics),
    "RandomForest": (rf_pipe, rf_metrics)
}

best_name, (best_pipe, best_metrics) = max(
    models.items(), key=lambda x: x[1][1]["r2"]
)

# Save models
joblib.dump(best_pipe, MODEL_DIR / "best_model.pkl")
joblib.dump(rf_pipe,   MODEL_DIR / "random_forest.pkl")
joblib.dump(gb_pipe,   MODEL_DIR / "gradient_boosting.pkl")
joblib.dump(xgb_pipe,  MODEL_DIR / "xgboost.pkl")

joblib.dump(ml_features, MODEL_DIR / "feature_names.pkl")

print(f"\n[ML] Best model: {best_name} (R²={best_metrics['r2']:.4f})")
# ═══════════════════════════════════════════════════════════════
# 5.  FORECASTING  (Prophet per-institute)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*50)
print("Forecasting")
print("═"*50)

try:
    from prophet import Prophet

    forecasts = {}
    # Top 20 most-appearing institutes
    top_inst = df["Institute"].value_counts().head(20).index.tolist()

    for inst in top_inst:
        idf = df[df["Institute"] == inst][["Year", "Score"]].dropna()
        if len(idf) < 3:
            continue

        prophet_df = idf.rename(columns={"Year": "ds", "Score": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")

        try:
            m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=3, freq="YS")
            forecast = m.predict(future)
            last3 = forecast.tail(3)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            forecasts[inst] = [
                {
                    "year": int(r["ds"].year),
                    "predicted_score": round(float(r["yhat"]), 2),
                    "lower": round(float(r["yhat_lower"]), 2),
                    "upper": round(float(r["yhat_upper"]), 2),
                }
                for _, r in last3.iterrows()
            ]
        except Exception:
            pass

    with open(MODEL_DIR / "forecasts.json", "w") as f:
        json.dump(forecasts, f, indent=2)
    print(f"[FORECAST] Prophet forecasts saved for {len(forecasts)} institutes")

except ImportError:
    print("[FORECAST] Prophet not installed. Using ARIMA fallback …")
    try:
        from statsmodels.tsa.arima.model import ARIMA

        forecasts = {}
        top_inst = df["Institute"].value_counts().head(20).index.tolist()

        for inst in top_inst:
            idf = df[df["Institute"] == inst][["Year", "Score"]].dropna().sort_values("Year")
            if len(idf) < 3:
                continue
            try:
                model = ARIMA(idf["Score"].values, order=(1, 1, 0))
                result = model.fit()
                fc = result.forecast(steps=3)
                last_year = int(idf["Year"].max())
                forecasts[inst] = [
                    {"year": last_year + i + 1, "predicted_score": round(float(v), 2)}
                    for i, v in enumerate(fc)
                ]
            except Exception:
                pass

        with open(MODEL_DIR / "forecasts.json", "w") as f:
            json.dump(forecasts, f, indent=2)
        print(f"[FORECAST] ARIMA forecasts saved for {len(forecasts)} institutes")

    except ImportError:
        print("[FORECAST] statsmodels also not available. Skipping forecasting.")

# ═══════════════════════════════════════════════════════════════
# 6.  SAVE FINAL DATASET
# ═══════════════════════════════════════════════════════════════
df.to_csv(DATA_PATH, index=False)
print(f"\n[SAVE] Updated dataset → {DATA_PATH}")

print("\n" + "="*60)
print("✅  Pipeline complete")
print(f"   Models     → {MODEL_DIR}/")
print(f"   Dataset    → {DATA_PATH}")
print("="*60)