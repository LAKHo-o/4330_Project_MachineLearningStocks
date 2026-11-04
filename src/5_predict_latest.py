from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# 1. 載入 2025-2026 最新數據
# ============================================================
print("Loading latest extended data...")
df = pd.read_csv(ROOT / 'keystats_extended.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
df = df.sort_values('Date').reset_index(drop=True)

# ============================================================
# 2. 特徵工程 (加入 MA6)
# ============================================================
drop_cols = ['Date', 'Unix', 'Ticker', 'stock_p_change', 'SP500_p_change']
base_features = [c for c in df.columns if c not in drop_cols]

for c in base_features:
    df[c + '_MA'] = df.groupby('Ticker')[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
all_features = base_features + [c + '_MA' for c in base_features]

# ============================================================
# 3. 切分訓練集與測試集 (用 2025 年訓練，預測 2026 年最新數據)
# ============================================================
# 假設 2025 年是訓練集，2026 年是測試集
train_mask = df['Date'] < '2026-01-01'
test_mask = df['Date'] >= '2026-01-01'

X_train = df.loc[train_mask, all_features].fillna(df[all_features].median())
y_train = (df.loc[train_mask, 'stock_p_change'] - df.loc[train_mask, 'SP500_p_change'] > 15).astype(int)

# 如果 2026 年沒有數據，就退而求其次用最後一個月
if test_mask.sum() == 0:
    print("No 2026 data found. Using the last available quarter.")
    latest_date = df['Date'].max()
    test_mask = df['Date'] == latest_date
    X_test = df.loc[test_mask, all_features].fillna(df[all_features].median())
else:
    X_test = df.loc[test_mask, all_features].fillna(df[all_features].median())

test_df = df.loc[test_mask].copy()

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ============================================================
# 4. 訓練改良模型 (MA6 + OOB 隨機森林)
# ============================================================
print("\nTraining the Improved Machine Learning Model...")
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=10, 
    class_weight='balanced',
    oob_score=True,
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"✅ Training complete. OOB Score: {rf.oob_score_:.4f}")

# ============================================================
# 5. 實時預測與輸出
# ============================================================
test_df['Prob_Outperform'] = rf.predict_proba(X_test)[:, 1]

# 找出最新一天（或最新一期）
latest_date = test_df['Date'].max()
latest_predictions = test_df[test_df['Date'] == latest_date].copy()

# 按概率排序，選出 Top 5 推薦股票
latest_predictions = latest_predictions.sort_values('Prob_Outperform', ascending=False)
top_picks = latest_predictions.head(5)

print("\n" + "="*60)
print(f"📊 REAL-TIME STOCK PREDICTION (Date: {latest_date.date()})")
print("="*60)
print("Based on the Improved MA6 + OOB Random Forest Model")
print("-" * 80)
print(f"{'Rank':<6} | {'Ticker':<8} | {'Probability':<12} | {'Prediction'}")
print("-" * 80)

for rank, (_, row) in enumerate(top_picks.iterrows(), 1):
    ticker = str(row['Ticker']).upper()
    prob = row['Prob_Outperform']
    prediction = "OUTPERFORM 🚀" if prob > 0.5 else "UNDERPERFORM 📉"
    print(f"{rank:<6} | {ticker:<8} | {prob*100:.2f}%{'':<6} | {prediction}")

print("="*60)
print("\n* Disclaimer: This is a demonstration of the system's real-time")
print("  prediction capability. It is for educational purposes only.")

# 儲存預測結果
output_path = ROOT / "reports" / "latest_predictions.csv"
output_path.parent.mkdir(exist_ok=True)
latest_predictions[['Date', 'Ticker', 'Prob_Outperform']].to_csv(output_path, index=False)
print(f"\n✅ Full prediction list saved to: {output_path}")
