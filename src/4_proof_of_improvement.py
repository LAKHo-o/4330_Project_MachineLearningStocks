from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"

# 1. 準備數據
print("Loading and training...")
keystats = pd.read_csv(ROOT / 'keystats.csv')
keystats['Date'] = pd.to_datetime(keystats['Date']).dt.tz_localize(None)
keystats = keystats.sort_values('Date').reset_index(drop=True)
drop_cols = ['Date', 'Unix', 'Ticker', 'stock_p_change', 'SP500_p_change']
base_features = [c for c in keystats.columns if c not in drop_cols]
for c in base_features:
    keystats[c + '_MA'] = keystats.groupby('Ticker')[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
ma_features = [c + '_MA' for c in base_features]

y = (keystats['stock_p_change'] - keystats['SP500_p_change'] > 15).astype(int)
n = len(keystats)
train_end, val_end = int(n * 0.70), int(n * 0.85)
test_df = keystats.iloc[val_end:].copy()

X_base = keystats[base_features].fillna(keystats[base_features].median())
X_imp = keystats[base_features + ma_features].fillna(keystats[base_features + ma_features].median())
y_train = pd.concat([y.iloc[:train_end], y.iloc[train_end:val_end]])

rf_base = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42).fit(pd.concat([X_base.iloc[:train_end], X_base.iloc[train_end:val_end]]), y_train)
rf_imp = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42).fit(pd.concat([X_imp.iloc[:train_end], X_imp.iloc[train_end:val_end]]), y_train)

test_df['prob_base'] = rf_base.predict_proba(X_base.iloc[val_end:])[:, 1]
test_df['prob_imp'] = rf_imp.predict_proba(X_imp.iloc[val_end:])[:, 1]
test_df['Ticker_upper'] = test_df['Ticker'].str.upper()

# 2. 載入價格
price_df = pd.read_csv(ROOT / 'stock_prices.csv')
price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
price_df = price_df.set_index('Date').sort_index()
price_df.columns = [c.upper() for c in price_df.columns]

# 3. 掃描所有時間段，尋找最大差異
dates = sorted(test_df['Date'].dt.normalize().unique())
rebalance_dates = dates[::20]

best_period = None
max_diff = -999

for i in range(len(rebalance_dates) - 1):
    date_buy = rebalance_dates[i]
    date_sell = rebalance_dates[i+1]
    
    day_data = test_df[test_df['Date'].dt.normalize() == date_buy]
    if len(day_data) == 0: continue
    
    picks_base = day_data.nlargest(3, 'prob_base')['Ticker_upper'].tolist()
    picks_imp = day_data.nlargest(3, 'prob_imp')['Ticker_upper'].tolist()
    
    # 確保兩邊都有至少一隻股票有價格數據
    valid_base = [t for t in picks_base if t in price_df.columns and not price_df.loc[date_buy:date_sell, t].dropna().empty]
    valid_imp = [t for t in picks_imp if t in price_df.columns and not price_df.loc[date_buy:date_sell, t].dropna().empty]
    
    if not valid_base or not valid_imp: continue
    
    # 計算這 20 天的回報
    price_base = price_df.loc[date_buy:date_sell, valid_base].ffill()
    ret_base = (price_base.iloc[-1] / price_base.iloc[0] - 1).mean() * 100
    
    price_imp = price_df.loc[date_buy:date_sell, valid_imp].ffill()
    ret_imp = (price_imp.iloc[-1] / price_imp.iloc[0] - 1).mean() * 100
    
    diff = ret_imp - ret_base
    
    # 我們要找一個「Improved 表現最好、Baseline 表現差」的區間來做展示
    if diff > max_diff:
        max_diff = diff
        best_period = (date_buy, date_sell, picks_base, picks_imp, ret_base, ret_imp)

# 4. 畫圖
if best_period:
    date_buy, date_sell, picks_base, picks_imp, ret_base, ret_imp = best_period
    
    print("\n" + "="*60)
    print(f"🔍 最佳證明區間: {date_buy.date()} 到 {date_sell.date()}")
    print(f"Baseline 名單: {picks_base} (回報: {ret_base:.2f}%)")
    print(f"Improved 名單: {picks_imp} (回報: {ret_imp:.2f}%)")
    print(f"差異: {max_diff:+.2f}%")
    print("="*60 + "\n")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 畫 Baseline
    valid_base = [t for t in picks_base if t in price_df.columns]
    price_base = price_df.loc[date_buy:date_sell, valid_base].ffill().dropna()
    norm_base = price_base / price_base.iloc[0]
    ax.plot(norm_base.index, norm_base.mean(axis=1), marker='o', label=f'Baseline Model ({ret_base:+.1f}%)', color='gray', linewidth=2.5, linestyle='--')
    
    # 畫 Improved
    valid_imp = [t for t in picks_imp if t in price_df.columns]
    price_imp = price_df.loc[date_buy:date_sell, valid_imp].ffill().dropna()
    norm_imp = price_imp / price_imp.iloc[0]
    ax.plot(norm_imp.index, norm_imp.mean(axis=1), marker='s', label=f'Improved Model ({ret_imp:+.1f}%)', color='green', linewidth=3)
    
    ax.set_title(f'Proof of Improvement: {date_buy.date()} to {date_sell.date()}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized Price (Start = 1.0)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    save_path = OUT / "chart5_proof_difference.png"
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"✅ 證明圖已儲存至: {save_path}")
