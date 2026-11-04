from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

INITIAL_CAPITAL = 100_000
HOLD_DAYS = 20
TOP_N = 3
COST = 0.0005

# ============================================================
# 1. 載入數據與定義特徵
# ============================================================
print("Loading data...")
keystats = pd.read_csv(ROOT / 'keystats.csv')
keystats['Date'] = pd.to_datetime(keystats['Date']).dt.tz_localize(None)
keystats = keystats.sort_values('Date').reset_index(drop=True)

sp500 = pd.read_csv(ROOT / 'sp500_index.csv')
sp500['Date'] = pd.to_datetime(sp500['Date']).dt.tz_localize(None)
sp500 = sp500.set_index('Date').sort_index()

drop_cols = ['Date', 'Unix', 'Ticker', 'stock_p_change', 'SP500_p_change']
base_features = [c for c in keystats.columns if c not in drop_cols]

# 計算 MA6 (Smoothing)
for c in base_features:
    keystats[c + '_MA'] = keystats.groupby('Ticker')[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
ma_features = [c + '_MA' for c in base_features]

y = (keystats['stock_p_change'] - keystats['SP500_p_change'] > 15).astype(int)
n = len(keystats)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
test_df = keystats.iloc[val_end:].copy()

# ============================================================
# 2. 訓練兩個模型：Baseline (43維) vs Improved (86維)
# ============================================================
print("Training Baseline Model (No MA6)...")
X_base = keystats[base_features].fillna(keystats[base_features].median())
X_train_base = pd.concat([X_base.iloc[:train_end], X_base.iloc[train_end:val_end]])
y_train = pd.concat([y.iloc[:train_end], y.iloc[train_end:val_end]])
rf_base = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
rf_base.fit(X_train_base, y_train)
test_df['prob_base'] = rf_base.predict_proba(X_base.iloc[val_end:])[:, 1]

print("Training Improved Model (With MA6 Smoothing)...")
X_imp = keystats[base_features + ma_features].fillna(keystats[base_features + ma_features].median())
X_train_imp = pd.concat([X_imp.iloc[:train_end], X_imp.iloc[train_end:val_end]])
rf_imp = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
rf_imp.fit(X_train_imp, y_train)
test_df['prob_imp'] = rf_imp.predict_proba(X_imp.iloc[val_end:])[:, 1]

# ============================================================
# 3. 回測引擎
# ============================================================
price_df = pd.read_csv(ROOT / 'stock_prices.csv')
price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
price_df = price_df.set_index('Date').sort_index()
price_df.columns = [c.upper() for c in price_df.columns]
test_df['Ticker_upper'] = test_df['Ticker'].str.upper()

def run_backtest(prob_col):
    cash, holdings, equity_curve, last_prices = INITIAL_CAPITAL, [], [], {}
    dates = sorted(test_df['Date'].dt.normalize().unique())
    
    for date in dates:
        if date in price_df.index:
            for t in price_df.columns:
                p = price_df.loc[date, t]
                if not np.isnan(p): last_prices[t] = p
        
        new_h = []
        for h in holdings:
            if h['ticks'] >= HOLD_DAYS:
                exit_p = last_prices.get(h['ticker'])
                if exit_p:
                    cash += h['shares'] * exit_p * (1 - COST)
                    continue
            h['ticks'] += 1
            new_h.append(h)
        holdings = new_h
        
        if not holdings:
            day_data = test_df[test_df['Date'].dt.normalize() == date]
            if len(day_data) > 0:
                picks = day_data.nlargest(TOP_N, prob_col)
                cash_per = cash / len(picks)
                for _, row in picks.iterrows():
                    ticker = row['Ticker_upper']
                    entry_p = last_prices.get(ticker)
                    if entry_p:
                        shares = (cash_per / (1+COST)) / entry_p
                        cash -= cash_per
                        holdings.append({'ticker': ticker, 'entry_date': date, 'entry_price': entry_p, 'shares': shares, 'invested': cash_per, 'ticks': 0})
        
        equity = cash + sum(h['shares'] * last_prices.get(h['ticker'], h['entry_price']) for h in holdings)
        equity_curve.append({'date': date, 'equity': equity})
        
    return pd.DataFrame(equity_curve).set_index('date')['equity']

eq_base = run_backtest('prob_base')
eq_imp = run_backtest('prob_imp')
spy_eq = INITIAL_CAPITAL * (sp500.reindex(eq_imp.index).ffill()['Close'] / sp500.reindex(eq_imp.index).ffill()['Close'].iloc[0])

# ============================================================
# 4. 計算指標並輸出 CSV
# ============================================================
def calc_metrics(eq_series, name):
    eq_series = eq_series.dropna()
    total_ret = eq_series.iloc[-1] / eq_series.iloc[0] - 1
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    cagr = (eq_series.iloc[-1] / eq_series.iloc[0])**(1/years) - 1 if years > 0 else 0
    rets = eq_series.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
    drawdown = eq_series / eq_series.cummax() - 1
    return {"Strategy": name, "Total Return": total_ret*100, "CAGR": cagr*100, 
            "Sharpe": sharpe, "Max DD": abs(drawdown.min()*100)}

metrics_df = pd.DataFrame([
    calc_metrics(eq_base, "Baseline (43 Features)"),
    calc_metrics(eq_imp, "Improved (43 + MA6 Smoothing)"),
    calc_metrics(spy_eq, "S&P 500 Market")
])
metrics_df.to_csv(OUT / "portfolio_metrics_clean.csv", index=False)
print("\n=== CLEAN METRICS (Without Leaky Model) ===")
print(metrics_df.to_string(index=False))

# ============================================================
# 5. 畫圖：只對比 Baseline vs Improved vs SPY (1x3 圖表)
# ============================================================
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Effect of MA6 Smoothing: Baseline vs Improved System', fontsize=18, fontweight='bold', y=1.05)

labels = metrics_df['Strategy']
colors = ['#A9A9A9', '#2ca02c', '#d62728'] # 灰、綠、紅

# 1. Total Return
axes[0].bar(labels, metrics_df['Total Return'], color=colors)
axes[0].set_title('Total Return (%)', fontweight='bold')
for i, v in enumerate(metrics_df['Total Return']):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

# 2. Sharpe Ratio
axes[1].bar(labels, metrics_df['Sharpe'], color=colors)
axes[1].set_title('Sharpe Ratio (Higher is Better)', fontweight='bold')
for i, v in enumerate(metrics_df['Sharpe']):
    axes[1].text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)
axes[1].patches[1].set_edgecolor('gold')
axes[1].patches[1].set_linewidth(4)

# 3. Max Drawdown
axes[2].bar(labels, metrics_df['Max DD'], color=colors)
axes[2].set_title('Max Drawdown (%) (Lower is Better)', fontweight='bold')
for i, v in enumerate(metrics_df['Max DD']):
    axes[2].text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()

# 【修復 Bug】使用 os.path.join 確保 Windows 路徑正確
save_path = os.path.join(str(OUT), "chart4_ma6_effect.png")
plt.savefig(save_path, dpi=150) # 移除 bbox_inches='tight' 避免 Error 22
plt.close()

print(f"\n✅ 已生成終極對比圖: {save_path}")
