from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
# 1. 載入數據與訓練模型
# ============================================================
print("Loading data and training models...")
keystats = pd.read_csv(ROOT / 'keystats.csv')
keystats['Date'] = pd.to_datetime(keystats['Date']).dt.tz_localize(None)
keystats = keystats.sort_values('Date').reset_index(drop=True)

sp500 = pd.read_csv(ROOT / 'sp500_index.csv')
sp500['Date'] = pd.to_datetime(sp500['Date']).dt.tz_localize(None)
sp500 = sp500.set_index('Date').sort_index()

drop_cols = ['Date', 'Unix', 'Ticker', 'stock_p_change', 'SP500_p_change']
base_features = [c for c in keystats.columns if c not in drop_cols]
for c in base_features:
    keystats[c + '_MA'] = keystats.groupby('Ticker')[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
all_features = base_features + [c + '_MA' for c in base_features]

X = keystats[all_features].fillna(keystats[all_features].median())
y = (keystats['stock_p_change'] - keystats['SP500_p_change'] > 15).astype(int)

n = len(keystats)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
test_df = keystats.iloc[val_end:].copy()
X_test = X.iloc[val_end:]

# 1a. 原版 (Leaky)
X_train_rand, _, y_train_rand, _ = train_test_split(X, y, test_size=0.15, random_state=42)
rf_rand = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
rf_rand.fit(X_train_rand, y_train_rand)
test_df['prob_rand'] = rf_rand.predict_proba(X_test)[:, 1]

# 1b. 改良版 (Time Split + OOB)
X_train_time = pd.concat([X.iloc[:train_end], X.iloc[train_end:val_end]])
y_train_time = pd.concat([y.iloc[:train_end], y.iloc[train_end:val_end]])
rf_time = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', oob_score=True, random_state=42)
rf_time.fit(X_train_time, y_train_time)
test_df['prob_time'] = rf_time.predict_proba(X_test)[:, 1]

# ============================================================
# 2. 回測引擎
# ============================================================
price_df = pd.read_csv(ROOT / 'stock_prices.csv')
price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
price_df = price_df.set_index('Date').sort_index()
price_df.columns = [c.upper() for c in price_df.columns]
test_df['Ticker_upper'] = test_df['Ticker'].str.upper()

def run_backtest(prob_col):
    cash, holdings, trades, equity_curve, last_prices = INITIAL_CAPITAL, [], [], [], {}
    
    for date in sorted(test_df['Date'].dt.normalize().unique()):
        if date in price_df.index:
            for t in price_df.columns:
                p = price_df.loc[date, t]
                if not np.isnan(p): last_prices[t] = p
        
        new_h = []
        for h in holdings:
            if h['ticks'] >= HOLD_DAYS:
                exit_p = last_prices.get(h['ticker'])
                if exit_p:
                    net = h['shares'] * exit_p * (1 - COST)
                    cash += net
                    trades.append({'Ticker': h['ticker'], 'Entry_Date': h['entry_date'], 'Entry_Price': h['entry_price'],
                                   'Exit_Date': date, 'Exit_Price': exit_p, 'PnL': net - h['invested']})
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
        
    return pd.DataFrame(equity_curve).set_index('date')['equity'], pd.DataFrame(trades)

print("Running backtests...")
eq_orig, trades_orig = run_backtest('prob_rand')
eq_imp, trades_imp = run_backtest('prob_time')

spy_eq = INITIAL_CAPITAL * (sp500.reindex(eq_imp.index).ffill()['Close'] / sp500.reindex(eq_imp.index).ffill()['Close'].iloc[0])

# ============================================================
# 3. 計算 Portfolio Metrics (教授要的量化指標)
# ============================================================
def calc_metrics(eq_series, name):
    eq_series = eq_series.dropna()
    total_ret = eq_series.iloc[-1] / eq_series.iloc[0] - 1
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    cagr = (eq_series.iloc[-1] / eq_series.iloc[0])**(1/years) - 1 if years > 0 else 0
    rets = eq_series.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
    drawdown = eq_series / eq_series.cummax() - 1
    return {
        "Strategy": name,
        "Total Return": f"{total_ret*100:.1f}%",
        "CAGR": f"{cagr*100:.1f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{drawdown.min()*100:.1f}%"
    }

metrics_df = pd.DataFrame([
    calc_metrics(eq_orig, "Original ML (Leaky)"),
    calc_metrics(eq_imp, "Improved ML (Our System)"),
    calc_metrics(spy_eq, "Buy & Hold S&P 500")
])

print("\n" + "="*70)
print("PORTFOLIO PERFORMANCE METRICS")
print("="*70)
print(metrics_df.to_string(index=False))
metrics_df.to_csv(OUT / "portfolio_metrics.csv", index=False)

# ============================================================
# 4. 畫圖 (Figure 1, 2, 2b)
# ============================================================
# Figure 1: 三方對比
print("\nGenerating Figure 1...")
plt.figure(figsize=(12, 5))
plt.plot(eq_orig.index, eq_orig, label='Original ML (Leaky)', color='gray', linestyle=':', linewidth=2)
plt.plot(eq_imp.index, eq_imp, label='Improved ML (Time Split + MA6 + OOB)', color='blue', linewidth=2)
plt.plot(spy_eq.index, spy_eq, label='Buy & Hold S&P 500', color='red', linestyle='--')
plt.yscale('log')
plt.title('Figure 1: Improved Model vs Original (Leaky) vs Market', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT / "chart1_equity_curve_v2.png"), dpi=150) # 修正 Path 錯誤
plt.close()

# Figure 2: 2x2 網格圖 (抽 4 隻股票)
print("Generating Figure 2 (4-Stock Grid)...")
unique_tickers = trades_imp['Ticker'].unique()
sample_tickers = unique_tickers[:4] if len(unique_tickers) >= 4 else unique_tickers

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, ticker in enumerate(sample_tickers):
    trade = trades_imp[trades_imp['Ticker'] == ticker].iloc[0]
    t_start = trade['Entry_Date'] - pd.Timedelta(days=30)
    t_end = trade['Exit_Date'] + pd.Timedelta(days=30)
    
    stock_prices = price_df.loc[t_start:t_end, ticker].dropna()
    ax = axes[i]
    ax.plot(stock_prices.index, stock_prices.values, color='black', linewidth=1.5, label=f'{ticker} Price')
    ax.scatter(trade['Entry_Date'], trade['Entry_Price'], color='green', s=150, marker='^', zorder=5, label='BUY')
    ax.scatter(trade['Exit_Date'], trade['Exit_Price'], color='red', s=150, marker='v', zorder=5, label='SELL')
    
    ax.set_title(f'{ticker} Trade Signal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

plt.suptitle('Figure 2: Traceable Trade Signals (4 Different Stocks)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUT / "chart2_trade_grid.png"), dpi=150)
plt.close()

# Figure 2b: 交易頻次分佈
print("Generating Figure 2b (Trade Distribution)...")
ticker_counts = trades_imp['Ticker'].value_counts()
top_tickers = ticker_counts.head(15)

plt.figure(figsize=(10, 6))
plt.bar(top_tickers.index, top_tickers.values, color='teal', alpha=0.8)
plt.title('Figure 2b: Portfolio Diversification (Top 15 Traded Stocks)', fontsize=14, fontweight='bold')
plt.xlabel('Stock Ticker')
plt.ylabel('Number of Trades')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT / "chart2b_trade_distribution.png"), dpi=150)
plt.close()

print("\n✅ 所有圖表與數據已生成：")
print(f"1. {OUT / 'chart1_equity_curve_v2.png'}")
print(f"2. {OUT / 'chart2_trade_grid.png'}")
print(f"3. {OUT / 'chart2b_trade_distribution.png'}")
print(f"4. {OUT / 'portfolio_metrics.csv'} (供報告使用的數據表)")
