from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 基本設定
# ============================================================
ROOT = Path(__file__).resolve().parents[1] # 請確保你的 Jupyter Notebook 和 csv 文件在同一個資料夾
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

INITIAL_CAPITAL = 100_000
HOLD_DAYS = 20
TOP_N = 3
COST = 0.0005
CASH_ANNUAL = 0.03
GAP_YEARS = 2 # 預測 2009 年時，訓練集最多只到 2007 年底 (防 1 年標籤洩漏)

# ============================================================
# 1. 載入數據與特徵工程
# ============================================================
print("Loading data...")
keystats = pd.read_csv(ROOT / 'keystats.csv')
keystats['Date'] = pd.to_datetime(keystats['Date']).dt.tz_localize(None)
keystats = keystats.sort_values('Date').reset_index(drop=True)

sp500 = pd.read_csv(ROOT / 'sp500_index.csv')
sp500['Date'] = pd.to_datetime(sp500['Date']).dt.tz_localize(None)
sp500 = sp500.set_index('Date').sort_index()

# 大市過濾器 (Golden Cross)
sp500['sma50'] = sp500['Close'].rolling(50).mean()
sp500['sma200'] = sp500['Close'].rolling(200).mean()
sp500['is_bull'] = (sp500['Close'] > sp500['sma200']) & (sp500['sma50'] > sp500['sma200'])

# 特徵 (MA6)
drop_cols = ['Date', 'Unix', 'Ticker', 'stock_p_change', 'SP500_p_change']
base_features = [c for c in keystats.columns if c not in drop_cols]
for c in base_features:
    keystats[c + '_MA'] = keystats.groupby('Ticker')[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
all_features = base_features + [c + '_MA' for c in base_features]

X = keystats[all_features].copy()
y = (keystats['stock_p_change'] - keystats['SP500_p_change'] > 15).astype(int)
X = X.fillna(X.median(numeric_only=True))

# 股價數據 (用於回測)
price_df = pd.read_csv(ROOT / 'stock_prices.csv')
price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
price_df = price_df.set_index('Date').sort_index()
price_df.columns = [c.upper() for c in price_df.columns]

# ============================================================
# 2. Walk-Forward 核心：逐年切分與預測
# ============================================================
keystats['Year'] = keystats['Date'].dt.year
test_years = sorted([yr for yr in keystats['Year'].unique() if yr >= 2009])

wf_test_dfs = []

for test_year in test_years:
    # 計算訓練集截止日
    train_end_date = pd.Timestamp(f"{test_year - GAP_YEARS}-12-31")
    
    train_mask = keystats['Date'] <= train_end_date
    test_mask = keystats['Year'] == test_year
    
    train_idx = keystats[train_mask].index
    test_idx = keystats[test_mask].index
    
    X_train_time, y_train_time = X.loc[train_idx], y.loc[train_idx]
    X_test_time, y_test_time = X.loc[test_idx], y.loc[test_idx]
    
    test_df_wf = keystats.loc[test_idx].copy()
    test_df_wf['Ticker_upper'] = test_df_wf['Ticker'].str.upper()
    
    print(f"Test Year: {test_year} | Train Size: {len(X_train_time)} | Test Size: {len(X_test_time)}")
    
    # --- A. Improved ML (Time Split) ---
    rf_time = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', oob_score=True, random_state=42, n_jobs=-1)
    rf_time.fit(X_train_time, y_train_time)
    test_df_wf['prob_time'] = rf_time.predict_proba(X_test_time)[:, 1]
    
    # --- B. Original ML (Leaky) ---
    # 模擬在當前年份內，把包含未來的數據合併，隨機切分
    X_all_wf = pd.concat([X_train_time, X_test_time])
    y_all_wf = pd.concat([y_train_time, y_test_time])
    X_train_rand, _, y_train_rand, _ = train_test_split(X_all_wf, y_all_wf, test_size=0.15, random_state=42)
    rf_rand = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_rand.fit(X_train_rand, y_train_rand)
    test_df_wf['prob_rand'] = rf_rand.predict_proba(X_test_time)[:, 1]
    
    wf_test_dfs.append(test_df_wf)

# 拼接所有年份的測試集
test_df_wf_all = pd.concat(wf_test_dfs).sort_values('Date').reset_index(drop=True)
print(f"\n✅ Walk-Forward 測試集拼接完成，總樣本數: {len(test_df_wf_all)}")

# ============================================================
# 3. 回測引擎 (沿用你的邏輯)
# ============================================================
def run_backtest(test_df, prob_col, use_filter=False):
    dates = sorted(test_df['Date'].dt.normalize().unique())
    cash = INITIAL_CAPITAL
    holdings = []
    trades = []
    equity_curve = []
    cash_daily = (1 + CASH_ANNUAL)**(1/252) - 1
    last_known_prices = {}

    for date in dates:
        if date in price_df.index:
            for ticker in price_df.columns:
                p = price_df.loc[date, ticker]
                if not np.isnan(p):
                    last_known_prices[ticker] = p

        if not holdings:
            cash *= (1 + cash_daily)

        new_holdings = []
        for h in holdings:
            if h['ticks_held'] >= HOLD_DAYS:
                exit_price = last_known_prices.get(h['ticker'])
                if exit_price and exit_price > 0:
                    net = h['shares'] * exit_price * (1 - COST)
                    cash += net
                    trades.append({'Strategy': prob_col, 'Ticker': h['ticker'], 'Entry_Date': h['entry_date'],
                                   'Entry_Price': h['entry_price'], 'Invested': h['invested'],
                                   'Exit_Date': date, 'Exit_Price': exit_price, 'Net_Proceeds': net,
                                   'PnL': net - h['invested'], 'Cash_After': cash})
                    continue
            h['ticks_held'] += 1
            new_holdings.append(h)
        holdings = new_holdings

        if len(holdings) == 0:
            is_bull = sp500.loc[date, 'is_bull'] if date in sp500.index else False
            if not use_filter or is_bull:
                day_data = test_df[test_df['Date'].dt.normalize() == date]
                if len(day_data) > 0:
                    picks = day_data.nlargest(TOP_N, prob_col)
                    cash_per = cash / len(picks)
                    for _, row in picks.iterrows():
                        ticker = row['Ticker_upper']
                        entry_price = last_known_prices.get(ticker)
                        if entry_price and entry_price > 0:
                            shares = (cash_per / (1+COST)) / entry_price
                            cash -= cash_per
                            holdings.append({'ticker': ticker, 'entry_date': date, 'entry_price': entry_price,
                                             'shares': shares, 'invested': cash_per, 'ticks_held': 0})

        equity = cash
        for h in holdings:
            p = last_known_prices.get(h['ticker'], h['entry_price'])
            equity += h['shares'] * p
        equity_curve.append({'date': date, 'equity': equity})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index('date')['equity']

print("Running Walk-Forward Backtests...")
trades_orig, eq_orig = run_backtest(test_df_wf_all, 'prob_rand', use_filter=False)
trades_imp, eq_imp = run_backtest(test_df_wf_all, 'prob_time', use_filter=False)
trades_gc, eq_gc = run_backtest(test_df_wf_all, 'prob_time', use_filter=True)

spy_test = sp500.reindex(eq_orig.index).ffill().bfill()
spy_eq = INITIAL_CAPITAL * (spy_test['Close'] / spy_test['Close'].iloc[0])

# ============================================================
# 4. 績效計算
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
        "Final Equity": eq_series.iloc[-1],
        "Total Return %": total_ret * 100,
        "CAGR %": cagr * 100,
        "Sharpe": sharpe,
        "Max DD %": drawdown.min() * 100
    }

summary_df = pd.DataFrame([
    calc_metrics(eq_orig, "1. Original ML (Leaky)"),
    calc_metrics(eq_imp, "2. Improved ML (Time Split)"),
    calc_metrics(eq_gc, "3. Improved ML + GC Filter"),
    calc_metrics(spy_eq, "4. Buy & Hold SPY")
])

print("\n" + "="*90)
print("WALK-FORWARD COMPARISON (Out-of-Sample 2009-2013)")
print("="*90)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ============================================================
# 5. 畫圖
# ============================================================
plt.figure(figsize=(14, 6))
for name, eq in [('Original ML (Leaky)', eq_orig), ('Improved ML', eq_imp), 
                 ('Improved ML + GC Filter', eq_gc), ('Buy & Hold SPY', spy_eq)]:
    plt.plot(eq.index, eq, label=name)

plt.yscale("log")
plt.title("Walk-Forward Validation (Out-of-Sample 2009-2013)")
plt.xlabel("Date")
plt.ylabel("Equity (log)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = OUT / "walk_forward_comparison.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"\n✅ 圖片已儲存至: {plot_path}")
