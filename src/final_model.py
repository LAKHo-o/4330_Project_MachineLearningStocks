from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# =========================
# 最終設定
# =========================
RANDOM_STATE = 42
GAP_DAYS = 20
MA_WINDOW = 6
LABEL_THRESHOLD = 15
TOP_N = 3
COST_RATE = 0.002
HOLD_DAYS_LIST = [5, 20, 60]
STOP_LOSS = 0.10
TAKE_PROFIT = 0.30

ROOT = Path(__file__).resolve().parents[1]


def find_file(filename: str) -> Path:
    candidates = [
        ROOT / filename,
        ROOT / "data" / filename,
        ROOT / "MachineLearningStocks" / filename,
        ROOT / "data" / "raw" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"搵唔到 {filename}。請放喺 repo root 或 data/ folder。"
    )


def load_data():
    keystats = pd.read_csv(find_file("keystats.csv"))
    prices = pd.read_csv(find_file("stock_prices.csv"))

    keystats["Date"] = pd.to_datetime(keystats["Date"])
    keystats = keystats.sort_values("Date").reset_index(drop=True)

    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.set_index("Date").sort_index()
    prices.columns = [str(c).upper() for c in prices.columns]

    return keystats, prices


def split_time(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val_raw = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    train_last_date = train["Date"].max()
    val_start_date = train_last_date + pd.Timedelta(days=GAP_DAYS)

    val = val_raw[val_raw["Date"] >= val_start_date].copy()
    return train, val, test


def add_ma_features(df: pd.DataFrame, feature_cols, window: int):
    df_new = df.copy()
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date")
        for col in feature_cols:
            df_new.loc[group.index, f"{col}_MA"] = (
                group[col].rolling(window=window, min_periods=1).mean()
            )
    return df_new


def build_xy(train, val, test, ma_window=None):
    drop_cols = [
        "Date",
        "Unix",
        "Ticker",
        "stock_p_change",
        "SP500_p_change",
    ]
    base_cols = [c for c in train.columns if c not in drop_cols]

    if ma_window is not None:
        train = add_ma_features(train, base_cols, ma_window)
        val = add_ma_features(val, base_cols, ma_window)
        test = add_ma_features(test, base_cols, ma_window)

        ma_cols = [f"{c}_MA" for c in base_cols]
        feature_cols = base_cols + ma_cols
    else:
        feature_cols = base_cols

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()

    median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(median)
    X_val = X_val.fillna(median)
    X_test = X_test.fillna(median)

    y_train = (
        train["stock_p_change"] - train["SP500_p_change"] > LABEL_THRESHOLD
    ).astype(int)
    y_val = (
        val["stock_p_change"] - val["SP500_p_change"] > LABEL_THRESHOLD
    ).astype(int)
    y_test = (
        test["stock_p_change"] - test["SP500_p_change"] > LABEL_THRESHOLD
    ).astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_oob_rf(X_train, y_train, X_val, y_val, X_test):
    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_full, y_full)

    oob_probs = rf.oob_decision_function_[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.30, 0.70, 0.05):
        pred = (oob_probs > threshold).astype(int)
        score = f1_score(y_full, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    test_probs = rf.predict_proba(X_test)[:, 1]

    return {
        "model": rf,
        "threshold": best_threshold,
        "oob_score": rf.oob_score_,
        "oob_f1": best_f1,
        "oob_probs": oob_probs,
        "test_probs": test_probs,
        "y_full": y_full,
    }


def evaluate_model(y_test, probs, threshold):
    pred = (probs > threshold).astype(int)
    return {
        "auc": roc_auc_score(y_test, probs),
        "f1": f1_score(y_test, pred, zero_division=0),
        "report": classification_report(
            y_test,
            pred,
            target_names=["Not Outperform", "Outperform"],
            zero_division=0,
        ),
    }


def backtest_cycle(
    test_df,
    test_probs,
    prices,
    hold_days=20,
    top_n=3,
    cost_rate=0.002,
    stop_loss=0.10,
    take_profit=0.30,
    initial_capital=1000.0,
):
    df = test_df[["Date", "Ticker"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker_upper"] = df["Ticker"].astype(str).str.upper()
    df["y_prob"] = test_probs

    all_dates = prices.index
    capital = initial_capital

    period_returns = []
    portfolio_values = []
    period_dates = []

    start_date = df["Date"].min()
    i = all_dates.searchsorted(start_date, side="left")

    while i + hold_days < len(all_dates):
        buy_date = all_dates[i]
        sell_date = all_dates[i + hold_days - 1]

        cycle_data = df[
            (df["Date"] >= buy_date) & (df["Date"] <= sell_date)
        ].copy()

        if cycle_data.empty:
            i += hold_days
            continue

        cycle_data = (
            cycle_data.sort_values("Date")
            .groupby("Ticker_upper")
            .tail(1)
        )
        selected = cycle_data.sort_values("y_prob", ascending=False).head(top_n)
        tickers = selected["Ticker_upper"].tolist()

        if not tickers:
            i += hold_days
            continue

        cash_per_stock = capital / len(tickers)
        final_value = 0.0

        for ticker in tickers:
            if ticker not in prices.columns:
                final_value += cash_per_stock
                continue

            buy_series = prices.loc[prices.index >= buy_date, ticker].dropna()
            if buy_series.empty:
                final_value += cash_per_stock
                continue

            buy_price = float(buy_series.iloc[0])

            sell_series = prices.loc[
                (prices.index > buy_date) & (prices.index <= sell_date),
                ticker,
            ].dropna()

            if sell_series.empty:
                final_value += cash_per_stock
                continue

            sell_price = None

            for price in sell_series:
                ret = (price - buy_price) / buy_price
                if ret <= -stop_loss:
                    sell_price = buy_price * (1 - stop_loss)
                    break
                if ret >= take_profit:
                    sell_price = buy_price * (1 + take_profit)
                    break

            if sell_price is None:
                sell_price = float(sell_series.iloc[-1])

            gross_ret = (sell_price - buy_price) / buy_price
            net_ret = gross_ret - cost_rate
            final_value += cash_per_stock * (1 + net_ret)

        period_return = (final_value - capital) / capital
        period_returns.append(period_return)
        portfolio_values.append(final_value)
        period_dates.append(buy_date)

        capital = final_value
        i += hold_days

    if not period_returns:
        return None

    returns = np.array(period_returns)
    values = np.array(portfolio_values)

    total_return = (values[-1] - initial_capital) / initial_capital * 100
    avg_return = returns.mean() * 100
    std_return = returns.std() * 100
    sharpe = (
        avg_return / std_return * np.sqrt(252 / hold_days)
        if std_return > 0
        else 0.0
    )

    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    win_rate = (returns > 0).mean() * 100

    return {
        "total_return": total_return,
        "avg_return": avg_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "periods": len(returns),
        "values": values,
        "dates": period_dates,
    }


def main():
    print("Loading data...")
    keystats, prices = load_data()

    print(f"Samples: {len(keystats)}")
    print(f"Price range: {prices.index.min()} ~ {prices.index.max()}")

    train, val, test = split_time(keystats)

    print("\nSplit:")
    print(f"Train: {len(train)} | {train['Date'].min()} ~ {train['Date'].max()}")
    print(f"Val:   {len(val)} | {val['Date'].min()} ~ {val['Date'].max()}")
    print(f"Test:  {len(test)} | {test['Date'].min()} ~ {test['Date'].max()}")
    print(f"Gap:   {(val['Date'].min() - train['Date'].max()).days} days")

    # =========================
    # 你的最終模型：MA=6 + OOB RF
    # =========================
    print("\n" + "=" * 60)
    print("Final model: MA window 6 + OOB Random Forest")
    print("=" * 60)

    (
        X_train_ma,
        X_val_ma,
        X_test_ma,
        y_train,
        y_val,
        y_test,
        feature_cols_ma,
    ) = build_xy(train, val, test, ma_window=MA_WINDOW)

    final_result = train_oob_rf(
        X_train_ma,
        y_train,
        X_val_ma,
        y_val,
        X_test_ma,
    )

    final_metrics = evaluate_model(
        y_test,
        final_result["test_probs"],
        final_result["threshold"],
    )

    print(f"Feature count: {len(feature_cols_ma)}")
    print(f"OOB Score: {final_result['oob_score']:.4f}")
    print(f"Threshold: {final_result['threshold']:.2f}")
    print(f"Test AUC: {final_metrics['auc']:.4f}")
    print(f"Test F1: {final_metrics['f1']:.4f}")
    print("\nClassification report:")
    print(final_metrics["report"])

    # =========================
    # Baseline：原版 43 特徵 + 同樣 OOB
    # =========================
    print("\n" + "=" * 60)
    print("Baseline: original 43 features + OOB Random Forest")
    print("=" * 60)

    (
        X_train_base,
        X_val_base,
        X_test_base,
        _,
        _,
        _,
        feature_cols_base,
    ) = build_xy(train, val, test, ma_window=None)

    baseline_result = train_oob_rf(
        X_train_base,
        y_train,
        X_val_base,
        y_val,
        X_test_base,
    )

    baseline_metrics = evaluate_model(
        y_test,
        baseline_result["test_probs"],
        baseline_result["threshold"],
    )

    print(f"Feature count: {len(feature_cols_base)}")
    print(f"OOB Score: {baseline_result['oob_score']:.4f}")
    print(f"Threshold: {baseline_result['threshold']:.2f}")
    print(f"Test AUC: {baseline_metrics['auc']:.4f}")
    print(f"Test F1: {baseline_metrics['f1']:.4f}")
    print("\nClassification report:")
    print(baseline_metrics["report"])

    # =========================
    # 回測比較
    # =========================
    print("\n" + "=" * 60)
    print("Backtest comparison")
    print("=" * 60)

    rows = []

    for hold_days in HOLD_DAYS_LIST:
        final_bt = backtest_cycle(
            test,
            final_result["test_probs"],
            prices,
            hold_days=hold_days,
            top_n=TOP_N,
            cost_rate=COST_RATE,
            stop_loss=STOP_LOSS,
            take_profit=TAKE_PROFIT,
        )

        baseline_bt = backtest_cycle(
            test,
            baseline_result["test_probs"],
            prices,
            hold_days=hold_days,
            top_n=TOP_N,
            cost_rate=COST_RATE,
            stop_loss=STOP_LOSS,
            take_profit=TAKE_PROFIT,
        )

        if final_bt is None or baseline_bt is None:
            continue

        print(f"\nHold {hold_days} days")
        print(
            f"  Final MA6+OOB : "
            f"return={final_bt['total_return']:+.2f}% | "
            f"sharpe={final_bt['sharpe']:.2f} | "
            f"maxDD={final_bt['max_drawdown']:.2f}% | "
            f"win={final_bt['win_rate']:.1f}% | "
            f"periods={final_bt['periods']}"
        )
        print(
            f"  Baseline 43f  : "
            f"return={baseline_bt['total_return']:+.2f}% | "
            f"sharpe={baseline_bt['sharpe']:.2f} | "
            f"maxDD={baseline_bt['max_drawdown']:.2f}% | "
            f"win={baseline_bt['win_rate']:.1f}% | "
            f"periods={baseline_bt['periods']}"
        )

        rows.append(
            {
                "model": "MA6_OOB",
                "hold_days": hold_days,
                "auc": final_metrics["auc"],
                "f1": final_metrics["f1"],
                "total_return": final_bt["total_return"],
                "sharpe": final_bt["sharpe"],
                "max_drawdown": final_bt["max_drawdown"],
                "win_rate": final_bt["win_rate"],
                "periods": final_bt["periods"],
            }
        )
        rows.append(
            {
                "model": "Baseline_43_OOB",
                "hold_days": hold_days,
                "auc": baseline_metrics["auc"],
                "f1": baseline_metrics["f1"],
                "total_return": baseline_bt["total_return"],
                "sharpe": baseline_bt["sharpe"],
                "max_drawdown": baseline_bt["max_drawdown"],
                "win_rate": baseline_bt["win_rate"],
                "periods": baseline_bt["periods"],
            }
        )

    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "final_results.csv"

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
