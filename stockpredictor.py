import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Price Analysis & Prediction", layout="wide")
st.title("📈 Stock Price Analysis and Prediction")

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
tickers = st.sidebar.multiselect("Select Tickers", ["AAPL","MSFT","GOOGL","AMZN","META"], default=["AAPL"])
start = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end = st.sidebar.date_input("End Date", datetime.now())

model_choice = st.sidebar.selectbox("Model", ["Linear (PyTorch)", "Random Forest", "LSTM"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Chart Overlays")
show_sma20 = st.sidebar.checkbox("Show SMA 20", True)
show_sma50 = st.sidebar.checkbox("Show SMA 50", False)
show_sma200 = st.sidebar.checkbox("Show SMA 200", False)
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)

# ---------------- Data ----------------
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def calc_indicators(df):
    d = df.copy()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    # RSI
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    d["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    exp12 = d["Close"].ewm(span=12, adjust=False).mean()
    exp26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = exp12 - exp26
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    return d

def plot_chart(df, show_sma20, show_sma50, show_sma200, show_rsi, show_macd):
    rows = 1 + int(show_rsi) + int(show_macd)
    heights = [0.6, 0.2, 0.2] if rows == 3 else [0.7, 0.3] if rows == 2 else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=heights)
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    if show_sma20: fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA 20"), row=1, col=1)
    if show_sma50: fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA 50"), row=1, col=1)
    if show_sma200: fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA 200"), row=1, col=1)
    r = 2
    if show_rsi:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"), row=r, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=r, col=1)
        r += 1
    if show_macd:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"), row=r, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal"), row=r, col=1)
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Quick Stats ----------------
def quick_stats(df):
    """Return period return, annualized volatility, and max drawdown."""
    d = df.copy()
    d["ret"] = d["Close"].pct_change().fillna(0.0)
    period_return = (1 + d["ret"]).prod() - 1.0
    ann_vol = d["ret"].std() * np.sqrt(252)
    equity = (1 + d["ret"]).cumprod()
    run_max = np.maximum.accumulate(equity)
    max_dd = (equity / run_max - 1.0).min() if len(equity) else np.nan
    return float(period_return), float(ann_vol), float(max_dd)

# ---------------- Multi-ticker Performance (normalized to 100) ----------------
def multi_ticker_performance(dfs):
    # dfs: dict[ticker] -> df with Date & Close
    merged = None
    for tkr, df in dfs.items():
        if df is None or df.empty: continue
        tmp = df[["Date", "Close"]].copy()
        tmp = tmp.rename(columns={"Close": tkr})
        if merged is None:
            merged = tmp
        else:
            merged = pd.merge_asof(merged.sort_values("Date"),
                                   tmp.sort_values("Date"),
                                   on="Date", direction="nearest")
    if merged is None: 
        return None
    # Normalize each series to 100 at start
    for tkr in dfs.keys():
        if tkr in merged.columns:
            first = merged[tkr].dropna().iloc[0] if merged[tkr].notna().any() else np.nan
            merged[tkr] = 100 * (merged[tkr] / first)
    return merged.dropna()

def plot_multi_performance(perf_df):
    fig = go.Figure()
    for col in perf_df.columns:
        if col == "Date": continue
        fig.add_trace(go.Scatter(x=perf_df["Date"], y=perf_df[col], mode="lines", name=col))
    fig.update_layout(title="Multi-Ticker Performance (rebased to 100)", yaxis_title="Index (100 = start)", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Models ----------------
def train_linear(df, lr=0.05, epochs=400, optimizer_name="Adam"):
    s = df[["Date","Close"]].dropna().sort_values("Date").reset_index(drop=True)
    if len(s) < 3: raise ValueError("Not enough data to train Linear model.")
    X_close = s["Close"].iloc[:-1].to_numpy()
    y_close = s["Close"].iloc[1:].to_numpy()
    eps = 1e-8
    mu = X_close.mean(); sigma = X_close.std(ddof=0) + eps
    X = ((X_close - mu) / sigma).reshape(-1,1)
    y = ((y_close - mu) / sigma).reshape(-1,1)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = nn.Linear(1,1)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr) if optimizer_name=="Adam" else torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        opt.zero_grad(); pred = model(X_t)
        loss = crit(pred, y_t); loss.backward(); opt.step()

    last_close = s["Close"].iloc[-1]
    last_x = (last_close - mu) / sigma
    model.eval()
    with torch.no_grad():
        next_scaled = model(torch.tensor([[last_x]], dtype=torch.float32)).item()
    next_pred = next_scaled * sigma + mu
    return model, float(next_pred)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def metrics_from_preds(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))) * 100.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, mape, rmse

def build_features(df):
    d = df.sort_values("Date").reset_index(drop=True).copy()
    d["ret_1"]  = d["Close"].pct_change(1)
    d["ret_5"]  = d["Close"].pct_change(5)
    d["ret_10"] = d["Close"].pct_change(10)
    d["sma_5"]  = d["Close"].rolling(5).mean()
    d["sma_20"] = d["Close"].rolling(20).mean()
    d["sma_50"] = d["Close"].rolling(50).mean()
    d["vol_10"] = d["Close"].pct_change().rolling(10).std()
    d["vol_20"] = d["Close"].pct_change().rolling(20).std()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    d["rsi_14"] = 100 - (100 / (1 + rs))
    exp12 = d["Close"].ewm(span=12, adjust=False).mean()
    exp26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"] = exp12 - exp26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["y_next"] = d["Close"].shift(-1)
    feature_cols = ["Close","ret_1","ret_5","ret_10","sma_5","sma_20","sma_50","vol_10","vol_20","rsi_14","macd","macd_signal"]
    d = d.dropna().reset_index(drop=True)
    X = d[feature_cols].values
    y = d["y_next"].values
    return X, y

def train_random_forest(df, test_frac=0.2, n_estimators=400, random_state=42):
    X, y = build_features(df)
    if len(X) < 50: raise ValueError("Not enough rows after feature building for Random Forest.")
    split = max(10, int(len(X) * (1 - test_frac)))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred_te = model.predict(X_te) if len(X_te) else np.array([])
    if len(X_te):
        mae, mape, rmse = metrics_from_preds(y_te, y_pred_te)
    else:
        mae = mape = rmse = float("nan")
    next_pred = model.predict(X[-1:].copy())[0]
    return model, float(next_pred), {"MAE": mae, "MAPE": mape, "RMSE": rmse}

# --------- LSTM (shape-safe) ----------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, series: np.ndarray, seq_len: int):
        s = np.asarray(series).reshape(-1)
        X_list, Y_list = [], []
        for i in range(len(s) - seq_len):
            X_list.append(s[i:i+seq_len])
            Y_list.append(s[i+seq_len])
        X = np.stack(X_list, axis=0)
        Y = np.asarray(Y_list)
        self.x = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        self.y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1).unsqueeze(-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def train_lstm_next_close(df, seq_len=30, epochs=25, lr=1e-3):
    s = df.sort_values("Date").reset_index(drop=True)
    close = s["Close"].values.astype(np.float32).reshape(-1)
    if len(close) <= seq_len + 5: raise ValueError("Not enough data for LSTM with the chosen sequence length.")
    mu, sigma = float(close.mean()), float(close.std() + 1e-8)
    z = (close - mu) / sigma
    ds = SeqDataset(z, seq_len); n = len(ds)
    split = max(50, int(n * 0.8))
    ds_tr, ds_te = torch.utils.data.Subset(ds, range(0, split)), torch.utils.data.Subset(ds, range(split, n))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=32, shuffle=True)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=64, shuffle=False)

    model = SimpleLSTM(hidden=32, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl_tr:
            if xb.dim() == 4 and xb.size(-1) == 1:
                xb = xb.squeeze(-1).unsqueeze(-1)
            opt.zero_grad(); pred = model(xb)
            loss = crit(pred, yb)
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        preds, trues = [], []
        for xb, yb in dl_te:
            if xb.dim() == 4 and xb.size(-1) == 1:
                xb = xb.squeeze(-1).unsqueeze(-1)
            p = model(xb); preds.append(p.numpy()); trues.append(yb.numpy())
        if preds:
            y_pred = np.vstack(preds).reshape(-1)
            y_true = np.vstack(trues).reshape(-1)
            y_pred_u = y_pred * sigma + mu
            y_true_u = y_true * sigma + mu
            mae = float(np.mean(np.abs(y_true_u - y_pred_u)))
            mape = float(np.mean(np.abs((y_true_u - y_pred_u) / (y_true_u + 1e-8)))) * 100.0
            rmse = float(np.sqrt(np.mean((y_true_u - y_pred_u) ** 2)))
        else:
            mae = mape = rmse = float("nan")

    last_seq = torch.tensor(z[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
    with torch.no_grad():
        next_z = model(last_seq).item()
    next_pred = next_z * sigma + mu
    return model, float(next_pred), {"MAE": mae, "MAPE": mape, "RMSE": rmse}

# ---------------- Main ----------------
# Multi-ticker performance section (top of page)
if len(tickers) >= 2:
    st.subheader("📊 Multi-Ticker Performance (rebased to 100)")
    dfs_for_merge = {t: get_stock_data(t, start, end) for t in tickers}
    perf = multi_ticker_performance(dfs_for_merge)
    if perf is not None and not perf.empty:
        plot_multi_performance(perf)
        # Download button for the comparison data
        st.download_button(
            "Download performance CSV",
            data=perf.to_csv(index=False).encode("utf-8"),
            file_name="multi_ticker_performance.csv",
            mime="text/csv",
        )
    st.divider()

for ticker in tickers:
    st.subheader(f"📊 {ticker} Price & Prediction")
    df = get_stock_data(ticker, start, end)
    if df is None or df.empty:
        st.error(f"Could not retrieve data for {ticker}."); 
        continue

    # Quick stats
    pr, vol, mdd = quick_stats(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Period Return", f"{pr:.2%}")
    c2.metric("Ann. Volatility", f"{vol:.2%}")
    c3.metric("Max Drawdown", f"{mdd:.2%}")

    # Indicators & chart
    df_ind = calc_indicators(df)
    plot_chart(df_ind, show_sma20, show_sma50, show_sma200, show_rsi, show_macd)

    # Models
    try:
        if model_choice == "Linear (PyTorch)":
            model, next_pred = train_linear(df)
            st.success(f"Predicted next-day close for {ticker}: ${next_pred:.2f}")
            metrics = {"MAE": np.nan, "MAPE": np.nan, "RMSE": np.nan}

        elif model_choice == "Random Forest":
            model, next_pred, metrics = train_random_forest(df)
            st.success(f"Predicted next-day close for {ticker}: ${next_pred:.2f}")

        elif model_choice == "LSTM":
            model, next_pred, metrics = train_lstm_next_close(df, seq_len=30, epochs=25, lr=1e-3)
            st.success(f"Predicted next-day close for {ticker}: ${next_pred:.2f}")

        # Metrics table (for RF/LSTM; Linear uses NaNs as placeholder)
        st.markdown("### ⚙️ Model Performance")
        mdf = pd.DataFrame(
            {"Metric": ["MAE","MAPE","RMSE"],
             "Value": [metrics["MAE"], f'{metrics["MAPE"]:.2f}%' if np.isfinite(metrics["MAPE"]) else "—", metrics["RMSE"]]}
        )
        st.dataframe(mdf, use_container_width=True)

    except Exception as e:
        st.error(f"Model run failed: {e}")

    # Download button for this ticker’s OHLCV + indicators
    out_df = df_ind.copy()
    st.download_button(
        f"Download {ticker} data (with indicators)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_data_with_indicators.csv",
        mime="text/csv",
    )


