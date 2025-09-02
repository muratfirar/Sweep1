# app.py
# ---------------------------------------------------------
# "Sweep + Displaced EMA" stratejisini backtesting.py ile koşturan
# ve REST API olarak sunan tek dosyalık FastAPI uygulaması.
#
# Uç noktalar:
#   GET  /health                  -> basit sağlık kontrolü
#   GET  /config                  -> varsayılan parametreleri döner
#   POST /backtest                -> backtest koşturur ve özet + (opsiyonel) işlemleri JSON döner
#   POST /backtest/plot           -> backtest koşturur ve interaktif HTML grafiğini döner
#
# Örnek istek:
#   curl -X POST https://<your-render-url>/backtest -H "Content-Type: application/json" \
#     -d '{"ticker":"SPY","period":"60d","interval":"15m","return_trades":true}'
#
# NOT:
# - YFinance bazen MultiIndex kolon döndürebilir; normalize ettik.
# - "slippage" parametresi bazı backtesting sürümlerinde yoktur; kullanmıyoruz.
# - Bokeh notebook çıktısı kapalı (server için güvenli).
# ---------------------------------------------------------

import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from backtesting import Backtest, Strategy, set_bokeh_output

# Sunucu ortamında notebook yok; uyarı/JS gereksinimini kapatıyoruz
set_bokeh_output(notebook=False)

# ========= Varsayılan Parametreler =========
DEFAULTS = {
    "intraday_interval": "15m",
    "intraday_period": "60d",
    "default_rr": 2.0,
    "use_displacement": True,
    "displacement_factor": 1.8,
    "sweep_lookback_bars": 192,
    "recent_sweep_window": 60,
    "ema_len": 50,
    "risk_per_trade_pct": 1.0,    # özsermayenin %
    "commission_pct": 0.0005,     # %0.05
}

# ========= İstek / Yanıt Şemaları =========
class BacktestRequest(BaseModel):
    ticker: str = Field(..., description="Örn: SPY, BTC-USD, XAUUSD=X, GARAN.IS")
    period: str = Field(DEFAULTS["intraday_period"], description="Örn: 60d, 30d, 2y")
    interval: str = Field(DEFAULTS["intraday_interval"], description="Örn: 15m, 30m, 1h, 1d")

    rr: float = Field(DEFAULTS["default_rr"], description="Risk:Ödül oranı")
    use_displacement: bool = Field(DEFAULTS["use_displacement"])
    displacement_factor: float = Field(DEFAULTS["displacement_factor"])
    sweep_lookback_bars: int = Field(DEFAULTS["sweep_lookback_bars"])
    recent_sweep_window: int = Field(DEFAULTS["recent_sweep_window"])
    ema_len: int = Field(DEFAULTS["ema_len"])

    risk_per_trade_pct: float = Field(DEFAULTS["risk_per_trade_pct"])
    commission_pct: float = Field(DEFAULTS["commission_pct"])

    cash: int = Field(100_000, description="Başlangıç nakit")
    return_trades: bool = Field(False, description="İşlemleri (trade listesi) JSON olarak ekle")
    max_trades: int = Field(250, description="Döndürülecek maksimum trade sayısı")
    do_plot: bool = Field(False, description="Grafik üret (HTML). /backtest JSON'da dönmez; bunun için /backtest/plot kullan")


class BacktestResponse(BaseModel):
    summary: Dict[str, Any]
    trades: Optional[List[Dict[str, Any]]] = None


# ========= Yardımcı Fonksiyonlar =========
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """YFinance verisini çeker; MultiIndex kolonları düzleştirir; OHLCV döner."""
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column"  # MultiIndex'i azaltmaya yardımcı
    )
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail=f"{ticker} için veri çekilemedi. period/interval uyumunu kontrol et.")

    # MultiIndex kolon varsa tek seviyeye indir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Beklenen kolonlar eksik: {missing}")

    df = df[keep].copy()
    # Index'i tz-naive yap
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(inplace=True)
    if df.empty:
        raise HTTPException(status_code=400, detail="Veri çekildi ama temizleme sonrası boş kaldı.")
    return df


def displaced_ema(close: pd.Series, length: int, use_disp: bool, disp_factor: float) -> pd.Series:
    """EMA + ileri kaydırma (sadece tetik filtresi olarak)."""
    ema = close.ewm(span=length, adjust=False).mean()
    if use_disp:
        disp_bars = max(1, int(round(disp_factor * 3)))  # 1.8 ~ 5 bar
        return ema.shift(disp_bars)
    return ema


def rolling_swing_high(high: pd.Series, lookback: int) -> pd.Series:
    return high.shift(1).rolling(lookback).max()


def rolling_swing_low(low: pd.Series, lookback: int) -> pd.Series:
    return low.shift(1).rolling(lookback).min()


def find_sweep_signals(
    df: pd.DataFrame,
    sweep_lookback_bars: int,
    ema_len: int,
    use_displacement: bool,
    displacement_factor: float,
) -> pd.DataFrame:
    """Sweep tespiti + displaced EMA hesapları."""
    out = df.copy()
    out["swing_hi"] = rolling_swing_high(out["High"], sweep_lookback_bars)
    out["swing_lo"] = rolling_swing_low(out["Low"], sweep_lookback_bars)
    out["swept_high"] = (out["High"] > out["swing_hi"])
    out["swept_low"] = (out["Low"] < out["swing_lo"])
    out["dema"] = displaced_ema(out["Close"], ema_len, use_displacement, displacement_factor)

    out["bars_since_high_sweep"] = np.nan
    out["bars_since_low_sweep"] = np.nan

    last_high_sweep = None
    last_low_sweep = None
    for i in range(len(out)):
        if out["swept_high"].iat[i]:
            last_high_sweep = i
        if out["swept_low"].iat[i]:
            last_low_sweep = i
        if last_high_sweep is not None:
            out["bars_since_high_sweep"].iat[i] = i - last_high_sweep
        if last_low_sweep is not None:
            out["bars_since_low_sweep"].iat[i] = i - last_low_sweep

    return out


# ========= Strateji =========
class SweepDisplacementStrategy(Strategy):
    # Bu alanlar, dinamik olarak "StrategyClassFactory" ile override edilecek
    rr: float = DEFAULTS["default_rr"]
    risk_pct: float = DEFAULTS["risk_per_trade_pct"]
    recent_window: int = DEFAULTS["recent_sweep_window"]

    sweep_lookback_bars: int = DEFAULTS["sweep_lookback_bars"]
    ema_len: int = DEFAULTS["ema_len"]
    use_displacement: bool = DEFAULTS["use_displacement"]
    displacement_factor: float = DEFAULTS["displacement_factor"]

    def init(self):
        df = self.data._df

        enriched = find_sweep_signals(
            df=df,
            sweep_lookback_bars=self.sweep_lookback_bars,
            ema_len=self.ema_len,
            use_displacement=self.use_displacement,
            displacement_factor=self.displacement_factor,
        )

        # backtesting.py -> self.I ile numpy vektörleri kaydediyoruz
        self.dema = self.I(lambda _: enriched["dema"].values, df["Close"])
        self.bars_since_high_sweep = self.I(lambda _: enriched["bars_since_high_sweep"].values, df["Close"])
        self.bars_since_low_sweep  = self.I(lambda _: enriched["bars_since_low_sweep"].values, df["Close"])

    def next(self):
        price = float(self.data.Close[-1])

        if not self.position:
            # ---- SHORT SETUP ----
            cond_recent_high_sweep = (
                (not np.isnan(self.bars_since_high_sweep[-1])) and
                (self.bars_since_high_sweep[-1] <= self.recent_window)
            )
            cond_below_dema = price < float(self.dema[-1]) if not np.isnan(self.dema[-1]) else False

            if cond_recent_high_sweep and cond_below_dema:
                stop = float(np.nanmax(self.data.High[-int(self.recent_window):]))
                if stop > price:
                    risk_per_unit = stop - price
                    if risk_per_unit > 0:
                        risk_cash = self.risk_pct / 100.0 * self.equity
                        size = max(1, int(risk_cash / risk_per_unit))
                        tp = price - self.rr * (stop - price)
                        self.sell(size=size, sl=stop, tp=tp)

            # ---- LONG SETUP ----
            cond_recent_low_sweep = (
                (not np.isnan(self.bars_since_low_sweep[-1])) and
                (self.bars_since_low_sweep[-1] <= self.recent_window)
            )
            cond_above_dema = price > float(self.dema[-1]) if not np.isnan(self.dema[-1]) else False

            if cond_recent_low_sweep and cond_above_dema:
                stop = float(np.nanmin(self.data.Low[-int(self.recent_window):]))
                if stop < price:
                    risk_per_unit = price - stop
                    if risk_per_unit > 0:
                        risk_cash = self.risk_pct / 100.0 * self.equity
                        size = max(1, int(risk_cash / risk_per_unit))
                        tp = price + self.rr * (price - stop)
                        self.buy(size=size, sl=stop, tp=tp)


def StrategyClassFactory(req: BacktestRequest):
    """İstek parametrelerini Strategy sınıfı özniteliklerine aktarır."""
    attrs = dict(
        rr=req.rr,
        risk_pct=req.risk_per_trade_pct,
        recent_window=req.recent_sweep_window,
        sweep_lookback_bars=req.sweep_lookback_bars,
        ema_len=req.ema_len,
        use_displacement=req.use_displacement,
        displacement_factor=req.displacement_factor,
    )
    return type("CustomSweepDisplacementStrategy", (SweepDisplacementStrategy,), attrs)


def run_backtest(req: BacktestRequest, want_plot: bool = False):
    """Backtest'i koşturur; pandas.Series (stats) ve opsiyonel plot HTML döner."""
    data = fetch_data(req.ticker, req.period, req.interval)
    StrategyCls = StrategyClassFactory(req)

    bt = Backtest(
        data,
        StrategyCls,
        cash=req.cash,
        commission=req.commission_pct,
        trade_on_close=False,
        exclusive_orders=True,
        hedging=False
    )

    stats = bt.run()
    plot_html = None
    if want_plot:
        # Dosyaya yazdırıp içeriğini string olarak döndürelim
        filename = f"plot_{req.ticker.replace('.', '_')}.html"
        bt.plot(filename=filename, open_browser=False)
        try:
            with open(filename, "r", encoding="utf-8") as f:
                plot_html = f.read()
        except Exception as e:
            plot_html = f"<html><body><h3>Plot üretilemedi</h3><pre>{e}</pre></body></html>"

    return stats, plot_html


def series_to_safe_dict(s: pd.Series) -> Dict[str, Any]:
    """backtesting stats (pandas.Series) -> JSON güvenli dict."""
    out = {}
    for k, v in s.items():
        if isinstance(v, (np.floating, np.float64, np.float32)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        else:
            # numpy datetime vb.
            try:
                if pd.isna(v):
                    out[k] = None
                else:
                    out[k] = v
            except Exception:
                out[k] = str(v)
    return out


def trades_df_to_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    """İşlem tablosunu (stats._trades) JSON'a çevirir."""
    if df is None or df.empty:
        return []
    # limit uygula
    df2 = df.head(limit).copy()
    # JSON-safe dönüşüm
    for col in df2.columns:
        if np.issubdtype(df2[col].dtype, np.floating):
            df2[col] = df2[col].astype(float)
        elif np.issubdtype(df2[col].dtype, np.integer):
            df2[col] = df2[col].astype(int)
        elif np.issubdtype(df2[col].dtype, np.datetime64):
            df2[col] = pd.to_datetime(df2[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df2.to_dict(orient="records")


# ========= FastAPI Uygulaması =========
app = FastAPI(title="Sweep + Displaced EMA Backtest API", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/config")
def config():
    return DEFAULTS


@app.post("/backtest", response_model=BacktestResponse)
def backtest_endpoint(req: BacktestRequest):
    stats, _ = run_backtest(req, want_plot=False)

    # Özet
    summary = series_to_safe_dict(stats)

    # İşlemler (opsiyonel)
    trades = None
    if req.return_trades:
        # backtesting.py, trades tablosunu genelde _trades attribute'u ile verir
        trades_df = getattr(stats, "_trades", None)
        if trades_df is not None:
            trades = trades_df_to_records(trades_df, req.max_trades)

    return BacktestResponse(summary=summary, trades=trades)


@app.post("/backtest/plot")
def backtest_plot_endpoint(req: BacktestRequest):
    """Backtest koştur ve interaktif HTML grafiğini döndür."""
    stats, html = run_backtest(req, want_plot=True)
    if not html:
        raise HTTPException(status_code=500, detail="Plot oluşturulamadı.")
    # HTML'i doğrudan döndürüyoruz
    return Response(content=html, media_type="text/html")


# Lokal geliştirme için:
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
