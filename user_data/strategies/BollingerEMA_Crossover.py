from __future__ import annotations

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import numpy as np


class BollingerEMA_Crossover(IStrategy):
    """
    Bollinger Bands + EMA kesişim sinyali tabanlı strateji
    - Bollinger Bands: SMA + standart sapma (kullanıcı verisi)
    - EMA kesişim sinyali: EMA slow ve EMA fast arasında
    """

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 200
    stoploss = -0.99

    # --- Parametreler (hyperopt ile optimize edilebilir) ---
    # Bollinger Bands parametreleri
    maLength = IntParameter(5, 200, default=20, space="buy", optimize=True)
    bbLength = IntParameter(5, 200, default=20, space="buy", optimize=True)
    mult = DecimalParameter(1.0, 5.0, default=2.0, decimals=2, space="buy", optimize=True)

    # EMA parametreleri
    emaSlowLength = IntParameter(5, 200, default=20, space="buy", optimize=True)
    emaFastLength = IntParameter(5, 200, default=10, space="buy", optimize=True)

    # RSI filtreleri (exit için)
    rsiLen = IntParameter(7, 21, default=14, space="sell", optimize=True)
    rsiOversold = IntParameter(15, 40, default=30, space="sell", optimize=True)
    rsiOverbought = IntParameter(60, 90, default=70, space="sell", optimize=True)

    # --- İndikatör hesaplamaları ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands (kullanıcı verisi)
        # src = dataframe["close"].rolling(self.maLength.value).mean() # Unused
        
        basis = dataframe["close"].rolling(self.bbLength.value).mean()
        stdev = dataframe["close"].rolling(self.bbLength.value).std()
        dev = self.mult.value * stdev

        upper = basis + dev
        lower = basis - dev

        # EMA hesaplamaları (3-period close ortalama)
        src_two = (dataframe["close"] + dataframe["low"] + dataframe["high"]) / 3
        ema_slow = src_two.ewm(span=self.emaSlowLength.value, adjust=False).mean()
        ema_fast = src_two.ewm(span=self.emaFastLength.value, adjust=False).mean()

        # İndikatörleri dataframe'e ekle
        dataframe["basis"] = basis
        dataframe["upper"] = upper
        dataframe["lower"] = lower
        dataframe["emaSlow"] = ema_slow
        dataframe["emaFast"] = ema_fast

        return dataframe

    # --- Girdi sinyalleri (EMA kesişim) ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA kesişim sinyalleri
        dataframe.loc[
            (dataframe["emaFast"] > dataframe["emaSlow"]) &
            (dataframe["emaFast"].shift(1) <= dataframe["emaSlow"].shift(1)),
            "enter_long"
        ] = 1

        dataframe.loc[
            (dataframe["emaFast"] < dataframe["emaSlow"]) &
            (dataframe["emaFast"].shift(1) >= dataframe["emaSlow"].shift(1)),
            "enter_short"
        ] = 1

        return dataframe

    # --- Çıkış sinyalleri (RSI + Bollinger Bands) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI filtreleri
        rsi_len = int(self.rsiLen.value)
        rsi_os = int(self.rsiOversold.value)
        rsi_ob = int(self.rsiOverbought.value)

        delta = dataframe["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_len).mean()
        rs = gain / (loss + 1e-9)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands çıkış sinyalleri
        dataframe.loc[
            (dataframe["rsi"] > rsi_ob) |
            (dataframe["upper"] < dataframe["close"]),
            "exit_long"
        ] = 1

        dataframe.loc[
            (dataframe["rsi"] < rsi_os) |
            (dataframe["lower"] > dataframe["close"]),
            "exit_short"
        ] = 1

        return dataframe
