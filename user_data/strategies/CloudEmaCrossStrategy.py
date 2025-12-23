from __future__ import annotations

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class CloudEmaCrossStrategy(IStrategy):
    """
    Kaynak koddaki bant + EMA bulutu hesaplarını üretir.
    Sinyaller EMA fast/slow kesişimine göre verilir.

    - srcTwo = (close + low + high) / 3
    - emaFast = EMA(srcTwo, emaFastLength)
    - emaSlow = EMA(srcTwo, emaSlowLength)

    Entry/Exit:
      enter_long  : emaFast crosses above emaSlow
      exit_long   : emaFast crosses below emaSlow

    Bollinger-benzeri üst/alt bantlar (lower/upper) stratejide filtre olarak kullanılmıyor,
    ama dataframe'e ekleniyor (istersen filtre ekleyebilirsin).
    """

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 300

    can_short = False  # İstersen True yapıp populate_exit_trend'e short ekleyebilirsin.

    # Basit risk ayarları (örnek)
    minimal_roi = {"0": 0.02}
    stoploss = -0.05
    trailing_stop = False

    # ---- options.* -> parametrik ----
    maLength = IntParameter(2, 200, default=20, space="buy", optimize=True)
    bbLength = IntParameter(2, 200, default=20, space="buy", optimize=True)
    mult = DecimalParameter(0.5, 5.0, default=2.0, decimals=2, space="buy", optimize=True)

    emaSlowLength = IntParameter(5, 400, default=50, space="buy", optimize=True)
    emaFastLength = IntParameter(2, 200, default=20, space="buy", optimize=True)

    @staticmethod
    def _sma(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(window=length, min_periods=length).mean()

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        # TradingView EMA’ya yakın: adjust=False
        return series.ewm(span=length, adjust=False, min_periods=length).mean()

    @staticmethod
    def _crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    @staticmethod
    def _crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Bant hesapları (senin formülünle) ---
        src = self._sma(dataframe["close"], int(self.maLength.value))
        dataframe["src"] = src

        bb_len = int(self.bbLength.value)

        # a = sma(src^2, bbLength)
        a = self._sma(np.power(src, 2), bb_len)

        # b = (sum(src, bbLength)^2) / (bbLength^2)
        sum_src = src.rolling(window=bb_len, min_periods=bb_len).sum()
        b = np.power(sum_src, 2) / (bb_len ** 2)

        stdev = np.sqrt(np.maximum(a - b, 0.0))
        dataframe["stdev_custom"] = stdev

        dev = float(self.mult.value) * stdev
        basis = self._sma(src, bb_len)

        dataframe["basis"] = basis
        dataframe["upper"] = basis + dev
        dataframe["lower"] = basis - dev

        # --- EMA bulutu ---
        srcTwo = (dataframe["close"] + dataframe["low"] + dataframe["high"]) / 3.0
        dataframe["srcTwo"] = srcTwo

        fast_len = int(self.emaFastLength.value)
        slow_len = int(self.emaSlowLength.value)

        dataframe["emaFast"] = self._ema(srcTwo, fast_len)
        dataframe["emaSlow"] = self._ema(srcTwo, slow_len)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry: EMA fast, EMA slow'u yukarı keser
        dataframe.loc[
            self._crossed_above(dataframe["emaFast"], dataframe["emaSlow"]),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit: EMA fast, EMA slow'u aşağı keser
        dataframe.loc[
            self._crossed_below(dataframe["emaFast"], dataframe["emaSlow"]),
            "exit_long"
        ] = 1

        return dataframe
