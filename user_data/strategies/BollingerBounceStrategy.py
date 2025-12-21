# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
)
import talib.abstract as ta
from technical import qtpylib

class BollingerBounceStrategy(IStrategy):
    """
    Bollinger Bounce Strategy
    - Buy: Price touches/crosses Upper Band 3 times in the last N candles (Trend Breakout)
    - Sell: Price touches/crosses Lower Band 3 times in the last N candles (Trend Reversal)
    """
    INTERFACE_VERSION = 3

    # Timeframe
    timeframe = "5m"

    # ROI (Return on Investment) - Set high to let the trend run
    minimal_roi = {
        "0": 0.50  # Sell if 50% profit
    }

    # Stoploss
    stoploss = -0.10

    # Trailing stoploss - Highly recommended for trend following
    # İz süren stop (Trailing Stop) ayarları:
    trailing_stop = True
    # Fiyatın %1 gerisinden takip et
    trailing_stop_positive = 0.01
    # %2 kara geçince takibi başlat
    trailing_stop_positive_offset = 0.02
    # Sadece offset (hedef kar) seviyesine ulaşınca iz sürmeyi aktifleştir
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 30

    # Parameters
    # Lookback window to count touches (e.g., check last 5 candles)
    lookback_window = IntParameter(3, 10, default=5, space="buy")
    # Satis (Exit) icin ayri pencere ayari, boylece hyperopt bagimsiz optimize edebilir
    exit_lookback_window = IntParameter(3, 10, default=5, space="sell")
    
    # Bollinger Band settings
    bb_window = 20
    bb_std = 2.0

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "bb_upperband": {"color": "green"},
                "bb_middleband": {"color": "blue"},
                "bb_lowerband": {"color": "red"},
            },
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), 
            window=self.bb_window, 
            stds=self.bb_std
        )
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']

        # Check if High touched or crossed Upper Band
        # 1 if High >= Upper Band, else 0
        dataframe['touched_upper'] = np.where(dataframe['high'] >= dataframe['bb_upperband'], 1, 0)

        # Check if Low touched or crossed Lower Band
        # 1 if Low <= Lower Band, else 0
        dataframe['touched_lower'] = np.where(dataframe['low'] <= dataframe['bb_lowerband'], 1, 0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate sum of touches in the last N candles
        # rolling(window).sum() counts how many times it happened in the window
        dataframe['upper_touch_count'] = dataframe['touched_upper'].rolling(window=self.lookback_window.value).sum()

        dataframe.loc[
            (
                # If touched upper band 3 or more times in the lookback window
                (dataframe['upper_touch_count'] >= 3) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate sum of touches in the last N candles
        dataframe['lower_touch_count'] = dataframe['touched_lower'].rolling(window=self.exit_lookback_window.value).sum()

        dataframe.loc[
            (
                # If touched lower band 3 or more times in the lookback window
                (dataframe['lower_touch_count'] >= 3) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
