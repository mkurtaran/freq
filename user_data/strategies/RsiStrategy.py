# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class RsiStrategy(IStrategy):
    """
    RSI Strategy
    Long when RSI < 35
    Short/Exit when RSI > 75
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # We set it to infinity/high to rely on the sell signal (RSI > 75)
    # But we can keep some safety nets or take profit levels if desired.
    minimal_roi = {
        "0": 0.20  # Safety net: sell if 20% profit
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    # User requested: Long < 35, Sell > 75
    buy_rsi = IntParameter(10, 40, default=35, space="buy")
    sell_rsi = IntParameter(60, 90, default=75, space="sell")

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "RSI": {
                    "rsi": {
                        "color": "red",
                        "fill_to": 0  # Opsiyonel: sıfırdan doldur
                    }
                }
            }
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe["rsi"].shift(1) > self.sell_rsi.value) &
                    (dataframe["volume"] > 0)
            ),
            "exit_long"] = 1

        return dataframe

