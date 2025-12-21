from datetime import datetime

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class HeikinAshiStrategy(IStrategy):
    """
    Heikin Ashi Swing Strategy v5 (Restored - Most Profitable)
    - Timeout süresi 24 saat.
    - Giriş eşikleri 40 (Stoch RSI & RSI).
    - Stop Loss 1.5 ATR.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'

    # ROI: Çok daha agresif zaman yönetimi
    minimal_roi = {
        "0": 100,  # Varsayılan: Sadece ATR TP ile çık
        "720": 0.005,  # 12 saat sonra %0.5 karda değilse çık
        "1440": -0.02  # 24 saat sonra %2 zarara kadar kabul et ve çık
    }

    # Hard Stop
    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # --------------------------------------------------------------------------
    # Parametreler
    # --------------------------------------------------------------------------

    # Giriş: Stochastic RSI Eşiği
    buy_stoch_rsi_entry = IntParameter(10, 60, default=40, space="buy", optimize=True)

    # Giriş: RSI Dip Eşiği
    buy_rsi_entry = IntParameter(20, 50, default=40, space="buy", optimize=True)

    # Giriş: ADX Eşiği
    buy_adx_min = IntParameter(10, 30, default=20, space="buy", optimize=True)

    # ATR Risk Yönetimi
    atr_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    # Stop Loss
    atr_sl_mult = DecimalParameter(1.5, 4.0, default=1.5, decimals=1, space="sell", optimize=True)
    # Take Profit
    atr_tp_mult = DecimalParameter(2.0, 6.0, default=3.5, decimals=1, space="sell", optimize=True)

    startup_candle_count: int = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe['fastk'] = stoch_rsi['fastk']
        dataframe['fastd'] = stoch_rsi['fastd']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_low'] = heikinashi['low']
        dataframe['ha_high'] = heikinashi['high']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ha_is_green'] = (dataframe['ha_close'] > dataframe['ha_open'])

        # Sinyal 1: Stochastic RSI Cross
        dataframe['signal_stoch'] = (
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_stoch_rsi_entry.value)
        )

        # Sinyal 2: RSI Dip Dönüşü
        dataframe['signal_rsi'] = (
                (dataframe['rsi'] < self.buy_rsi_entry.value) &
                (dataframe['rsi'] > dataframe['rsi'].shift(1))
        )

        dataframe.loc[
            (
                # ADX Filtresi
                    (dataframe['adx'] > self.buy_adx_min.value) &

                    # Sinyallerden biri
                    (dataframe['signal_stoch'] | dataframe['signal_rsi']) &

                    # Heikin Ashi Teyidi
                    (dataframe['ha_is_green'])
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_df = dataframe.loc[dataframe['date'] <= trade.open_date_utc]

        if not candle_df.empty:
            candle = candle_df.iloc[-1]
            entry_atr = candle['atr']
            stoploss_price = trade.open_rate - (entry_atr * self.atr_sl_mult.value)

            if trade.open_rate > 0:
                stoploss_pct = (stoploss_price - trade.open_rate) / trade.open_rate

                if stoploss_pct < -0.10:
                    return -0.10

                return stoploss_pct

        return -0.10

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):

        # Zaman Bazlı Çıkış (Time-based Exit)
        # 24 saat (1 gün) sonunda hala açıksa kapat
        if (current_time - trade.open_date_utc).days >= 1:
            return "timeout_exit"

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_df = dataframe.loc[dataframe['date'] <= trade.open_date_utc]

        if not candle_df.empty:
            candle = candle_df.iloc[-1]
            entry_atr = candle['atr']
            tp_price = trade.open_rate + (entry_atr * self.atr_tp_mult.value)

            if current_rate >= tp_price:
                return "atr_take_profit"

        return None
