# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy import (DecimalParameter,
                                IntParameter, IStrategy)
from pandas import DataFrame


# --------------------------------

class NewStrategy(IStrategy):
    """
    LSVT (Liquidation & Volume Trend) Strategy
    ProTrade AI V23'ten Freqtrade'e optimize edilmiştir.

    Mantık:
    - Fiyat Bollinger alt bandının altına sarkıp üstünde kapatmalı (Ayı tuzağı/Likidasyon temizliği).
    - Bu hareket sırasında hacim, ortalama hacmin X katı olmalı (Kurumsal giriş).
    - RSI aşırı alım bölgesinde olmamalı.
    """

    # Strateji Ayarları
    INTERFACE_VERSION = 3
    timeframe = '15m'
    stoploss = -0.10  # %10 stop (Opsiyonel: ROI ve Trailing Stop kullanılabilir)

    # ProTrade AI Parametreleri (Optimize edilebilir)
    buy_bb_len = IntParameter(10, 50, default=20, space="buy")
    buy_bb_mult = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space="buy")
    buy_vol_len = IntParameter(10, 40, default=20, space="buy")
    buy_vol_factor = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space="buy")
    buy_rsi_max = IntParameter(40, 60, default=55, space="buy")

    sell_rsi_min = IntParameter(40, 60, default=45, space="sell")

    # ROI Ayarları (HTML'deki TP mantığına benzer)
    minimal_roi = {
        "0": 0.05,  # %5 kar hedefi
        "30": 0.03,
        "60": 0.01
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bantları
        for val in self.buy_bb_len.range:
            bb = qtpylib.bollinger_bands(dataframe['close'], window=val, stds=self.buy_bb_mult.value)
            dataframe[f'bb_lower_{val}'] = bb['lower']
            dataframe[f'bb_upper_{val}'] = bb['upper']
            dataframe[f'bb_mid_{val}'] = bb['mid']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Hacim Hareketli Ortalaması (SMA)
        dataframe['vol_avg'] = dataframe['volume'].rolling(window=self.buy_vol_len.value).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb_lower = dataframe[f'bb_lower_{self.buy_bb_len.value}']

        dataframe.loc[
            (
                # Fiyatın düşük seviyesi BB alt bandının altında (Likidasyon iğnesi)
                    (dataframe['low'] < bb_lower) &
                    # Mum kapanışı BB alt bandının üstünde (Geri dönüş/Recovery)
                    (dataframe['close'] > bb_lower) &
                    # Hacim Patlaması: Mevcut hacim > Ortalama Hacim * Faktör
                    (dataframe['volume'] > (dataframe['vol_avg'] * self.buy_vol_factor.value)) &
                    # RSI Filtresi: Aşırı alımda değilken giriş
                    (dataframe['rsi'] < self.buy_rsi_max.value) &
                    (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb_upper = dataframe[f'bb_upper_{self.buy_bb_len.value}']

        dataframe.loc[
            (
                # Fiyatın yüksek seviyesi BB üst bandının üstünde
                    (dataframe['high'] > bb_upper) &
                    # Mum kapanışı BB üst bandının altında (Red yeme)
                    (dataframe['close'] < bb_upper) &
                    # Hacim Patlaması (Satış baskısı)
                    (dataframe['volume'] > (dataframe['vol_avg'] * self.buy_vol_factor.value)) &
                    # RSI Filtresi: Momentum varken çıkış
                    (dataframe['rsi'] > self.sell_rsi_min.value)
            ),
            'exit_long'] = 1

        return dataframe

    # --- Futures (Short) İşlemler İçin Opsiyonel ---
    def populate_entry_short(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb_upper = dataframe[f'bb_upper_{self.buy_bb_len.value}']

        if self.config.get('trading_mode', 'spot') == 'futures':
            dataframe.loc[
                (
                        (dataframe['high'] > bb_upper) &
                        (dataframe['close'] < bb_upper) &
                        (dataframe['volume'] > (dataframe['vol_avg'] * self.buy_vol_factor.value)) &
                        (dataframe['rsi'] > self.sell_rsi_min.value)
                ),
                'enter_short'] = 1
        return dataframe

    def populate_exit_short(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb_lower = dataframe[f'bb_lower_{self.buy_bb_len.value}']

        if self.config.get('trading_mode', 'spot') == 'futures':
            dataframe.loc[
                (
                        (dataframe['low'] < bb_lower) &
                        (dataframe['close'] > bb_lower) &
                        (dataframe['volume'] > (dataframe['vol_avg'] * self.buy_vol_factor.value)) &
                        (dataframe['rsi'] < self.buy_rsi_max.value)
                ),
                'exit_short'] = 1
        return dataframe
