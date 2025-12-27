# -----------------------------
# FreqAI ile Entegre Freqtrade Stratejisi
# Timeframe: 1h | can_short = True
# -----------------------------
from functools import reduce

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy


class FreqAIFreqtradeStrategy(IStrategy):
    """
    FreqAI kullanarak RSI ve Donchian Channel ile çalışan strateji.
    FreqAI sadece filtre görevi görür.
    """

    # -------------------------
    # ZORUNLU AYARLAR
    # -------------------------
    timeframe = '15m'  # Zaman aralığı
    can_short = True  # Short işlemlere izin ver
    startup_candle_count = 200  # Başlangıç için minimum mum sayısı (≥100)

    # Stoploss: Dinamik olarak ATR'ye göre ayarlanacak
    stoploss = -0.10

    # ROI (Return On Investment) tablosu - Dinamik olarak ATR'ye göre ayarlanacak
    minimal_roi = {
        "0": 0.05,
        "30": 0.02,
        "60": 0.01
    }

    # FreqAI aktif ediliyor
    use_freqai = True

    # ATR Oranı Ayarları
    atr_multiplier_stoploss = 2.0  # Stoploss = Close - (ATR * 2.0)
    atr_multiplier_takeprofit = 3.0  # TakeProfit = Close + (ATR * 3.0)

    # -------------------------
    # FREQAI YAPILANDIRMA (Opsiyonel ama önerilir)
    # -------------------------
    freqai_info = {
        "model_training_hours": 24,  # Modelin ne sıklıkla eğitileceği (saat)
        "train_period_days": 7,  # Eğitim için kullanılan geçmiş gün sayısı
        "identifier": "freqai_RSI_Donchian",  # Model kimliği
        "freqaimodel": "LightGBMRegressor",  # Model sınıfı
        "live_retrain_hours": 12,  # Canlı ortamda yeniden eğitim aralığı
        "conv_width": 24,  # Tahmin penceresi (1h timeframe → 24 mum = 1 gün)
        "feature_parameters": {
            "include_timeframes": ["1h"],
            "include_corr_pairlist": [],
            "label_period_candles": 24,
            "include_shifted_candles": 2,
            "indicator_periods_candles": [10, 20],
            "plot_feature_importances": 0,
        },
        "data_split_parameters": {
            "test_size": 0.25,
            "random_state": 42,
        },
        "model_training_parameters": {
        }
    }

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    def feature_engineering_standard(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Standart feature'ları oluşturur.
        RSI (14), EMA, Donchian Channel ve momentum indikatörleri eklenir.
        """
        # RSI (14) hesapla
        dataframe['%rsi'] = ta.RSI(dataframe, timeperiod=14)

        # EMA (9 ve 21) - Trend filtresi
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)

        # Donchian Channel (20 periyot) hesapla
        donchian_period = 20
        dataframe['donchian_upper'] = ta.MAX(dataframe['high'], timeperiod=donchian_period)
        dataframe['donchian_lower'] = ta.MIN(dataframe['low'], timeperiod=donchian_period)
        dataframe['donchian_mid'] = (dataframe['donchian_upper'] + dataframe['donchian_lower']) / 2

        # Fiyatın Donchian bandına yakınlığını ölçen özellikler
        dataframe['%price_pos_in_channel'] = (
                                                     dataframe['close'] - dataframe['donchian_lower']
                                             ) / (
                                                     dataframe['donchian_upper'] - dataframe['donchian_lower']
                                             )

        # MACD - Momentum filtresi
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_diff'] = macd['macdhist']

        # ATR - Volatilite filtresi
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Güvenlik için NaN değerleri doldur
        dataframe = dataframe.ffill().fillna(0)

        return dataframe

    # -------------------------
    # INDICATORS
    # -------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Tüm indikatörleri hesaplar.
        `feature_engineering_standard` fonksiyonunu çağırır.
        """
        dataframe = self.feature_engineering_standard(dataframe)

        # FreqAI otomatik olarak aşağıdaki kolonları dataframe'e ekler:
        # - `do_predict` : 1 = tahmin mevcut, 0 = tahmin yok
        # - `prediction` : Tahmin değeri (pozitif = alım sinyali, negatif = satım sinyali)

        return dataframe

    # -------------------------
    # ENTRY TREND
    # -------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Entry sinyalleri (Long ve Short) - Dengeli koşullar
        """
        # -------------------------
        # LONG (ALIM) SİNYALİ
        # -------------------------
        long_conditions = []

        # 1️⃣ RSI < 30 (Oversold/Neutral)
        long_conditions.append(dataframe['%rsi'] < 30)

        # 2️⃣ Fiyat Donchian lower banda yakın (price_pos_in_channel < 0.3)
        long_conditions.append(dataframe['%price_pos_in_channel'] < 0.3)

        # 3️⃣ Uptrend filtresi: EMA 9 > EMA 21 VEYA MACD pozitif
        trend_filter_long = (dataframe['ema_9'] > dataframe['ema_21']) | (dataframe['macd'] > dataframe['macd_signal'])
        long_conditions.append(trend_filter_long)

        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_conditions),
                'enter_long'
            ] = 1
        else:
            dataframe['enter_long'] = 0

        # -------------------------
        # SHORT (SATIM) SİNYALİ
        # -------------------------
        short_conditions = []

        # 1️⃣ RSI > 55 (Overbought/Neutral)
        short_conditions.append(dataframe['%rsi'] > 55)

        # 2️⃣ Fiyat Donchian upper banda yakın (price_pos_in_channel > 0.7)
        short_conditions.append(dataframe['%price_pos_in_channel'] > 0.7)

        # 3️⃣ Downtrend filtresi: EMA 9 < EMA 21 VEYA MACD negatif
        trend_filter_short = (dataframe['ema_9'] < dataframe['ema_21']) | (dataframe['macd'] < dataframe['macd_signal'])
        short_conditions.append(trend_filter_short)

        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_conditions),
                'enter_short'
            ] = 1
        else:
            dataframe['enter_short'] = 0

        return dataframe

    # -------------------------
    # CUSTOM STOPLOSS (ATR Oranına Göre Dinamik)
    # -------------------------
    def custom_stoploss(self, pair: str, trade, current_time, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        ATR oranına göre dinamik stoploss hesaplar.
        Volatilite yüksekse stoploss geniş, düşükse dar olur.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe.empty:
            return self.stoploss

        # Son mum verilerini al
        last_candle = dataframe.iloc[-1]
        atr = last_candle['atr']
        close = last_candle['close']

        if atr == 0:
            return self.stoploss

        # ATR oranına göre stoploss hesapla
        # Long işlem için: Entry fiyatı - (ATR * multiplier)
        # Short işlem için: Entry fiyatı + (ATR * multiplier)
        if trade.is_short:
            stoploss_price = trade.open_rate + (atr * self.atr_multiplier_stoploss)
        else:
            stoploss_price = trade.open_rate - (atr * self.atr_multiplier_stoploss)

        # Stoploss yüzdesini hesapla
        stoploss_percent = (stoploss_price - current_rate) / current_rate

        # Minimum stoploss'u aşmadığından emin ol
        if stoploss_percent < self.stoploss:
            return self.stoploss

        return stoploss_percent

    # -------------------------
    # EXIT TREND
    # -------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Exit sinyalleri - ATR oranına göre kar al
        """
        # ATR oranına göre kar al seviyesi hesapla
        dataframe['atr_takeprofit'] = dataframe['close'] + (dataframe['atr'] * self.atr_multiplier_takeprofit)

        # Fiyat kar al seviyesine ulaştığında çık
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe
