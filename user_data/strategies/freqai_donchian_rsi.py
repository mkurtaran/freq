from typing import Dict

import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class FreqaiDonchianRsi(IStrategy):
    """
    FreqAI kullanan, RSI ve Donchian Channel'a dayalı bir stratejidir.
    FreqAI burada giriş filtresi (confirmation) olarak görev yapar.
    """

    # --- Zorunlu Temel Ayarlar ---
    INTERFACE_VERSION = 3

    # Timeframe: 1h
    timeframe = '1h'

    # Short izni aktif
    can_short = True

    # Startup candle count: RSI ve Donchian için yeterli veri (En az 100)
    startup_candle_count: int = 200

    # --- Strateji Parametreleri ---
    # RSI periyodu
    rsi_period = 14
    # RSI Eşikleri
    rsi_buy_threshold = 35
    rsi_sell_threshold = 65

    # Donchian periyodu
    donchian_period = 20
    # Fiyat bandın ne kadarına "yakın" sayılsın? (% cinsinden tolerans)
    # Örneğin: 0.02 => Fiyat Lower Band'ın %2 üstü veya Upper'ın %2 altı
    band_tolerance = 0.02

    # Minimal ROI (Kapalı pozisyonlar için, FreqAI exit sinyali ekleyebilirsiniz)
    minimal_roi = {
        "0": 0.05,  # %5 kar
        "240": 0.03,  # 4 saatte %3
        "720": 0.01  # 12 saatte %1
    }

    stoploss = -0.10

    # --- FreqAI Ayarları ---
    def feature_engineering_expand_all(
            self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        """
        FreqAI için tüm feature'ları genişleten fonksiyon.
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-volume-raw"] = dataframe["volume"]
        dataframe["%-price-raw"] = dataframe["close"]
        return dataframe

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        """
        FreqAI için temel feature'lar.
        """
        dataframe["pct-change"] = dataframe["close"].pct_change()
        dataframe["volume-raw"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        """
        ZORUNLU FONKSİYON.
        FreqAI modeline öğretelecek özel indikatörleri burada tanımlarız.
        RSI ve Donchian Channel burada dataframe'e eklenir.
        """
        # RSI (14 periyot)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        # Donchian Channel (Upper, Lower, Mid)
        dataframe["dc-upper"] = dataframe["high"].rolling(window=self.donchian_period).max()
        dataframe["dc-lower"] = dataframe["low"].rolling(window=self.donchian_period).min()
        dataframe["dc-mid"] = (dataframe["dc-upper"] + dataframe["dc-lower"]) / 2

        # Diğer yardımcı feature'lar (Model başarısı için ekstra veri)
        dataframe["rsi-fast"] = ta.RSI(dataframe, timeperiod=6)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        FreqAI'nin tahmin etmeye çalışacağı hedef (target) değişkeni.
        Burada '&s-close' (gelecekteki fiyat hareketi) hedeflenir.
        """
        dataframe["&s-close"] = (
                dataframe["close"]
                .shift(-self.freqai_info["feature_parameters"]["period"])
                .rolling(self.freqai_info["feature_parameters"]["period"])
                .mean()
                / dataframe["close"]
                - 1
        )
        return dataframe

    def set_freqai_params(self, data: Dict) -> Dict:
        """
        FreqAI konfigürasyon parametreleri.
        """
        return {
            "feature_parameters": {
                "include_corr_pairlist": False,
                "period": 14,  # Tahmin periyodu
                "shift_periods": 1,
                "DI_threshold": 0,
                "weight_factor": 0,
                "use_feature_engineering": True
            },
            "data_split_parameters": {
                "test_size": 0.2,
                "shuffle": False
            },
            "model_training_parameters": {
                # Kullanacağınız model tipine göre (XGBoost, LightGBM vb.) değişebilir
                "n_estimators": 800,
                "learning_rate": 0.05,
            }
        }

    # --- Strateji Giriş/Çıkış Mantığı ---

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Strateji sinyallerini üretmek için gerekli indikatörleri hesaplar.
        FreqAI verileri (prediction) dataframe'e otomatik olarak eklenecektir.
        """
        # Sinyal mantığı için RSI ve Donchian'ı tekrar hesaplıyoruz
        # (Feature engineering'de hesaplananlar eğitim için, buradakiler karar için)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        dataframe['dc-upper'] = dataframe['high'].rolling(window=self.donchian_period).max()
        dataframe['dc-lower'] = dataframe['low'].rolling(window=self.donchian_period).min()
        dataframe['dc-mid'] = (dataframe['dc-upper'] + dataframe['dc-lower']) / 2

        # FreqAI'nin çalışması için gerekli çağrı (Interface v3 için gerekebilir,
        # ancak FreqAI bu fonksiyonları otomatik hook'lar.
        # Manuel hesaplamalarımız yukarıda yapıldı.)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Giriş (Long/Short) koşullarını belirler.
        FreqAI burada filtre görevi görür.
        """
        # --- LONG Koşulları ---
        # 1. RSI < 35 (Aşırı Satım)
        # 2. Fiyat Donchian Lower Band'a yakın (Lower Band'ın %2 üzerinde veya altında)
        # 3. FreqAI tahmini pozitif (do_predict == 1 VE prediction > 0)

        long_condition = (
                (dataframe['rsi'] < self.rsi_buy_threshold) &
                (dataframe['close'] <= dataframe['dc-lower'] * (1 + self.band_tolerance)) &
                (dataframe['do_predict'] == 1) &
                (dataframe['prediction'] > 0)
        )

        dataframe.loc[long_condition, 'enter_long'] = 1

        # --- SHORT Koşulları ---
        # 1. RSI > 65 (Aşırı Alım)
        # 2. Fiyat Donchian Upper Band'a yakın
        # 3. FreqAI tahmini negatif (do_predict == 1 VE prediction < 0)

        short_condition = (
                (dataframe['rsi'] > self.rsi_sell_threshold) &
                (dataframe['close'] >= dataframe['dc-upper'] * (1 - self.band_tolerance)) &
                (dataframe['do_predict'] == 1) &
                (dataframe['prediction'] < 0)
        )

        dataframe.loc[short_condition, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Çıkış koşulları.
        FreqAI'nin yön değiştirmesi veya RSI'ın normalleşmesi çıkış tetikleyicisi olabilir.
        """
        # Long Exit
        # FreqAI negatife dönerse veya RSI çok aşırı alırsa çık
        dataframe.loc[
            (
                    (dataframe['prediction'] < 0) |
                    (dataframe['rsi'] > 70)
            ),
            'exit_long'
        ] = 1

        # Short Exit
        # FreqAI pozitife dönerse veya RSI çok aşırı satıma düşerse çık
        dataframe.loc[
            (
                    (dataframe['prediction'] > 0) |
                    (dataframe['rsi'] < 30)
            ),
            'exit_short'
        ] = 1

        return dataframe
