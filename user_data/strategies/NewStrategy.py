import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class NewStrategy(IStrategy):

    timeframe = "5m"
    startup_candle_count = 20
    process_only_new_candles = True
    can_short = False

    # =====================
    # ROI / STOPLOSS
    # =====================
    minimal_roi = {
        "0": 0.01,
        "30": 0.005,
        "60": 0
    }

    stoploss = -0.03
    trailing_stop = False

    # =====================
    # FREQAI SETTINGS
    # =====================
    use_freqai = True

    # =====================
    # INDICATORS / FEATURES
    # =====================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        upper, middle, lower = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upperband'] = upper
        dataframe['bb_middleband'] = middle
        dataframe['bb_lowerband'] = lower

        # ===== FREQAI FEATURES =====
        dataframe['f_rsi'] = dataframe['rsi']
        dataframe['f_bb_dist'] = (dataframe['close'] - dataframe['bb_lowerband']) / dataframe['close']

        return dataframe

    # =====================
    # ENTRY
    # =====================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if 'do_predict' not in dataframe.columns:
            return dataframe

        dataframe.loc[
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['prediction'] > 0.51)
            ),
            'enter_long'
        ] = 1

        return dataframe

    # =====================
    # EXIT
    # =====================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if 'prediction' in dataframe.columns:
            dataframe.loc[
                (dataframe['prediction'] < 0.49),
                'exit_long'
            ] = 1
        else:
            # Fallback exit condition if no prediction is available
            dataframe.loc[
                (dataframe['close'] > dataframe['bb_middleband']),
                'exit_long'
            ] = 1

        return dataframe
