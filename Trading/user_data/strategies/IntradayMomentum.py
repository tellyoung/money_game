from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy

import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy

class IntradayMomentum(IStrategy):
    INTERFACE_VERSION: int = 3
    can_short = True

    timeframe = "5m"

    # ROI table:
    minimal_roi = {"0": 0.15, "30": 0.1, "60": 0.05}

    # Stoploss:
    stoploss = -0.265

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # 目前最优0.8
    band_mult = 0.8

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators required for the strategy.
        """
        # Calculate VWAP
        hlc = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['vwap'] = (hlc * dataframe['volume']).cumsum() / dataframe['volume'].cumsum()

        # Calculate rolling mean and sigma
        dataframe['move_open'] = (dataframe['close'] / dataframe['open'] - 1).abs()
        dataframe['move_open_rolling_mean'] = dataframe['move_open'].rolling(window=14, min_periods=13).mean()
        dataframe['sigma_open'] = dataframe['move_open_rolling_mean'].shift(1)

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals based on the strategy logic.
        """
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Calculate upper and lower bands
        dataframe['UB'] = dataframe['open'] * (1 + self.band_mult * dataframe['sigma_open'])
        dataframe['LB'] = dataframe['open'] * (1 - self.band_mult * dataframe['sigma_open'])

        # Long entry signal: Price > UB and Price > VWAP
        dataframe.loc[
            (dataframe['close'] > dataframe['UB']) & (dataframe['close'] > dataframe['vwap']),
            'enter_long'
        ] = 1

        # Short entry signal: Price < LB and Price < VWAP
        dataframe.loc[
            (dataframe['close'] < dataframe['LB']) & (dataframe['close'] < dataframe['vwap']),
            'enter_short'
        ] = 1

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Exit long when price drops below VWAP
        dataframe.loc[
            (dataframe['close'] < dataframe['vwap']),
            'exit_long'
        ] = 1

        # Exit short when price rises above VWAP
        dataframe.loc[
            (dataframe['close'] > dataframe['vwap']),
            'exit_short'
        ] = 1

        return dataframe



