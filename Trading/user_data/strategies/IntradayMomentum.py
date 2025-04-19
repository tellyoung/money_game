from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy

import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)

class IntradayMomentum(IStrategy):
    INTERFACE_VERSION: int = 3
    can_short = True

    timeframe = "5m"

    # ROI table (hyperoptable)
    minimal_roi = {
        "0": 0.05,  # Default values, will be overridden by hyperopt
        "30": 0.03,
        "60": 0.01
    }

    # Stoploss (hyperoptable)
    stoploss = -0.01  # Default value, will be overridden by hyperopt

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False


    # Hyperoptable parameters
    band_mult_UB = DecimalParameter(0.3, 5., default=0.8, space="buy", optimize=True)
    band_mult_LB = DecimalParameter(0.3, 5., default=0.8, space="buy", optimize=True)
    sell_threshold = DecimalParameter(0.95, 1.05, default=1.0, space="sell", optimize=True)


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
        dataframe['UB'] = dataframe['open'] * (1 + self.band_mult_UB.value * dataframe['sigma_open'])
        dataframe['LB'] = dataframe['open'] * (1 - self.band_mult_LB.value * dataframe['sigma_open'])

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

        # print(dataframe[['close', 'UB', 'LB', 'vwap', "enter_long", "enter_short"]].tail(100))
        # print(dataframe.loc[dataframe['enter_long'] == 1, ['date', 'close', 'UB', 'LB', 'vwap', "enter_long", "enter_short"]]).tail(100))
        # print(dataframe.loc[dataframe['enter_short'] == 1, ['date', 'close', 'UB', 'LB', 'vwap', "enter_long", "enter_short"]]).tail(100))

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # dataframe.loc[
        #     (dataframe['close'] < dataframe['LB']) & (dataframe['close'] < dataframe['vwap']),
        #     'exit_long'
        # ] = 1

        # dataframe.loc[
        #     (dataframe['close'] > dataframe['UB']) & (dataframe['close'] > dataframe['vwap']),
        #     'exit_short'
        # ] = 1

        # Exit long when price drops below VWAP
        # dataframe.loc[
        #     (dataframe['close'] < dataframe['vwap']),
        #     'exit_long'
        # ] = 1

        # # Exit short when price rises above VWAP
        # dataframe.loc[
        #     (dataframe['close'] > dataframe['vwap']),
        #     'exit_short'
        # ] = 1

        # Exit long when price drops below VWAP * sell_threshold
        dataframe.loc[
            (dataframe['close'] < dataframe['vwap'] * self.sell_threshold.value),
            'exit_long'
        ] = 1

        # Exit short when price rises above VWAP * sell_threshold
        dataframe.loc[
            (dataframe['close'] > dataframe['vwap'] * self.sell_threshold.value),
            'exit_short'
        ] = 1
        return dataframe




