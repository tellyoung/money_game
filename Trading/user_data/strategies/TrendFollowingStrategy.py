from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy

import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)


class TrendFollowingStrategy(IStrategy):
    INTERFACE_VERSION: int = 3
    can_short = True
    timeframe = "5m"

    # ROI table:
    minimal_roi = {"0": 0.15, "30": 0.1, "60": 0.05}
    # minimal_roi = {"0": 1}

    # Stoploss:
    stoploss = -0.265

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Hyperoptable parameters
    trend_ema_span = IntParameter(5, 50, default=20, space="buy", optimize=True)
    obv_threshold = DecimalParameter(-1.0, 1.0, default=0.1, space="buy", optimize=True)

    # Sell-related Hyperoptable parameters
    sell_rsi_threshold = IntParameter(30, 70, default=50, space="sell", optimize=True)
    sell_rsi_timeperiod = IntParameter(5, 25, default=14, space="sell", optimize=True)
    sell_obv_multiplier = DecimalParameter(0.8, 1.2, default=1.0, space="sell", optimize=True)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate OBV
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        
        # Add your trend following indicators here
        dataframe['trend'] = dataframe['close'].ewm(span=self.trend_ema_span.value, adjust=False).mean()
        
        # Add RSI indicator
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.sell_rsi_timeperiod.value)
        
        return dataframe
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following buy signals here
        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) & 
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1) * (1 + self.obv_threshold.value)), 
            'enter_long'] = 1
        
        # Add your trend following sell signals here
        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) & 
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1) * (1 - self.obv_threshold.value)), 
            'enter_short'] = 1
        
        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following exit signals for long positions here
        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) & 
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1) * self.sell_obv_multiplier.value) &
            (dataframe['rsi'] > self.sell_rsi_threshold.value), 
            'exit_long'] = 1
        
        # Add your trend following exit signals for short positions here
        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) & 
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1) * self.sell_obv_multiplier.value) &
            (dataframe['rsi'] < self.sell_rsi_threshold.value),
            'exit_short'] = 1

        return dataframe
