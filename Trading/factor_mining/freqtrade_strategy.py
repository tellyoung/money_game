from freqtrade.strategy import IStrategy
import pandas as pd
from trading_strategy import TradingStrategy


class FactorBasedStrategy(IStrategy):
    # 定义策略参数
    INTERFACE_VERSION = 3
    timeframe = '1h'
    minimal_roi = {"0": 0.1}
    stoploss = -0.1
    trailing_stop = False

    def __init__(self, config):
        super().__init__(config)
        # 假设已经有挖掘出的因子列表
        self.factors = ['RandomFactor_1', 'RandomFactor_2']
        self.strategy = TradingStrategy(self.factors)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        填充指标到数据框。

        参数:
            dataframe: 包含市场数据的数据框。
            metadata: 元数据字典。

        返回:
            填充指标后的数据框。
        """
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        填充买入信号到数据框。

        参数:
            dataframe: 包含市场数据的数据框。
            metadata: 元数据字典。

        返回:
            填充买入信号后的数据框。
        """
        signal = self.strategy.signal_threshold(dataframe, method='mean_std', threshold=1)
        dataframe.loc[signal > 0, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        填充卖出信号到数据框。

        参数:
            dataframe: 包含市场数据的数据框。
            metadata: 元数据字典。

        返回:
            填充卖出信号后的数据框。
        """
        signal = self.strategy.signal_threshold(dataframe, method='mean_std', threshold=1)
        dataframe.loc[signal < 0, 'exit_long'] = 1
        return dataframe