import numpy as np
from sklearn.ensemble import RandomForestClassifier


class TradingStrategy:
    def __init__(self, factors):
        """
        初始化交易策略类。

        参数:
            factors: 挖掘出的因子列表。
        """
        self.factors = factors

    def signal_threshold(self, data, method='mean_std', threshold=1):
        """
        根据信号设定买卖阈值。

        参数:
            data: 包含因子的数据框。
            method: 阈值方法，可选 'mean_std' 或 'quantile'。
            threshold: 阈值参数，均值标准差法为标准差倍数，分位数法为分位数值。

        返回:
            买卖信号数组。
        """
        if method == 'mean_std':
            mean = data[self.factors].mean()
            std = data[self.factors].std()
            upper = mean + threshold * std
            lower = mean - threshold * std
            signal = np.where(data[self.factors] > upper, 1, np.where(data[self.factors] < lower, -1, 0))
        elif method == 'quantile':
            upper = data[self.factors].quantile(1 - threshold)
            lower = data[self.factors].quantile(threshold)
            signal = np.where(data[self.factors] > upper, 1, np.where(data[self.factors] < lower, -1, 0))
        else:
            raise ValueError('不支持的方法，请选择 mean_std 或 quantile')
        return signal

    def train_model(self, X, y):
        """
        根据因子训练机器学习模型。

        参数:
            X: 特征矩阵，包含因子数据。
            y: 目标变量。

        返回:
            训练好的机器学习模型。
        """
        model = RandomForestClassifier()
        model.fit(X, y)
        return model