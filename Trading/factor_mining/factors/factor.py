import os
import pandas as pd
import numpy as np


class Factor:
    """因子基类，所有因子都应继承自此类"""
    def __init__(self):
        self.name = self._get_factor_name()  # 因子名称，默认为类名
        self.description = ""  # 因子描述
        self.category = ""  # 因子类别
        
    def _get_factor_name(self) -> str:
        """获取因子名称"""
        return self.__class__.__name__
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """生成因子值的抽象方法，子类必须实现"""
        raise NotImplementedError("子类必须实现generate方法")
    

class DynamicFactor(Factor):
    """动态生成的因子类"""
    def __init__(self, name, operation, features, window, lag):
        super().__init__()
        self.name = name
        self.operation = operation
        self.features = features
        self.window = window
        self.lag = lag
        self.description = f"动态生成因子: {operation}({', '.join(features)})"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 基础特征组合
        if len(self.features) == 1:
            base_feature = data[self.features[0]]
        else:
            # 组合多个特征
            base_feature = pd.Series(index=data.index)
            for i, feature in enumerate(self.features):
                if i == 0:
                    base_feature = data[feature]
                else:
                    # 随机选择加减乘除操作
                    op = np.random.choice(['+', '-', '*', '/'])
                    if op == '+':
                        base_feature = base_feature + data[feature]
                    elif op == '-':
                        base_feature = base_feature - data[feature]
                    elif op == '*':
                        base_feature = base_feature * data[feature]
                    else:
                        # 避免除以零
                        base_feature = base_feature / (data[feature] + 1e-10)
        
        # 应用操作
        if self.operation == 'rolling_mean':
            factor_value = base_feature.rolling(window=self.window).mean()
        elif self.operation == 'rolling_std':
            factor_value = base_feature.rolling(window=self.window).std()
        elif self.operation == 'pct_change':
            factor_value = base_feature.pct_change(periods=self.lag)
        elif self.operation == 'lag':
            factor_value = base_feature.shift(self.lag)
        elif self.operation == 'rank':
            factor_value = base_feature.rank(pct=True)
        elif self.operation == 'zscore':
            mean = base_feature.rolling(window=self.window).mean()
            std = base_feature.rolling(window=self.window).std()
            factor_value = (base_feature - mean) / (std + 1e-10)
        
        return factor_value

