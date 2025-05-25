import json
import os
import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy
from typing import Dict, Any

class MyStrategy(IStrategy):
    """
    在策略类初始化时加载factor_logic.json，动态注册所有有效因子。
    在populate_indicators中，利用这些因子逻辑生成因子数据。
    在populate_entry_trend和populate_exit_trend中，加载买卖信号生成逻辑（如zscore、quantile、ML模型），并据此生成买卖信号。
    支持ML模型时，自动加载pkl模型文件并预测信号。
    这样策略类就能无缝衔接你的因子挖掘pipeline，直接用于回测。
    """
    INTERFACE_VERSION = 3
    minimal_roi = {"0": 0.05}
    stoploss = -0.1
    timeframe = '5m'

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # 加载因子逻辑
        self.factor_logic_path = os.path.join("/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/factor_mining/factor_data/factor_logic.json")
        self.factors = self.load_factor_logic(self.factor_logic_path)
        # 预加载ML模型（如有）
        self.ml_models = self.load_ml_models()

    def load_factor_logic(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            factor_logic = json.load(f)
        return factor_logic

    def load_ml_models(self):
        models = {}
        for factor_name in self.factors:
            model_path = os.path.join(os.path.dirname(__file__), '../..', f'Trading/factor_mining/{factor_name}_ml_model.pkl')
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    models[factor_name] = pickle.load(f)
        return models

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 根据factor_logic.json动态生成所有有效因子
        for name, info in self.factors.items():
            params = info['params']
            # 这里只做简单示例：如有更复杂逻辑可扩展
            if 'window' in params and 'features' in params:
                feat = params['features'][0] if isinstance(params['features'], list) else params['features']
                if params.get('operation') == 'rolling_mean' and feat in dataframe:
                    dataframe[name] = dataframe[feat].rolling(window=params['window']).mean()
                elif params.get('operation') == 'rolling_std' and feat in dataframe:
                    dataframe[name] = dataframe[feat].rolling(window=params['window']).std()
                elif params.get('operation') == 'pct_change' and feat in dataframe:
                    dataframe[name] = dataframe[feat].pct_change(params['window'])
                elif params.get('operation') == 'lag' and feat in dataframe:
                    dataframe[name] = dataframe[feat].shift(params['lag'])
                elif params.get('operation') == 'zscore' and feat in dataframe:
                    mean = dataframe[feat].rolling(window=params['window']).mean()
                    std = dataframe[feat].rolling(window=params['window']).std()
                    dataframe[name] = (dataframe[feat] - mean) / (std + 1e-10)
                # 其他操作可按需扩展
        return dataframe

    def generate_signal(self, dataframe: pd.DataFrame, factor_name: str, method: str = 'zscore', buy_thr=1, sell_thr=-1, window=60):
        series = dataframe[factor_name]
        if method == 'zscore':
            mean = series.rolling(window).mean()
            std = series.rolling(window).std()
            z = (series - mean) / (std + 1e-10)
            signal = pd.Series(0, index=series.index)
            signal[z > buy_thr] = 1
            signal[z < sell_thr] = -1
            return signal
        elif method == 'quantile':
            q_high = series.rolling(window).quantile(0.8)
            q_low = series.rolling(window).quantile(0.2)
            signal = pd.Series(0, index=series.index)
            signal[series > q_high] = 1
            signal[series < q_low] = -1
            return signal
        elif method == 'ml' and factor_name in self.ml_models:
            model = self.ml_models[factor_name]
            X = series.values.reshape(-1, 1)
            preds = model.predict(X)
            return pd.Series(preds, index=series.index)
        else:
            return pd.Series(0, index=series.index)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 以第一个有效因子为例，实际可批量处理
        best_factor = list(self.factors.keys())[0]
        method = 'zscore'  # 可改为'quantile'或'ml'
        signal = self.generate_signal(dataframe, best_factor, method=method, buy_thr=1, sell_thr=-1, window=60)
        dataframe.loc[signal == 1, 'enter_long'] = 1
        dataframe.loc[signal == -1, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 简单示例：信号反向时平仓
        best_factor = list(self.factors.keys())[0]
        method = 'zscore'
        signal = self.generate_signal(dataframe, best_factor, method=method, buy_thr=1, sell_thr=-1, window=60)
        dataframe.loc[signal == -1, 'exit_long'] = 1
        dataframe.loc[signal == 1, 'exit_short'] = 1
        return dataframe



"""
import pickle
import pandas as pd

# 1. 加载新数据并生成所有best_factors特征（用你的engine流程）
# engine.data = pd.read_feather("新数据.feather")
# engine.load_factor_logic()
# engine.generate_factors(parallel=False)
X_new = engine.data[best_factors].values

# 2. 加载模型
with open("all_factors_ml_model.pkl", "rb") as f:
    model = pickle.load(f)

# 3. 预测买卖信号
pred_signal = model.predict(X_new)  # 0/1 或 -1/1
engine.data["ml_signal"] = pred_signal
"""