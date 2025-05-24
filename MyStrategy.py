
# 自动生成的freqtrade策略
from freqtrade.strategy import IStrategy
import pandas as pd
import numpy as np

import pickle
import os
            

class MyStrategy(IStrategy):
    timeframe = "1h"
    minimal_roi = { "0": 0.1 }
    stoploss = -0.1
    trailing_stop = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 可在此处添加自定义因子或特征
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["buy"] = 0

        # RandomFactor_3 ML模型信号
        model_path = "RandomFactor_3_ml_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            X = dataframe["RandomFactor_3"].values.reshape(-1, 1)
            preds = model.predict(X)
            dataframe["ml_pred_RandomFactor_3"] = preds
            dataframe.loc[dataframe["ml_pred_RandomFactor_3"] == 1, "buy"] = 1

        # RandomFactor_14 ML模型信号
        model_path = "RandomFactor_14_ml_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            X = dataframe["RandomFactor_14"].values.reshape(-1, 1)
            preds = model.predict(X)
            dataframe["ml_pred_RandomFactor_14"] = preds
            dataframe.loc[dataframe["ml_pred_RandomFactor_14"] == 1, "buy"] = 1

        # RandomFactor_1_quantile_0.1 ML模型信号
        model_path = "RandomFactor_1_quantile_0.1_ml_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            X = dataframe["RandomFactor_1_quantile_0.1"].values.reshape(-1, 1)
            preds = model.predict(X)
            dataframe["ml_pred_RandomFactor_1_quantile_0.1"] = preds
            dataframe.loc[dataframe["ml_pred_RandomFactor_1_quantile_0.1"] == 1, "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["sell"] = 0

        # RandomFactor_3 ML模型信号
        if "ml_pred_RandomFactor_3" in dataframe.columns:
            dataframe.loc[dataframe["ml_pred_RandomFactor_3"] == -1, "sell"] = 1

        # RandomFactor_14 ML模型信号
        if "ml_pred_RandomFactor_14" in dataframe.columns:
            dataframe.loc[dataframe["ml_pred_RandomFactor_14"] == -1, "sell"] = 1

        # RandomFactor_1_quantile_0.1 ML模型信号
        if "ml_pred_RandomFactor_1_quantile_0.1" in dataframe.columns:
            dataframe.loc[dataframe["ml_pred_RandomFactor_1_quantile_0.1"] == -1, "sell"] = 1

        return dataframe
