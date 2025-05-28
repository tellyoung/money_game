import sys
import importlib
import json
import os
import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy
from typing import Dict, Any
import logging  # 添加logger属性，兼容因子加载等日志输出

# 将factor_mining目录加入sys.path，确保importlib能找到factors包。
factor_mining_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../factor_mining'))
sys.path.insert(0, factor_mining_path)

class MyStrategy(IStrategy):
    """
    在策略类初始化时加载factor_logic.json，动态注册所有有效因子。
    在populate_indicators中，利用这些因子逻辑生成因子数据。
    在populate_entry_trend和populate_exit_trend中，加载买卖信号生成逻辑（如zscore、quantile、ML模型），并据此生成买卖信号。
    支持ML模型时，自动加载pkl模型文件并预测信号。
    这样策略类就能无缝衔接你的因子挖掘pipeline，直接用于回测。
    """
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"

    # ROI table (hyperoptable)
    minimal_roi = {
        "0": 0.05,  # Default values, will be overridden by hyperopt
        "30": 0.03,
        "60": 0.01
    }

    # Stoploss (hyperoptable)
    stoploss = -0.02  # Default value, will be overridden by hyperopt

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.logger = logging.getLogger('MyStrategy')
        # 加载因子逻辑
        self.factors = {}
        self.best_factor_logic_path = os.path.join("/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/factor_mining/factor_data/factor_logic.json")
        self.load_factor_logic(self.best_factor_logic_path)

        # 加载best_factors顺序
        best_factors_path = os.path.join("/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/factor_mining/factor_data/best_factors.json")
        with open(best_factors_path, 'r', encoding='utf-8') as f:
            self.best_factors = json.load(f)

        # 预加载联合多因子ML模型（如有）
        self.all_factors_ml_model = self.load_factors_ml_model()

    def load_factor_logic(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            factor_logic = json.load(f)
        # 记录原始顺序
        self.factor_order = list(factor_logic.keys())
        # 按类型分组
        self.basic_factors = {}
        self.advanced_factors = {}
        self.parametric_dynamic_factors = {}
        self.dynamic_factors = {}
        for name, info in factor_logic.items():
            try:
                module = importlib.import_module(info["module"])
                cls = getattr(module, info["class"])
                factor = cls(**info["params"])
                # 分类
                if info["module"].endswith("basic_factors"):
                    self.basic_factors[name] = factor
                elif info["module"].endswith("advanced_factors"):
                    self.advanced_factors[name] = factor
                elif info["class"] == "ParametricDynamicFactor":
                    self.parametric_dynamic_factors[name] = factor
                elif info["class"] == "DynamicFactor":
                    self.dynamic_factors[name] = factor
                else:
                    self.dynamic_factors[name] = factor  # 兜底
                self.logger.info(f"因子加载成功: {name}")
            except Exception as e:
                self.logger.error(f"因子加载失败: {name}, 错误: {e}")
        # 合并所有因子，便于后续兼容
        self.factors = {**self.basic_factors, **self.advanced_factors, **self.parametric_dynamic_factors, **self.dynamic_factors}
        self.logger.info(f"已加载因子逻辑: {list(self.factors.keys())}")

    def load_factors_ml_model(self):
        # 支持加载联合多因子ML模型
        model_path = os.path.join("/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/factor_mining/factor_data/ml_model.pkl")
        if os.path.exists(model_path):
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        按依赖顺序分阶段生成所有因子，最后按原始顺序排列。
        """
        # 1. basic_factors
        for name, factor in self.basic_factors.items():
            try:
                dataframe[name] = factor.generate(dataframe.copy())
                self.logger.info(f"因子生成成功: {name}")
            except Exception as e:
                self.logger.error(f"因子生成失败: {name}, 错误: {e}")
                dataframe[name] = 0
        # 2. advanced_factors
        for name, factor in self.advanced_factors.items():
            try:
                dataframe[name] = factor.generate(dataframe.copy())
                self.logger.info(f"因子生成成功: {name}")
            except Exception as e:
                self.logger.error(f"因子生成失败: {name}, 错误: {e}")
                dataframe[name] = 0
        # 3. parametric_dynamic_factors
        for name, factor in self.parametric_dynamic_factors.items():
            try:
                dataframe[name] = factor.generate(dataframe.copy())
                self.logger.info(f"因子生成成功: {name}")
            except Exception as e:
                self.logger.error(f"因子生成失败: {name}, 错误: {e}")
                dataframe[name] = 0
        # 4. dynamic_factors
        for name, factor in self.dynamic_factors.items():
            try:
                dataframe[name] = factor.generate(dataframe.copy())
                self.logger.info(f"因子生成成功: {name}")
            except Exception as e:
                self.logger.error(f"因子生成失败: {name}, 错误: {e}")
                dataframe[name] = 0

        return dataframe

    def generate_signal(self, dataframe: pd.DataFrame, factor_name: str = None, method: str = 'zscore', buy_thr=1, sell_thr=-1, window=60, use_all_factors=False):
        """
        支持单因子和多因子ML推理。use_all_factors=True时，使用联合ML模型。
        """
        if use_all_factors and self.all_factors_ml_model is not None:
            # 多因子ML推理，严格按best_factors顺序取特征
            X = dataframe[self.best_factors].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            # 置信度阈值
            prob_thr = 0.9
            if hasattr(self.all_factors_ml_model, 'predict_proba'):
                probas = self.all_factors_ml_model.predict_proba(X)
                preds = self.all_factors_ml_model.predict(X)
                preds = np.where(preds == 0, -1, preds)

                # 只在最大概率大于阈值时才输出预测，否则为0
                max_proba = np.max(probas, axis=1)
                preds = np.where(max_proba >= prob_thr, preds, 0)
            else:
                preds = self.all_factors_ml_model.predict(X)
            return pd.Series(preds, index=dataframe.index)
        if factor_name is None:
            return pd.Series(0, index=dataframe.index)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 支持ML联合信号和单因子信号
        use_ml = self.all_factors_ml_model is not None
        if use_ml:
            signal = self.generate_signal(dataframe, use_all_factors=True)
        dataframe.loc[signal == 1, 'enter_long'] = 1
        dataframe.loc[signal == -1, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 支持ML联合信号和单因子信号
        use_ml = self.all_factors_ml_model is not None
        if use_ml:
            signal = self.generate_signal(dataframe, use_all_factors=True)
        dataframe.loc[signal == -1, 'exit_long'] = 1
        dataframe.loc[signal == 1, 'exit_short'] = 1
        return dataframe



