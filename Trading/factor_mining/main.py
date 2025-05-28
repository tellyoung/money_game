import os
import pandas as pd
from typing import List, Dict, Any, Callable, Union
from datetime import datetime, timedelta
import logging
import importlib
import inspect
from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
from factors.factor import Factor, ParametricDynamicFactor, DynamicFactor
warnings.filterwarnings('ignore')  # 忽略警告

# 新增导入
from ml_evaluator import MLEvaluator
from feature_generator import FeatureGenerator

import json
import pickle

class FactorMiningEngine:
    """因子挖掘引擎主类，负责协调整个因子挖掘流程"""
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)  # 加载配置文件
        self.data = None  # 存储从Feather文件加载的历史数据
        self.factors = {}  # 存储所有注册的因子
        self.factor_scores = {}  # 存储因子评估结果
        self.logger = self._setup_logger()  # 初始化日志系统
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件，获取数据路径、因子模块等参数"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            return {}
            
    def _setup_logger(self) -> logging.Logger:
        """配置日志系统，同时输出到控制台和文件"""
        logger = logging.getLogger('FactorMining')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('factor_mining.log')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 将处理器添加到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def load_data(self, data_path: str = None) -> None:
        """加载Feather格式的历史数据"""
        data_path = data_path or self.config.get('data_path')
        if not data_path:
            self.logger.error("数据路径未指定")
            return
            
        try:
            self.logger.info(f"正在加载数据: {data_path}")
            self.data = pd.read_feather(data_path)  # 读取Feather格式数据
            self.logger.info(f"数据加载完成，形状: {self.data.shape}")
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
    
    def discover_factors(self) -> None:
        """发现并注册所有可用因子，通过动态导入模块实现"""
        factor_modules = self.config.get('factor_modules', [])
        
        for module_name in factor_modules:
            try:
                module = importlib.import_module(module_name)
                print("加载factor类:", module_name)
                # 查找所有Factor子类
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and obj != Factor:
                        factor = obj()
                        self.factors[factor.name] = factor  # 注册因子
                        self.logger.info(f"已注册因子: {factor.name}")
            except Exception as e:
                self.logger.error(f"加载因子模块 {module_name} 失败: {e}")
    
    def generate_factors(self, parallel: bool = True) -> None:
        """生成所有注册因子，可选择并行计算以提高效率"""
        if self.data is None:
            self.logger.error("请先加载数据")
            return
            
        self.logger.info(f"开始生成因子，总数: {len(self.factors)}")
        
        if parallel and len(self.factors) > 1:
            # 使用多进程并行生成因子
            with Pool(processes=min(os.cpu_count(), len(self.factors))) as pool:
                results = pool.map(self._generate_single_factor, list(self.factors.items()))
                
            for factor_name, factor_data in results:
                self.data[factor_name] = factor_data
        else:
            # 顺序生成因子
            for factor_name, factor in self.factors.items():
                try:
                    self.data[factor_name] = factor.generate(self.data.copy())
                    self.logger.info(f"因子生成成功: {factor_name}")
                except Exception as e:
                    self.logger.error(f"因子生成失败: {factor_name}, 错误: {e}")
                    
        self.logger.info(f"因子生成完成，数据形状: {self.data.shape}")
    
    def _generate_single_factor(self, factor_tuple) -> tuple:
        """生成单个因子（用于并行处理）"""
        factor_name, factor = factor_tuple
        try:
            return factor_name, factor.generate(self.data.copy())
        except Exception as e:
            self.logger.error(f"因子生成失败: {factor_name}, 错误: {e}")
            return factor_name, None
    
    def calculate_returns(self, price_col: str = 'close', period: int = 1) -> None:
        """计算收益率并添加到数据中"""
        if self.data is None:
            self.logger.error("请先加载数据")
            return
            
        if price_col not in self.data.columns:
            self.logger.error(f"价格列 {price_col} 不存在")
            return
            
        # 计算简单收益率
        self.data[f'returns'] = self.data[price_col].pct_change(period).shift(-1)
        
        # 可以添加其他类型的收益率计算
        self.logger.info(f"已计算 {period} 天收益率，内容: returns_{period}d")
        
        # 更新目标列配置
        self.config['target_column'] = f'returns'
    
    def handle_missing_values(self, method: str = 'ffill', columns: List[str] = None, 
                             inplace: bool = True) -> pd.DataFrame:
        """处理数据中的缺失值
        
        参数:
            method: 处理方法，可选值包括 'ffill'（向前填充）、'bfill'（向后填充）、
                    'mean'（均值填充）、'median'（中位数填充）、'interpolate'（插值）
            columns: 需要处理的列名列表，默认为所有列
            inplace: 是否直接在原数据上修改
            
        返回:
            处理后的数据（如果inplace为False）
        """
        if self.data is None:
            self.logger.error("请先加载数据")
            return None
            
        # 选择需要处理的列
        if columns is None:
            columns = self.data.columns
            
        # 复制数据（如果不直接修改原数据）
        data = self.data if inplace else self.data.copy()
        
        self.logger.info(f"开始处理缺失值，方法: {method}，处理列数: {len(columns)}")
        
        # 根据指定方法处理缺失值
        if method == 'ffill':
            # 向前填充（使用前一个有效值）
            data[columns] = data[columns].fillna(method='ffill')
        elif method == 'bfill':
            # 向后填充（使用后一个有效值）
            data[columns] = data[columns].fillna(method='bfill')
        elif method == 'mean':
            # 均值填充
            data[columns] = data[columns].fillna(data[columns].mean())
        elif method == 'median':
            # 中位数填充
            data[columns] = data[columns].fillna(data[columns].median())
        elif method == 'interpolate':
            # 线性插值
            data[columns] = data[columns].interpolate(method='linear')
        else:
            self.logger.error(f"不支持的缺失值处理方法: {method}")
            return None if inplace else data
            
        # 检查处理后是否还有缺失值
        remaining_nan = data[columns].isna().sum().sum()
        if remaining_nan > 0:
            self.logger.warning(f"处理后仍有 {remaining_nan} 个缺失值，可能需要进一步处理")
            
        self.logger.info(f"缺失值处理完成，处理方法: {method}")

        # 替换无穷值为 NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        # 限制值的范围 只选择数值列
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        # 对数值列应用 clip 操作
        self.data[numeric_columns] = self.data[numeric_columns].clip(lower=-1e10, upper=1e10)
        if data.isnull().values.any():
            self.logger.warning("输入数据包含异常值，正在处理...")
            nan_rows = data.isnull().any(axis=1).sum()
            self.logger.info(f"输入数据中包含 NaN 的行数: {nan_rows}")
            data = data.fillna(0)  # 这里选择填充为 0

        return None if inplace else data
        
    def evaluate_factors(self, target_col: str = 'returns', handle_missing: bool = True, 
                        missing_method: str = 'ffill') -> None:
        """评估所有因子的有效性，增加了缺失值处理选项"""
        if self.data is None:
            self.logger.error("请先加载数据")
            return
            
        self.logger.info("开始评估因子有效性")
        
        # 确保目标列存在
        if target_col not in self.data.columns:
            self.logger.error(f"目标列 {target_col} 不存在")
            return
            
        # 过滤掉非因子列
        factor_cols = [col for col in self.data.columns if col in self.factors]
        
        # 处理缺失值
        if handle_missing:
            self.handle_missing_values(method=missing_method, columns=factor_cols + [target_col])
            
        # 使用时间序列交叉验证评估因子
        tscv = TimeSeriesSplit(n_splits=5)
        
        for factor_col in factor_cols:
            try:
                factor_scores = []
                
                for train_idx, test_idx in tscv.split(self.data):
                    train_data = self.data.iloc[train_idx]
                    test_data = self.data.iloc[test_idx]
                    
                    # 计算因子IC(Information Coefficient)：因子值与未来收益的相关性
                    ic = test_data[factor_col].corr(test_data[target_col])
                    factor_scores.append(ic)
                
                # 计算因子的平均IC和IR(Information Ratio)
                mean_ic = np.mean(factor_scores)  # 平均IC反映因子预测能力
                ir = mean_ic / np.std(factor_scores) if np.std(factor_scores) != 0 else np.nan  # IR衡量因子稳定性
                
                # 计算因子的AUC（将因子作为分类器）
                try:
                    labels = (test_data[target_col] > 0).astype(int)
                    auc = roc_auc_score(labels, test_data[factor_col])  # AUC衡量因子区分涨跌的能力
                except:
                    auc = np.nan
                
                self.factor_scores[factor_col] = {
                    'mean_ic': mean_ic,
                    'ir': ir,
                    'auc': auc,
                    'stability': np.std(factor_scores)  # 稳定性指标，标准差越小越好
                }
                
                self.logger.info(f"因子评估完成: {factor_col}, IC: {mean_ic:.4f}, IR: {ir:.4f}, AUC: {auc:.4f}")
                
            except Exception as e:
                self.logger.error(f"因子评估失败: {factor_col}, 错误: {e}")

    def analyze_factor_timeliness(self, window_size: int = 20) -> Dict[str, Dict[str, float]]:
        """分析因子时效性，评估因子有效性随时间的变化"""
        self.logger.info("开始分析因子时效性")
        
        timeliness_results = {}
        
        for factor_name in self.factor_scores:
            try:
                # 计算滚动IC，观察因子预测能力的稳定性
                rolling_ic = self.data[factor_name].rolling(window=window_size).corr(self.data['returns'])
                
                # 计算IC衰减，评估因子预测能力随时间的衰减速度
                ic_decay = self._calculate_ic_decay(factor_name, window=window_size)
                
                # 计算因子换手率，评估因子排序的稳定性
                turnover = self._calculate_factor_turnover(factor_name, window=window_size)
                
                timeliness_results[factor_name] = {
                    'rolling_ic_mean': rolling_ic.mean(),  # 滚动IC均值
                    'rolling_ic_std': rolling_ic.std(),    # 滚动IC标准差，反映稳定性
                    'ic_decay': ic_decay,                  # IC衰减指标
                    'turnover': turnover                   # 因子换手率
                }
                
                self.logger.info(f"因子时效性分析完成: {factor_name}")
                
            except Exception as e:
                self.logger.error(f"因子时效性分析失败: {factor_name}, 错误: {e}")
        
        return timeliness_results
    
    def _calculate_ic_decay(self, factor_name: str, window: int = 20) -> float:
        """计算因子IC衰减，评估因子预测能力随时间的衰减速度"""
        # 这里简化处理，实际应用中应使用更复杂的衰减模型
        decay = 0
        for i in range(1, 6):  # 计算未来5期的IC衰减
            shifted_factor = self.data[factor_name].shift(i)
            ic = shifted_factor.corr(self.data['returns'])
            decay += ic * (1 / i)  # 简单加权，近期影响更大
        
        return decay
    
    def _calculate_factor_turnover(self, factor_name: str, window: int = 20) -> float:
        """计算因子换手率，评估因子排序的稳定性"""
        # 计算因子排名变化
        ranks = self.data[factor_name].rank(pct=True)
        rank_changes = ranks.diff().abs()
        
        return rank_changes.rolling(window=window).mean().mean()
    
    def select_best_factors(self, top_n: int = 10) -> List[str]:
        """选择表现最好的因子，综合考虑IC、IR和AUC"""
        if not self.factor_scores:
            self.logger.error("请先评估因子")
            return []
            
        # 根据IC、IR和AUC的乘积排序因子，综合评估因子质量
        sorted_factors = sorted(
            self.factor_scores.items(), 
            key=lambda x: (x[1]['mean_ic'] * x[1]['ir'] * x[1]['auc']), 
            reverse=True
        )
        
        # 选择前N个因子
        best_factors = [factor[0] for factor in sorted_factors[:top_n]]
        self.logger.info(f"已选择最佳因子: {best_factors}")
        
        return best_factors
    
    def evaluate_factors_ml(self, target_col: str = 'returns') -> None:
        """使用机器学习方法评估因子有效性
        通过多种机器学习模型评估因子的预测能力
        """
        if self.data is None:
            self.logger.error("请先加载数据")
            return
            
        self.logger.info("开始使用机器学习方法评估因子有效性")
        
        # 准备因子列表
        factor_cols = [col for col in self.data.columns if col in self.factors]
        
        # 创建机器学习评估器
        ml_evaluator = MLEvaluator(self.config.get('ml_evaluator', {}))
        
        # 评估因子
        ml_evaluator.evaluate(self.data, factor_cols, target_col=target_col)
        
        # 获取评估结果
        self.ml_performance = ml_evaluator.performance
        self.ml_feature_importance = ml_evaluator.feature_importance
        
        # 获取共识因子
        self.consensus_factors = ml_evaluator.get_factor_consensus()
        
        self.logger.info(f"机器学习评估完成，共识因子: {self.consensus_factors}")
        
        return ml_evaluator
    
    def generate_new_features(self, factors: List[str] = None) -> None:
        """基于现有因子生成新的特征
        使用特征生成器自动创建大量候选因子
        
        参数:
            factors: 要用于生成新特征的因子列表，默认为所有已注册因子
        """
        if self.data is None:
            self.logger.error("请先加载数据")
            return
            
        # 默认使用所有已注册的因子
        factors = factors or list(self.factors.keys())
        self.logger.info(f"开始基于现有因子生成新特征，原始因子数量: {len(factors)}")
        
        # 创建特征生成器
        feature_generator = FeatureGenerator(self.config.get('feature_generator', {}))
        
        # 准备因子数据
        factor_data = {factor: self.data[factor] for factor in factors if factor in self.data.columns}
        
        # 生成新特征
        new_features = feature_generator.generate_all_features(factor_data)
        
        # 将新特征添加到数据中
        for feature_name, feature_data in new_features.items():
            self.data[feature_name] = feature_data
            # 假设feature_generator能返回操作、特征、窗口等参数
            # 这里以operation/features/window/lag为例，实际应从feature_generator获得
            params = feature_generator.get_feature_params(feature_name) if hasattr(feature_generator, 'get_feature_params') else {}
            operation = params.get('operation', 'mean')
            features = params.get('features', [feature_name])
            window = params.get('window', None)
            lag = params.get('lag', None)
            factor = DynamicFactor(feature_name, operation, features, window, lag)
            self.factors[feature_name] = factor
        self.logger.info(f"新特征生成完成，新增因子数量: {len(new_features)}")
        return list(new_features.keys())
    
    def save_results(self, output_path: str = None) -> None:
        """保存因子数据和评估结果到文件"""
        output_path = output_path or self.config.get('output_path', 'factor_data.feather')
        
        try:
            # 保存因子数据到Feather文件
            self.data.to_feather(output_path)
            self.logger.info(f"因子数据已保存到: {output_path}")
            
            # 保存因子评估结果到CSV文件
            scores_df = pd.DataFrame(self.factor_scores).T
            scores_df.to_csv(output_path.replace('.feather', '_scores.csv'))
            self.logger.info(f"因子评估结果已保存到: {output_path.replace('.feather', '_scores.csv')}")
            
            # 保存机器学习评估结果
            if hasattr(self, 'ml_performance'):
                ml_scores_df = pd.DataFrame(self.ml_performance).T
                ml_scores_df.to_csv(output_path.replace('.feather', '_ml_scores.csv'))
                self.logger.info(f"机器学习评估结果已保存到: {output_path.replace('.feather', '_ml_scores.csv')}")
                
            # 保存共识因子
            if hasattr(self, 'consensus_factors'):
                consensus_df = pd.DataFrame(self.consensus_factors, columns=['factor', 'rank'])
                consensus_df.to_csv(output_path.replace('.feather', '_consensus_factors.csv'))
                self.logger.info(f"共识因子已保存到: {output_path.replace('.feather', '_consensus_factors.csv')}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

    def save_factor_logic(self, filepath: str = None) -> None:
        """
        保存有效因子的生成逻辑（如参数、操作、特征名等）到JSON文件，便于后续复用。
        """
        filepath = filepath or self.config.get('factor_logic_path', 'factor_logic.json')
        factor_logic = {}
        for name, factor in self.factors.items():
            if hasattr(factor, "__dict__"):
                logic = factor.__dict__.copy()
                logic.pop("description", None)
                logic.pop("category", None)
                cls_name = factor.__class__.__name__
                module_name = factor.__class__.__module__
                # 针对不同动态因子类自动适配
                if cls_name in ("DynamicFactor", "ParametricDynamicFactor"):
                    factor_logic[name] = {
                        "class": cls_name,
                        "module": module_name,
                        "params": {
                            "name": factor.name,
                            "operation": getattr(factor, "operation", None),
                            "features": getattr(factor, "features", None),
                            "window": getattr(factor, "window", None),
                            "lag": getattr(factor, "lag", None)
                        }
                    }
                else:
                    factor_logic[name] = {
                        "class": cls_name,
                        "module": module_name,
                        "params": logic
                    }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(factor_logic, f, ensure_ascii=False, indent=2)
        self.logger.info(f"因子逻辑已保存到: {filepath}")
    
    def load_factor_logic(self, filepath: str = None) -> None:
        """
        加载因子生成逻辑，自动注册到self.factors，便于新数据直接复用。
        """
        filepath = filepath or self.config.get('factor_logic_path', 'factor_logic.json')
        with open(filepath, "r", encoding="utf-8") as f:
            factor_logic = json.load(f)
        for name, info in factor_logic.items():
            module = importlib.import_module(info["module"])
            cls = getattr(module, info["class"])
            factor = cls(**info["params"])
            self.factors[name] = factor
        self.logger.info(f"已加载因子逻辑: {list(self.factors.keys())}")

    def generate_signals(self, factor_name: str, method: str = None, buy_thr=1, sell_thr=-1, window=60, threshold_dict_path=None) -> pd.Series:
        """
        根据因子生成交易信号。
        method: "zscore"/"quantile"
        支持传入阈值字典路径，自动读取最优阈值。
        """
        if method is None:
            method = self.config.get("signal_method", "zscore")
        # 支持自动读取阈值
        if threshold_dict_path is not None:
            with open(threshold_dict_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            if factor_name in thresholds:
                buy_thr = thresholds[factor_name]["buy_thr"]
                sell_thr = thresholds[factor_name]["sell_thr"]
        factor_series = self.data[factor_name]
        if method == "zscore":
            mean = factor_series.rolling(window).mean()
            std = factor_series.rolling(window).std()
            z = (factor_series - mean) / (std + 1e-10)
            signal = pd.Series(0, index=factor_series.index)
            signal[z > buy_thr] = 1
            signal[z < sell_thr] = -1
            return signal
        elif method == "quantile":
            q_high = factor_series.rolling(window).quantile(0.8)
            q_low = factor_series.rolling(window).quantile(0.2)
            signal = pd.Series(0, index=factor_series.index)
            signal[factor_series > q_high] = 1
            signal[factor_series < q_low] = -1
            return signal
        else:
            raise ValueError("不支持的信号生成方法")
    
    def generate_all_signals(self, factor_names: list, method: str = "zscore", buy_thr=1, sell_thr=-1, window=60, ml_model_path=None):
        """
        一次性为所有有效因子生成买卖信号。
        method: "zscore"/"quantile"/"ml"
        - 传统方法：每个因子单独生成信号
        - ML方法：用所有因子联合训练模型，保存模型，预测信号
        返回：信号字典（仅用于本地分析，实盘/新数据请用模型推理）
        """
        signals = {}
        if method in ["zscore", "quantile"]:
            for fname in factor_names:
                signals[fname] = self.generate_signals(factor_name=fname, method=method, buy_thr=buy_thr, sell_thr=sell_thr, window=window)
        else:
            raise ValueError("不支持的信号生成方法")
        return signals
    
    def train_ml(self, factor_names: list, ml_model_path=None, train_start=None, train_end=None, val_start=None, val_end=None):
        """
        训练ML模型，支持自定义训练/验证时间区间。
        - factor_names: 用于训练的因子名列表
        - ml_model_path: 模型保存路径
        - train_start, train_end, val_start, val_end: 时间区间（字符串或datetime），如'2023-01-01'
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
        import pickle
        import pandas as pd
        import numpy as np
        import os
        # 1. 数据切分
        df = self.data.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # 去除时区，统一为tz-naive
            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
        # 训练集mask
        train_mask = np.ones(len(df), dtype=bool)
        if train_start is not None:
            train_start = pd.to_datetime(train_start)
            if hasattr(train_start, 'tzinfo') and train_start.tzinfo is not None:
                train_start = train_start.tz_localize(None)
            train_mask &= (df['date'] >= train_start)
        if train_end is not None:
            train_end = pd.to_datetime(train_end)
            if hasattr(train_end, 'tzinfo') and train_end.tzinfo is not None:
                train_end = train_end.tz_localize(None)
            train_mask &= (df['date'] <= train_end)
        # 验证集mask
        val_mask = np.ones(len(df), dtype=bool)
        if val_start is not None:
            val_start = pd.to_datetime(val_start)
            if hasattr(val_start, 'tzinfo') and val_start.tzinfo is not None:
                val_start = val_start.tz_localize(None)
            val_mask &= (df['date'] >= val_start)
        if val_end is not None:
            val_end = pd.to_datetime(val_end)
            if hasattr(val_end, 'tzinfo') and val_end.tzinfo is not None:
                val_end = val_end.tz_localize(None)
            val_mask &= (df['date'] <= val_end)
        # 验证集去除与训练集重叠部分
        val_mask &= ~train_mask
        # 如果未指定验证集区间，则val_mask为~train_mask（训练集以外的数据）
        if val_start is None and val_end is None:
            val_mask = ~train_mask

        # 2. 特征与标签
        X = df[factor_names].values
        y = (df[self.config.get('target_column', 'returns')] > 0).astype(int).values
        # 去除缺失
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        train_mask = train_mask[mask]
        val_mask = val_mask[mask]

        # 3. 训练
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, y_train = X[train_mask], y[train_mask]
        model.fit(X_train, y_train)
        # 保存模型
        ml_model_path = ml_model_path or self.config.get('ml_model_path', 'all_factors_ml_model.pkl')
        report_path_prefix = os.path.join(os.path.dirname(ml_model_path), os.path.splitext(os.path.basename(ml_model_path))[0])
        with open(ml_model_path, "wb") as f:
            pickle.dump(model, f)

        # 4. 评估
        def eval_and_save(X, y, prefix):
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            try:
                auc = roc_auc_score(y, model.predict_proba(X)[:,1])
            except Exception:
                auc = None
            cm = confusion_matrix(y, preds)
            report = classification_report(y, preds, output_dict=True)
            feat_imp = dict(zip(factor_names, model.feature_importances_))
            # 新增召回率和F1
            recall = report['1']['recall'] if '1' in report else None
            f1 = report['1']['f1-score'] if '1' in report else None
            # 日志
            self.logger.info(f"ML模型{prefix}准确率: {acc:.4f}, AUC: {auc if auc is not None else 'N/A'}, Recall: {recall if recall is not None else 'N/A'}, F1: {f1 if f1 is not None else 'N/A'}")
            self.logger.info(f"{prefix}混淆矩阵:\n{cm}")
            self.logger.info(f"{prefix}特征重要性: {feat_imp}")
            # 保存
            pd.DataFrame({"accuracy":[acc], "auc":[auc], "recall":[recall], "f1":[f1]}).to_csv(f"{report_path_prefix}_{prefix}_score.csv", index=False)
            pd.DataFrame(cm).to_csv(f"{report_path_prefix}_{prefix}_confusion_matrix.csv", index=False)
            pd.DataFrame(report).to_csv(f"{report_path_prefix}_{prefix}_classification_report.csv")
            pd.Series(feat_imp).to_csv(f"{report_path_prefix}_{prefix}_feature_importance.csv")
        
        # 训练集评估
        eval_and_save(X_train, y_train, "train")
        # 验证集评估
        if val_mask.sum() > 0:
            X_val, y_val = X[val_mask], y[val_mask]
            eval_and_save(X_val, y_val, "val")
        return model

    def find_best_signal_thresholds(self, factor_names, method="zscore", window=60, metric="sharpe", save_path="best_signal_thresholds.json"):
        """
        针对每个因子的分布自适应生成阈值网格，自动遍历买卖阈值组合，寻找最优阈值，并保存。
        metric: "sharpe"/"return"/"ic"
        """
        best_thresholds = {}
        for fname in factor_names:
            series = self.data[fname].dropna()
            best_score = -np.inf
            best_buy = None
            best_sell = None
            # 针对不同分布自适应生成阈值网格
            if method == "zscore":
                mean = series.mean()
                std = series.std()
                # 以均值±N*std为中心，N自适应（覆盖99%分布）
                N = max(2, min(4, int((series.max() - series.min()) / (std + 1e-8))))
                buy_range = np.arange(mean + 0.5*std, mean + N*std, std*0.1)
                sell_range = np.arange(mean - N*std, mean - 0.5*std, std*0.1)
            elif method == "quantile":
                # 用分位数做阈值
                quantiles = np.arange(0.7, 0.99, 0.05)
                buy_range = [series.quantile(q) for q in quantiles]
                sell_range = [series.quantile(1-q) for q in quantiles]
            else:
                # 默认用分位数
                quantiles = np.arange(0.7, 0.99, 0.05)
                buy_range = [series.quantile(q) for q in quantiles]
                sell_range = [series.quantile(1-q) for q in quantiles]
            for buy_thr in buy_range:
                for sell_thr in sell_range:
                    if buy_thr <= sell_thr:
                        continue  # 保证买阈值大于卖阈值
                    signal = self.generate_signals(factor_name=fname, method=method, buy_thr=buy_thr, sell_thr=sell_thr, window=window)
                    ret = (signal.shift(1) * self.data[self.config.get('target_column', 'returns')]).fillna(0)
                    if metric == "sharpe":
                        score = ret.mean() / (ret.std() + 1e-8)
                    elif metric == "return":
                        score = ret.sum()
                    elif metric == "ic":
                        score = signal.corr(self.data[self.config.get('target_column', 'returns')])
                    else:
                        score = 0
                    if score > best_score:
                        best_score = score
                        best_buy = float(buy_thr)
                        best_sell = float(sell_thr)
            best_thresholds[fname] = {"buy_thr": best_buy, "sell_thr": best_sell, "score": best_score}
            self.logger.info(f"{fname} 最优阈值: buy={best_buy}, sell={best_sell}, {metric}={best_score:.4f}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(best_thresholds, f, ensure_ascii=False, indent=2)
        self.logger.info(f"所有因子最优阈值已保存到: {save_path}")
        return best_thresholds


class FactorGenerator:
    """因子自动生成器，用于生成新的候选因子"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}  # 配置参数
        self.logger = self._setup_logger()  # 初始化日志系统
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('FactorGenerator')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('factor_generator.log')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 将处理器添加到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def generate_random_factors(self, data: pd.DataFrame, n_factors: int = 10) -> List[Factor]:
        """生成随机因子，通过组合基本特征和操作生成新因子"""
        self.logger.info(f"开始生成{self.config.get('random_factors_count', n_factors)}个随机因子")
        
        # 从基本特征中生成随机因子
        generated_factors = []
        basic_features = [col for col in data.columns if col not in ['date', 'symbol']]
        
        for i in range(n_factors):
            try:
                factor = self._create_random_factor(data, basic_features, f"RandomFactor_{i+1}")
                generated_factors.append(factor)
                self.logger.info(f"随机因子生成成功: {factor.name}")
            except Exception as e:
                self.logger.error(f"随机因子生成失败: {e}")
                
        return generated_factors
    
    def _create_random_factor(self, data: pd.DataFrame, basic_features: List[str], name: str) -> DynamicFactor:
        """创建单个随机因子，随机组合特征和操作"""
        # 随机选择操作和特征
        operations = ['rolling_mean', 'rolling_std', 'pct_change', 'lag', 'rank', 'zscore']
        operation = np.random.choice(operations)
        
        # 随机选择1-3个特征
        n_features = np.random.randint(1, 4)
        features = np.random.choice(basic_features, n_features, replace=False).tolist()
        
        # 随机选择参数
        window = np.random.randint(5, 60)
        lag = np.random.randint(1, 10)
        
        # 返回全局定义的 DynamicFactor 类实例
        return ParametricDynamicFactor(name, operation, features, window, lag)


if __name__ == "__main__":
    # config_path 优先级：环境变量 > 代码参数 > 默认
    config_path = os.environ.get("FACTOR_MINING_CONFIG") or "/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/factor_mining/config.yaml"
    engine = FactorMiningEngine(config_path=config_path)
    
    # 加载数据
    engine.load_data()
    
    # 发现并注册因子
    engine.discover_factors()
    
    # 生成自动因子
    generator = FactorGenerator(config=engine.config)
    random_factors = generator.generate_random_factors(engine.data, n_factors=20)
    
    # 注册自动生成的因子
    for factor in random_factors:
        engine.factors[factor.name] = factor
        engine.logger.info(f"已注册自动生成因子: {factor.name}")
    
    # 生成所有因子
    engine.generate_factors(parallel=False)
    
    # 计算每日收益率
    engine.calculate_returns(price_col='close', period=1)  

    # 处理缺失值（可选，evaluate_factors中也可以设置）
    engine.handle_missing_values(method='ffill')

    """
        todo: 因子逻辑加载存在问题
    """
    # 基于现有因子生成新特征
    # new_features = engine.generate_new_features()
    # engine.logger.info(f"生成的新特征: {new_features[:10]}（共{len(new_features)}个）")

    # 评估因子（传统方法）
    engine.evaluate_factors()
    
    # 评估因子（机器学习）
    # ml_evaluator = engine.evaluate_factors_ml()
    # print(f"机器学习评估最佳因子: {engine.consensus_factors}")

    # 分析因子时效性
    timeliness = engine.analyze_factor_timeliness()
    
    # 选择最佳因子（结合传统和机器学习评估）
    best_factors = engine.select_best_factors(top_n=20)

    # === 自动寻找并保存每个因子的最佳买卖阈值 ===
    # best_thresholds = engine.find_best_signal_thresholds(best_factors, method="zscore", window=60, metric="sharpe", save_path="best_signal_thresholds.json")
    # engine.logger.info(f"所有因子最优阈值: {best_thresholds}")

    # 保存结果
    engine.save_results()

    # === 保存有效因子的生成逻辑，便于复用 ===
    engine.save_factor_logic()  # 路径从config读取
    engine.logger.info("有效因子逻辑已保存。")

    # === 新增：演示如何加载保存的因子逻辑（可用于新数据/部署） ===
    engine.load_factor_logic()  # 路径从config读取
    engine.logger.info("已加载因子逻辑，可直接用于新数据。")

    # === 批量生成所有有效因子的买卖信号 ===
    # all_signals = engine.generate_all_signals(best_factors, method="zscore", buy_thr=1, sell_thr=-1, window=60, save_signals_path="all_signals_zscore.csv")
    # engine.logger.info(f"所有因子zscore信号样例（前5行）:\n{pd.DataFrame(all_signals).head()}")

    # ML方法（联合所有有效因子）
    all_signals_ml = engine.train_ml(best_factors, 
                                     train_start="2025-01-01", 
                                     train_end="2025-01-20", 
                                     val_start="2025-01-20", 
                                     val_end="2025-02-01")

    print("因子挖掘完成!")
    print(f"最佳因子: {best_factors}")

    # === 保存best_factors到文件 ===
    with open("/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/factor_mining/factor_data/best_factors.json", "w", encoding="utf-8") as f:
        json.dump(best_factors, f, ensure_ascii=False, indent=2)
    print("best_factors已保存到 best_factors.json")


