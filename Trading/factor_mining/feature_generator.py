import pandas as pd
import numpy as np
from factors.factor import Factor,DynamicFactor
import logging
from typing import List, Dict, Any, Callable, Union
import itertools
import inspect
from sklearn.preprocessing import PolynomialFeatures

class FeatureGenerator:
    """基于现有因子自动生成新因子的特征生成模块
    该模块通过多种变换和组合方式，从现有因子生成大量候选因子
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        self.operations = self._get_operations()  # 可用的数学操作
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统，同时输出到控制台和文件"""
        logger = logging.getLogger('FeatureGenerator')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('feature_generator.log')
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
        
    def _get_operations(self) -> Dict[str, Callable]:
        """获取可用的数学操作
        这些操作将用于组合现有因子生成新因子
        """
        return {
            'add': lambda x, y: x + y,                # 加法
            'subtract': lambda x, y: x - y,           # 减法
            'multiply': lambda x, y: x * y,           # 乘法
            'divide': lambda x, y: x / (y + 1e-10),   # 除法（避免除以零）
            'log': lambda x: np.log(x + 1e-10),       # 对数（避免对数为零）
            'sqrt': lambda x: np.sqrt(np.abs(x)),     # 平方根（避免负数平方根）
            'power2': lambda x: x ** 2,               # 平方
            'power3': lambda x: x ** 3,               # 立方
            'abs': lambda x: np.abs(x),               # 绝对值
            'exp': lambda x: np.exp(x),               # 指数
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)) # Sigmoid函数
        }
        
    def generate_math_combinations(self, factors: Dict[str, pd.Series], 
                                  max_combination: int = 2) -> Dict[str, pd.Series]:
        """生成因子间的数学组合
        通过基本数学运算组合现有因子，生成新的候选因子
        
        参数:
            factors: 现有因子的字典，键为因子名，值为因子数据
            max_combination: 最大组合数
            
        返回:
            新生成的因子字典
        """
        self.logger.info(f"开始生成因子数学组合，最大组合数: {max_combination}")
        
        new_factors = {}
        
        # 生成单因子操作
        for factor_name, factor_series in factors.items():
            for op_name, op_func in self.operations.items():
                if op_name in ['add', 'subtract', 'multiply', 'divide']:
                    # 这些是二元操作，跳过单因子处理
                    continue
                    
                try:
                    # 应用操作
                    new_factor = op_func(factor_series)
                    new_name = f"{op_name}_{factor_name}"
                    
                    # 检查有效性（确保没有全部是NaN或Inf）
                    if not np.isnan(new_factor).all() and not np.isinf(new_factor).all():
                        new_factors[new_name] = new_factor
                        self.logger.info(f"生成新因子: {new_name}")
                except Exception as e:
                    self.logger.error(f"生成因子失败: {op_name}_{factor_name}, 错误: {e}")
        
        # 生成多因子组合
        factor_names = list(factors.keys())
        
        # 生成组合对
        for n in range(2, max_combination + 1):
            for combo in itertools.combinations(factor_names, n):
                # 只考虑二元操作
                if n == 2:
                    factor1, factor2 = combo
                    for op_name, op_func in self.operations.items():
                        if op_name not in ['add', 'subtract', 'multiply', 'divide']:
                            # 这些是一元操作，跳过
                            continue
                            
                        try:
                            # 应用操作
                            new_factor = op_func(factors[factor1], factors[factor2])
                            new_name = f"{factor1}_{op_name}_{factor2}"
                            
                            # 检查有效性
                            if not np.isnan(new_factor).all() and not np.isinf(new_factor).all():
                                new_factors[new_name] = new_factor
                                self.logger.info(f"生成新因子: {new_name}")
                        except Exception as e:
                            self.logger.error(f"生成因子失败: {new_name}, 错误: {e}")
        
        self.logger.info(f"数学组合生成完成，新增因子数量: {len(new_factors)}")
        return new_factors
    
    def generate_time_series_features(self, factors: Dict[str, pd.Series], 
                                    windows: List[int] = [5, 10, 20, 60]) -> Dict[str, pd.Series]:
        """生成时间序列特征，如移动平均、波动率等
        基于不同时间窗口生成各种统计特征
        
        参数:
            factors: 现有因子的字典
            windows: 时间窗口大小列表
            
        返回:
            新生成的时间序列特征字典
        """
        self.logger.info(f"开始生成时间序列特征，窗口大小: {windows}")
        
        new_factors = {}
        
        for factor_name, factor_series in factors.items():
            for window in windows:
                try:
                    # 移动平均（平滑因子）
                    new_factors[f"{factor_name}_ma_{window}"] = factor_series.rolling(window=window).mean()
                    
                    # 移动标准差（波动率）
                    new_factors[f"{factor_name}_std_{window}"] = factor_series.rolling(window=window).std()
                    
                    # 移动最大值
                    new_factors[f"{factor_name}_max_{window}"] = factor_series.rolling(window=window).max()
                    
                    # 移动最小值
                    new_factors[f"{factor_name}_min_{window}"] = factor_series.rolling(window=window).min()
                    
                    # 移动偏度（分布不对称性）
                    new_factors[f"{factor_name}_skew_{window}"] = factor_series.rolling(window=window).skew()
                    
                    # 移动峰度（分布尾部厚度）
                    new_factors[f"{factor_name}_kurt_{window}"] = factor_series.rolling(window=window).kurt()
                    
                    # 移动相关系数（与其他因子的相关性）
                    for other_factor in factors:
                        if other_factor != factor_name:
                            new_factors[f"{factor_name}_corr_{other_factor}_{window}"] = \
                                factor_series.rolling(window=window).corr(factors[other_factor])
                    
                    self.logger.info(f"为 {factor_name} 生成窗口 {window} 的时间序列特征")
                except Exception as e:
                    self.logger.error(f"生成时间序列特征失败: {factor_name}, 窗口: {window}, 错误: {e}")
        
        self.logger.info(f"时间序列特征生成完成，新增因子数量: {len(new_factors)}")
        return new_factors
    
    def generate_polynomial_features(self, factors: Dict[str, pd.Series], 
                                    degree: int = 2) -> Dict[str, pd.Series]:
        """生成多项式特征
        通过多项式组合现有因子，生成更高次的特征
        
        参数:
            factors: 现有因子的字典
            degree: 多项式阶数
            
        返回:
            新生成的多项式特征字典
        """
        self.logger.info(f"开始生成多项式特征，阶数: {degree}")
        
        # 转换为DataFrame
        factors_df = pd.DataFrame(factors)
        # if factors_df.isnull().values.any():
        #     self.logger.warning("输入数据包含异常值，正在处理...")
        #     nan_rows = factors_df.isnull().any(axis=1).sum()
        #     self.logger.info(f"输入数据中包含 NaN 的行数: {nan_rows}")
        #     factors_df = factors_df.fillna(0)  # 这里选择填充为 0
        #     # 限制值的范围
        #     factors_df = factors_df.clip(lower=-1e10, upper=1e10)
        #     # 替换无穷值为 NaN
        #     factors_df = factors_df.replace([np.inf, -np.inf], np.nan)


        # 使用sklearn的PolynomialFeatures生成多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(factors_df)
        
        # 获取新特征名称
        feature_names = poly.get_feature_names_out(factors_df.columns)
        
        # 转换回字典
        new_factors = {}
        for i, feature_name in enumerate(feature_names):
            if feature_name in factors_df.columns:
                # 跳过原始特征
                continue
                
            try:
                new_factors[feature_name] = pd.Series(poly_features[:, i], index=factors_df.index)
                self.logger.info(f"生成多项式特征: {feature_name}")
            except Exception as e:
                self.logger.error(f"生成多项式特征失败: {feature_name}, 错误: {e}")
        
        self.logger.info(f"多项式特征生成完成，新增因子数量: {len(new_factors)}")
        return new_factors
    
    def generate_cross_sectional_features(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """生成横截面特征，如排名、分位数等
        基于因子在横截面上的相对位置生成新特征
        
        参数:
            factors: 现有因子的字典
            
        返回:
            新生成的横截面特征字典
        """
        self.logger.info("开始生成横截面特征")
        
        new_factors = {}
        
        for factor_name, factor_series in factors.items():
            try:
                # 排名（百分位）
                new_factors[f"{factor_name}_rank"] = factor_series.rank(pct=True)
                
                # 分位数指标
                for q in [0.1, 0.25, 0.75, 0.9]:
                    new_factors[f"{factor_name}_quantile_{q}"] = (factor_series <= factor_series.quantile(q)).astype(float)
                
                # z-score标准化（归一化到均值0标准差1）
                mean = factor_series.mean()
                std = factor_series.std()
                new_factors[f"{factor_name}_zscore"] = (factor_series - mean) / (std + 1e-10)
                
                # 相对强弱（与前一期相比）
                new_factors[f"{factor_name}_rs"] = factor_series / factor_series.shift(1)
                
                self.logger.info(f"为 {factor_name} 生成横截面特征")
            except Exception as e:
                self.logger.error(f"生成横截面特征失败: {factor_name}, 错误: {e}")
        
        self.logger.info(f"横截面特征生成完成，新增因子数量: {len(new_factors)}")
        return new_factors
    
    def generate_all_features(self, factors: Dict[str, pd.Series], 
                             config: Dict[str, Any] = None) -> Dict[str, pd.Series]:
        """生成所有类型的特征
        整合所有特征生成方法，生成全面的候选因子
        
        参数:
            factors: 现有因子的字典
            config: 配置参数
            
        返回:
            所有新生成的特征字典
        """
        config = config or {}
        
        # 获取配置参数
        max_combination = config.get('max_combination', 2)  # 最大组合数
        windows = config.get('windows', [5, 10, 20, 60])  # 时间窗口
        degree = config.get('polynomial_degree', 2)       # 多项式阶数
        
        # 生成各类特征
        new_factors = {}
        
        # # 数学组合特征
        # math_factors = self.generate_math_combinations(factors, max_combination)
        # new_factors.update(math_factors)
        
        # # 时间序列特征
        # ts_factors = self.generate_time_series_features(factors, windows)
        # new_factors.update(ts_factors)
        
        # # 多项式特征
        # poly_factors = self.generate_polynomial_features(factors, degree)
        # new_factors.update(poly_factors)
        
        # 横截面特征
        cs_factors = self.generate_cross_sectional_features(factors)
        new_factors.update(cs_factors)
        
        self.logger.info(f"所有特征生成完成，总共新增因子: {len(new_factors)}")
        return new_factors    