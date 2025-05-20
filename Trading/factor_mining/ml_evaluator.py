import importlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import logging
from typing import Dict, List, Any, Tuple

class MLEvaluator:
    """使用机器学习模型评估因子有效性的模块
    该模块通过多种机器学习算法评估因子的预测能力，
    计算特征重要性，并提供因子选择的共识机制
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        self.models = self._init_models()  # 初始化评估模型
        self.feature_importance = {}  # 存储特征重要性结果
        self.performance = {}  # 存储模型性能结果
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统，同时输出到控制台和文件"""
        logger = logging.getLogger('MLEvaluator')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('ml_evaluator.log')
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
        
    def _init_models(self) -> Dict[str, Any]:
        """初始化用于评估的机器学习模型
        默认包含随机森林、梯度提升和逻辑回归模型
        """
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,  # 决策树数量
                max_depth=5,       # 树的最大深度
                random_state=42,   # 随机种子，保证结果可复现
                n_jobs=-1          # 使用所有CPU核心
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,  # 提升树数量
                max_depth=3,       # 树的最大深度
                random_state=42    # 随机种子
            ),
            'LogisticRegression': LogisticRegression(
                penalty='l2',      # L2正则化
                C=1.0,             # 正则化强度的倒数
                random_state=42,   # 随机种子
                n_jobs=-1          # 使用所有CPU核心
            )
        }
        
        # 添加配置中的自定义模型
        custom_models = self.config.get('models', {})
        for name, model_config in custom_models.items():
            try:
                model_class = getattr(importlib.import_module(model_config['module']), model_config['class'])
                models[name] = model_class(**model_config.get('params', {}))
                self.logger.info(f"已加载自定义模型: {name}")
            except Exception as e:
                self.logger.error(f"加载自定义模型失败: {name}, 错误: {e}")
                
        return models
        
    def evaluate(self, data: pd.DataFrame, factors: List[str], target_col: str = 'returns', 
                 n_splits: int = 5) -> None:
        """使用多种机器学习模型评估因子有效性
        通过时间序列交叉验证，评估因子在预测目标变量上的表现
        
        参数:
            data: 包含因子和目标变量的数据框
            factors: 要评估的因子列表
            target_col: 目标变量列名
            n_splits: 时间序列交叉验证的折数
        """
        self.logger.info(f"开始机器学习评估，因子数量: {len(factors)}")
        
        # 确保目标列存在且为二元分类
        if target_col not in data.columns:
            self.logger.error(f"目标列 {target_col} 不存在")
            return
            
        # 创建二元目标变量（上涨/下跌）
        data['target'] = (data[target_col] > 0).astype(int)
        
        # 移除缺失值
        valid_data = data[factors + ['target']].dropna()
        X = valid_data[factors]
        y = valid_data['target']
        
        # 时间序列交叉验证（防止未来数据泄露）
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for model_name, model in self.models.items():
            self.logger.info(f"使用模型 {model_name} 进行评估")
            model_performance = []
            feature_importance = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 标准化特征（对线性模型和基于距离的模型很重要）
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 训练模型
                model.fit(X_train_scaled, y_train)
                
                # 预测并评估
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                # 计算性能指标
                accuracy = accuracy_score(y_test, y_pred)  # 准确率
                auc = roc_auc_score(y_test, y_prob)       # AUC值
                
                model_performance.append({
                    'accuracy': accuracy,
                    'auc': auc
                })
                
                # 记录特征重要性
                if hasattr(model, 'feature_importances_'):
                    # 树模型使用feature_importances_
                    feature_importance.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    # 线性模型使用系数
                    feature_importance.append(model.coef_[0])
            
            # 计算平均性能
            avg_performance = {
                'accuracy_mean': np.mean([p['accuracy'] for p in model_performance]),
                'accuracy_std': np.std([p['accuracy'] for p in model_performance]),
                'auc_mean': np.mean([p['auc'] for p in model_performance]),
                'auc_std': np.std([p['auc'] for p in model_performance])
            }
            
            self.performance[model_name] = avg_performance
            
            # 计算平均特征重要性
            if feature_importance:
                avg_importance = np.mean(feature_importance, axis=0)
                self.feature_importance[model_name] = dict(zip(factors, avg_importance))
                
                # 按重要性排序
                sorted_importance = sorted(
                    self.feature_importance[model_name].items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                self.logger.info(f"{model_name} 特征重要性排名: {sorted_importance[:10]}")
            
            self.logger.info(f"{model_name} 评估完成: {avg_performance}")
    
    def calculate_mutual_information(self, data: pd.DataFrame, factors: List[str], 
                                    target_col: str = 'returns') -> Dict[str, float]:
        """计算因子与目标变量之间的互信息
        互信息是一种衡量两个随机变量依赖性的非参数方法
        
        参数:
            data: 包含因子和目标变量的数据框
            factors: 要评估的因子列表
            target_col: 目标变量列名
            
        返回:
            包含因子与目标变量互信息的字典
        """
        self.logger.info("计算因子与目标变量的互信息")
        
        # 确保目标列存在且为二元分类
        if target_col not in data.columns:
            self.logger.error(f"目标列 {target_col} 不存在")
            return {}
            
        # 创建二元目标变量
        data['target'] = (data[target_col] > 0).astype(int)
        
        # 移除缺失值
        valid_data = data[factors + ['target']].dropna()
        X = valid_data[factors]
        y = valid_data['target']
        
        # 计算互信息（衡量变量间的依赖性）
        mi_scores = mutual_info_classif(X, y)
        mi_dict = dict(zip(factors, mi_scores))
        
        # 按互信息排序
        sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f"互信息排名: {sorted_mi[:10]}")
        
        return mi_dict
    
    def get_top_factors(self, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """获取每个模型认为最重要的前N个因子
        
        参数:
            n: 要获取的顶级因子数量
            
        返回:
            字典，键为模型名称，值为按重要性排序的因子列表
        """
        top_factors = {}
        
        for model_name, importance in self.feature_importance.items():
            # 按重要性绝对值排序
            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors[model_name] = sorted_importance[:n]
            
        return top_factors
    
    def get_factor_consensus(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取所有模型一致认为重要的因子
        通过综合多个模型的评估结果，找出最具共识的因子
        
        参数:
            n: 要获取的顶级因子数量
            
        返回:
            按共识度排序的因子列表
        """
        if not self.feature_importance:
            return []
            
        # 计算每个因子在所有模型中的平均排名
        factor_ranks = {}
        
        for model_name, importance in self.feature_importance.items():
            # 按重要性排序
            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # 为每个因子分配排名
            for rank, (factor, _) in enumerate(sorted_importance):
                if factor not in factor_ranks:
                    factor_ranks[factor] = []
                factor_ranks[factor].append(rank)
        
        # 计算平均排名
        avg_ranks = {factor: np.mean(ranks) for factor, ranks in factor_ranks.items()}
        
        # 按平均排名排序（排名越小越重要）
        consensus = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        return consensus[:n]    