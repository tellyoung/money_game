import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import talib as ta

# 设置随机种子确保结果可复现
np.random.seed(42)

class CryptoFeatureEngineering:
    def __init__(self, data_path: str, time_periods: List[int] = [5, 10, 20, 50, 100]):
        """
        初始化特征工程类
        
        参数:
            data_path: 数据存储路径
            time_periods: 用于技术指标计算的不同时间周期
        """
        self.data_path = data_path
        self.time_periods = time_periods
        self.data = None
        self.features = None
        self.target = None
        
    def load_data(self, pairs: List[str], timeframe: str = '5m') -> Dict[str, pd.DataFrame]:
        """加载feather格式的数据"""
        loaded_data = {}
        for pair in pairs:
            file_path = os.path.join(self.data_path, f"{pair.replace('/', '_')}-{timeframe}.feather")
            if os.path.exists(file_path):
                try:
                    df = pd.read_feather(file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    loaded_data[pair] = df
                except Exception as e:
                    print(f"无法加载{pair}的数据: {e}")
            else:
                print(f"文件{file_path}不存在")
        return loaded_data
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为单个交易对创建特征"""
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"数据缺少必要的列: {required_columns}")
            
        # 复制数据避免修改原始数据
        result = df.copy()
        
        # 1. 价格相关特征
        for period in self.time_periods:
            # 移动平均线
            result[f'ma_{period}'] = ta.SMA(result['close'], timeperiod=period)
            result[f'ma_{period}_ratio'] = result['close'] / result[f'ma_{period}']
            
            # 指数移动平均线
            result[f'ema_{period}'] = ta.EMA(result['close'], timeperiod=period)
            result[f'ema_{period}_ratio'] = result['close'] / result[f'ema_{period}']
            
            # 最高价、最低价相关
            result[f'high_{period}'] = ta.MAX(result['high'], timeperiod=period)
            result[f'low_{period}'] = ta.MIN(result['low'], timeperiod=period)
            result[f'high_{period}_ratio'] = result['close'] / result[f'high_{period}']
            result[f'low_{period}_ratio'] = result['close'] / result[f'low_{period}']
            
            # 价格变化率
            result[f'return_{period}'] = result['close'].pct_change(period)
            
        # 2. 波动率特征
        for period in self.time_periods:
            result[f'atr_{period}'] = ta.ATR(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'atr_{period}_ratio'] = result[f'atr_{period}'] / result['close']
            
            result[f'bb_upper_{period}'], result[f'bb_middle_{period}'], result[f'bb_lower_{period}'] = ta.BBANDS(
                result['close'], timeperiod=period
            )
            result[f'bb_width_{period}'] = (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}']) / result[f'bb_middle_{period}']
            result[f'bb_position_{period}'] = (result['close'] - result[f'bb_lower_{period}']) / (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}'])
            
            # Keltner通道
            ma = ta.SMA(result['close'], timeperiod=period)
            atr = ta.ATR(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'kc_upper_{period}'] = ma + 2 * atr
            result[f'kc_lower_{period}'] = ma - 2 * atr
            result[f'kc_width_{period}'] = (result[f'kc_upper_{period}'] - result[f'kc_lower_{period}']) / ma
            result[f'kc_position_{period}'] = (result['close'] - result[f'kc_lower_{period}']) / (result[f'kc_upper_{period}'] - result[f'kc_lower_{period}'])
            
            # Donchian通道
            result[f'donchian_upper_{period}'] = ta.MAX(result['high'], timeperiod=period)
            result[f'donchian_lower_{period}'] = ta.MIN(result['low'], timeperiod=period)
            result[f'donchian_middle_{period}'] = (result[f'donchian_upper_{period}'] + result[f'donchian_lower_{period}']) / 2
            result[f'donchian_width_{period}'] = (result[f'donchian_upper_{period}'] - result[f'donchian_lower_{period}']) / result[f'donchian_middle_{period}']
            result[f'donchian_position_{period}'] = (result['close'] - result[f'donchian_lower_{period}']) / (result[f'donchian_upper_{period}'] - result[f'donchian_lower_{period}'])
            
            # 历史波动率
            result[f'volatility_{period}'] = result['close'].pct_change().rolling(period).std() * np.sqrt(period)
            
        # 3. 交易量特征
        for period in self.time_periods:
            result[f'volume_ma_{period}'] = ta.SMA(result['volume'], timeperiod=period)
            result[f'volume_ratio_{period}'] = result['volume'] / result[f'volume_ma_{period}']
            
            # 价格与交易量相关性
            result[f'price_volume_corr_{period}'] = result['close'].pct_change().rolling(period).corr(result['volume'].pct_change())
            
        # 4. 动量指标
        result['rsi'] = ta.RSI(result['close'], timeperiod=14)
        for period in self.time_periods:
            result[f'macd_{period}'], result[f'macd_signal_{period}'], result[f'macd_hist_{period}'] = ta.MACD(
                result['close'], fastperiod=period//2, slowperiod=period, signalperiod=9
            )
            result[f'macd_ratio_{period}'] = result[f'macd_{period}'] / result[f'macd_signal_{period}']
            
            result[f'stoch_k_{period}'], result[f'stoch_d_{period}'] = ta.STOCH(
                result['high'], result['low'], result['close'], 
                fastk_period=period, slowk_period=3, slowk_matype=0, 
                slowd_period=3, slowd_matype=0
            )
            
            result[f'willr_{period}'] = ta.WILLR(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'cci_{period}'] = ta.CCI(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'adosc_{period}'] = ta.ADOSC(result['high'], result['low'], result['close'], result['volume'], 
                                              fastperiod=3, slowperiod=period)
            
        # 5. 趋势指标
        for period in self.time_periods:
            result[f'adx_{period}'] = ta.ADX(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'plus_di_{period}'] = ta.PLUS_DI(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'minus_di_{period}'] = ta.MINUS_DI(result['high'], result['low'], result['close'], timeperiod=period)
            result[f'plus_di_diff_{period}'] = result[f'plus_di_{period}'] - result[f'minus_di_{period}']
            
            # Ichimoku云
            if period >= 9:
                result[f'tenkan_sen_{period}'] = (ta.MAX(result['high'], timeperiod=period//2) + 
                                                  ta.MIN(result['low'], timeperiod=period//2)) / 2
                result[f'kijun_sen_{period}'] = (ta.MAX(result['high'], timeperiod=period) + 
                                                 ta.MIN(result['low'], timeperiod=period)) / 2
                result[f'senkou_span_a_{period}'] = (result[f'tenkan_sen_{period}'] + result[f'kijun_sen_{period}']) / 2
                result[f'senkou_span_b_{period}'] = (ta.MAX(result['high'], timeperiod=period*2) + 
                                                    ta.MIN(result['low'], timeperiod=period*2)) / 2
                result[f'cloud_gap_{period}'] = result[f'senkou_span_a_{period}'] - result[f'senkou_span_b_{period}']
                result[f'cloud_position_{period}'] = (result['close'] - 
                                                    (result[f'senkou_span_a_{period}'] + result[f'senkou_span_b_{period}']) / 2) / \
                                                    (result[f'senkou_span_a_{period}'] - result[f'senkou_span_b_{period}']).abs()
                
        # 6. 交易量指标
        result['obv'] = ta.OBV(result['close'], result['volume'])
        for period in self.time_periods:
            result[f'obv_ma_{period}'] = ta.SMA(result['obv'], timeperiod=period)
            result[f'obv_ratio_{period}'] = result['obv'] / result[f'obv_ma_{period}']
            
            result[f'cmf_{period}'] = self.calculate_cmf(result, period)
            
        # 7. 价格模式识别
        for pattern in [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 
            'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 
            'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 
            'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 
            'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 
            'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 
            'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 
            'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 
            'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]:
            func = getattr(ta, pattern)
            result[pattern.lower()] = func(result['open'], result['high'], result['low'], result['close'])
            
        # 8. 时间特征
        result['hour'] = result.index.hour
        result['day'] = result.index.day
        result['day_of_week'] = result.index.dayofweek
        result['week_of_year'] = result.index.isocalendar().week
        result['month'] = result.index.month
        result['is_weekend'] = result.index.dayofweek >= 5
        
        # 9. 波动率聚集特征
        for period in [5, 10, 20, 50]:
            result[f'return_abs_{period}'] = result[f'return_{period}'].abs()
            result[f'return_squared_{period}'] = result[f'return_{period}'] ** 2
            
        # 10. 相对强度特征
        for pair in ['BTC/USDT', 'ETH/USDT']:  # 假设有这些交易对的数据
            if pair in loaded_data and pair != current_pair:
                result[f'rel_strength_{pair}'] = result['close'] / loaded_data[pair]['close']
                for period in self.time_periods:
                    result[f'rel_strength_{pair}_{period}_ratio'] = result[f'rel_strength_{pair}'] / \
                                                                    result[f'rel_strength_{pair}'].rolling(period).mean()
        
        # 11. 高阶统计特征
        for period in [20, 50, 100]:
            result[f'kurtosis_{period}'] = result['close'].pct_change().rolling(period).kurt()
            result[f'skewness_{period}'] = result['close'].pct_change().rolling(period).skew()
            
        # 12. 交易量分布特征
        for period in [20, 50, 100]:
            result[f'volume_kurtosis_{period}'] = result['volume'].pct_change().rolling(period).kurt()
            result[f'volume_skewness_{period}'] = result['volume'].pct_change().rolling(period).skew()
            
        # 13. 价格与交易量关系特征
        for period in [20, 50, 100]:
            result[f'price_volume_corr_{period}'] = result['close'].pct_change().rolling(period).corr(result['volume'].pct_change())
            result[f'price_volume_cov_{period}'] = result['close'].pct_change().rolling(period).cov(result['volume'].pct_change())
            
        # 14. 累计收益特征
        for period in [5, 10, 20, 50, 100]:
            result[f'cumulative_return_{period}'] = (1 + result['close'].pct_change()).rolling(period).apply(np.prod, raw=True) - 1
            
        # 15. 分形特征
        for period in [20, 50, 100]:
            result[f'hurst_{period}'] = self.calculate_hurst_exponent(result['close'], period)
            
        # 16. 自相关特征
        for lag in [1, 2, 3, 5, 10]:
            result[f'autocorr_{lag}'] = result['close'].pct_change().rolling(50).apply(lambda x: x.autocorr(lag=lag), raw=True)
            
        # 17. 滚动波动率特征
        for period in [5, 10, 20, 50]:
            result[f'rolling_vol_{period}'] = result['close'].pct_change().rolling(period).std() * np.sqrt(period)
            
        # 18. 市场情绪特征（假设存在情绪指标数据）
        if 'sentiment' in result.columns:
            for period in self.time_periods:
                result[f'sentiment_ma_{period}'] = result['sentiment'].rolling(period).mean()
                result[f'sentiment_diff_{period}'] = result['sentiment'] - result[f'sentiment_ma_{period}']
                
        # 19. 交叉特征
        for p1 in self.time_periods[:2]:  # 使用较短的周期
            for p2 in self.time_periods[2:]:  # 使用较长的周期
                if p1 < p2:
                    result[f'ma_cross_{p1}_{p2}'] = (result[f'ma_{p1}'] > result[f'ma_{p2}']).astype(int)
                    result[f'ema_cross_{p1}_{p2}'] = (result[f'ema_{p1}'] > result[f'ema_{p2}']).astype(int)
                    
        # 20. 相对价格水平特征
        for period in self.time_periods:
            result[f'price_level_{period}'] = (result['close'] - result['close'].rolling(period).min()) / \
                                            (result['close'].rolling(period).max() - result['close'].rolling(period).min())
            
        # 填充NaN值
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill')
        result = result.dropna()
        
        return result
    
    def calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算Chaikin资金流向指标"""
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        return cmf
    
    def calculate_hurst_exponent(self, series: pd.Series, period: int) -> pd.Series:
        """计算Hurst指数，衡量时间序列的长期记忆性"""
        hurst_values = []
        for i in range(len(series)):
            if i < period:
                hurst_values.append(np.nan)
            else:
                s = series.iloc[i-period:i]
                # 计算Hurst指数
                lags = range(2, 10)
                tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst_values.append(poly[0] * 2.0)
        return pd.Series(hurst_values, index=series.index)
    
    def create_target(self, df: pd.DataFrame, forecast_period: int = 10, threshold: float = 0.005) -> pd.Series:
        """
        创建目标变量：预测未来价格变动方向
        
        参数:
            df: 包含价格数据的DataFrame
            forecast_period: 预测未来多少个周期
            threshold: 定义涨跌的阈值
        """
        # 计算未来收益
        df[f'future_return_{forecast_period}'] = df['close'].pct_change(forecast_period).shift(-forecast_period)
        
        # 创建分类目标：1表示上涨，0表示下跌，2表示横盘
        df['target'] = 2  # 默认为横盘
        df.loc[df[f'future_return_{forecast_period}'] > threshold, 'target'] = 1  # 上涨
        df.loc[df[f'future_return_{forecast_period}'] < -threshold, 'target'] = 0  # 下跌
        
        # 移除包含NaN的行
        df = df.dropna(subset=['target'])
        
        return df['target']
    
    def prepare_data_for_modeling(self, df: pd.DataFrame, target: pd.Series, test_size: float = 0.2) -> Tuple:
        """准备用于建模的数据"""
        # 移除目标变量和不需要的列
        features = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
        
        # 分割训练集和测试集（按时间顺序）
        train_size = int(len(features) * (1 - test_size))
        X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
        y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]
        
        return X_train, X_test, y_train, y_test

class LightGBMPredictor:
    def __init__(self, params: Optional[Dict] = None):
        """初始化LightGBM预测器"""
        self.model = None
        self.feature_importance = None
        
        # LightGBM参数
        self.params = params or {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42,
            'n_jobs': -1
        }
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series, 
              num_boost_round: int = 1000, early_stopping_rounds: int = 50) -> None:
        """训练LightGBM模型"""
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50
        )
        
        # 保存特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.model.predict(X).argmax(axis=1)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估模型性能"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['下跌', '上涨', '横盘'])
        
        return {
            'accuracy': accuracy,
            'report': report,
            'y_pred': y_pred
        }
    
    def plot_feature_importance(self, top_n: int = 30, figsize: Tuple[int, int] = (10, 12)) -> None:
        """绘制特征重要性图"""
        if self.feature_importance is None:
            raise ValueError("特征重要性尚未计算，请先训练模型")
            
        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('特征重要性 (Gain)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

def main():
    """主函数：运行完整的特征工程和预测流程"""
    # 配置参数
    data_path = '/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20240101_20250417/futures'  # 替换为你的数据路径
    pairs = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']  # 替换为你的交易对列表
    forecast_period = 10  # 预测未来10个周期
    threshold = 0.005  # 0.5%的价格变动阈值
    
    # 初始化特征工程类
    fe = CryptoFeatureEngineering(data_path)
    
    # 加载数据
    loaded_data = fe.load_data(pairs)
    
    # 为每个交易对创建特征并训练模型
    for pair, df in loaded_data.items():
        print(f"\n处理交易对: {pair}")
        
        # 特征工程
        print("正在进行特征工程...")
        features_df = fe.engineer_features(df)
        
        # 创建目标变量
        print("正在创建目标变量...")
        target = fe.create_target(features_df, forecast_period, threshold)
        
        # 准备建模数据
        print("正在准备建模数据...")
        X_train, X_test, y_train, y_test = fe.prepare_data_for_modeling(features_df, target)
        
        # 使用时间序列交叉验证
        print("正在使用时间序列交叉验证...")
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_model = None
        best_val_score = float('inf')
        
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 初始化模型
            model = LightGBMPredictor()
            
            # 训练模型
            model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # 评估验证集性能
            val_results = model.evaluate(X_val_fold, y_val_fold)
            if val_results['accuracy'] < best_val_score:
                best_val_score = val_results['accuracy']
                best_model = model
                
        # 在测试集上评估最佳模型
        print("正在评估模型...")
        test_results = best_model.evaluate(X_test, y_test)
        
        print(f"\n交易对 {pair} 的模型性能:")
        print(f"准确率: {test_results['accuracy']:.4f}")
        print("分类报告:")
        print(test_results['report'])
        
        # 绘制特征重要性
        best_model.plot_feature_importance(top_n=50)
        
        # 保存特征重要性到CSV
        best_model.feature_importance.to_csv(f'{pair.replace("/", "_")}_feature_importance.csv', index=False)
        
        print(f"特征重要性已保存到 {pair.replace('/', '_')}_feature_importance.csv")

if __name__ == "__main__":
    main()    