import pandas as pd
import numpy as np
from .factor import Factor

class PriceVolumeCorrelationFactor(Factor):
    """价格成交量相关性因子：衡量价格和成交量之间的关联程度"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "price_volume_corr_20d"
        self.description = "20日价格与成交量相关性"
        self.category = "correlation"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算价格收益率
        returns = data['close'].pct_change()
        
        # 计算价格与成交量的滚动相关性
        corr = returns.rolling(window=20).corr(data['volume'])
        
        return corr


class VolatilityBreakoutFactor(Factor):
    """波动率突破因子：识别价格波动率突然变化的情况"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "volatility_breakout"
        self.description = "波动率突破信号"
        self.category = "breakout"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算20日价格标准差
        std_20 = data['close'].rolling(window=20).std()
        
        # 计算5日价格标准差
        std_5 = data['close'].rolling(window=5).std()
        
        # 波动率突破信号：短期波动率与长期波动率的比值
        breakout = std_5 / (std_20 + 1e-10)
        
        return breakout


class TrendStrengthFactor(Factor):
    """趋势强度因子：衡量当前价格趋势的强度"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "trend_strength_50d"
        self.description = "50日趋势强度"
        self.category = "trend"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算50日移动平均线
        ma_50 = data['close'].rolling(window=50).mean()
        
        # 计算趋势强度：价格与均线的偏离程度除以波动率
        trend_strength = (data['close'] - ma_50) / (data['close'].rolling(window=50).std() + 1e-10)
        
        return trend_strength


class MeanReversionFactor(Factor):
    """均值回归因子：预测价格向均值回归的可能性"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "mean_reversion_10d"
        self.description = "10日均值回归信号"
        self.category = "mean_reversion"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算10日移动平均线
        ma_10 = data['close'].rolling(window=10).mean()
        
        # 计算均值回归信号：价格与均线的负偏离程度除以波动率
        # 负值表示价格低于均值，可能会反弹；正值表示价格高于均值，可能会回落
        mean_reversion = -(data['close'] - ma_10) / (data['close'].rolling(window=10).std() + 1e-10)
        
        return mean_reversion


class VolumeWeightedReturnFactor(Factor):
    """成交量加权收益率因子：考虑成交量的收益率指标"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "vw_return_5d"
        self.description = "5日成交量加权收益率"
        self.category = "volume_weighted"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算每日收益率
        returns = data['close'].pct_change()
        
        # 计算成交量加权收益率：收益率乘以成交量，再除以总成交量
        vw_return = (returns * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
        
        return vw_return


class NewsSentimentFactor(Factor):
    """新闻情绪因子：基于新闻文本分析的情绪指标"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "news_sentiment"
        self.description = "新闻情绪因子"
        self.category = "sentiment"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 注意：这个因子需要新闻数据，这里仅作为示例
        # 实际应用中需要接入新闻API或其他情绪数据源
        if 'news_sentiment' not in data.columns:
            # 如果没有新闻数据，返回随机值作为占位符
            return pd.Series(np.random.randn(len(data)), index=data.index)
            
        return data['news_sentiment']


class SeasonalityFactor(Factor):
    """季节性因子：捕捉价格的季节性变化模式"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "seasonality_factor"
        self.description = "季节性因子"
        self.category = "time_series"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 从日期中提取月份
        if 'date' in data.columns:
            data['month'] = pd.to_datetime(data['date']).dt.month
        else:
            # 如果没有日期列，使用索引作为日期
            data['month'] = pd.to_datetime(data.index).dt.month
            
        # 计算每月平均收益率
        monthly_returns = data.groupby('month')['close'].pct_change().mean()
        
        # 为每个数据点分配对应月份的平均收益率
        seasonality = data['month'].map(monthly_returns)
        
        return seasonality


class MarketNeutralFactor(Factor):
    """市场中性因子：消除市场整体影响的因子"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "market_neutral_momentum"
        self.description = "市场中性动量因子"
        self.category = "market_neutral"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算个股动量
        stock_momentum = data['close'].pct_change(periods=20)
        
        # 计算市场动量（假设数据中包含市场指数）
        if 'market_close' in data.columns:
            market_momentum = data['market_close'].pct_change(periods=20)
        else:
            # 如果没有市场数据，使用个股动量的移动平均作为代理
            market_momentum = stock_momentum.rolling(window=50).mean()
            
        # 计算市场中性动量：个股动量减去市场动量
        market_neutral_momentum = stock_momentum - market_momentum
        
        return market_neutral_momentum