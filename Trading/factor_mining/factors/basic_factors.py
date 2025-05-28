import pandas as pd
import numpy as np
from .factor import Factor


class MomentumFactor(Factor):
    """动量因子：衡量资产价格的上涨或下跌趋势强度"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "momentum_20d"
        self.description = "20日价格动量"
        self.category = "price_momentum"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算20日价格变化率作为动量指标
        return data['close'].pct_change(periods=20)


class VolatilityFactor(Factor):
    """波动率因子：衡量资产价格的波动程度"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "volatility_20d"
        self.description = "20日收益率波动率"
        self.category = "risk"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算20日收益率的标准差作为波动率指标
        returns = data['close'].pct_change()
        return returns.rolling(window=20).std()


class VolumeFactor(Factor):
    """成交量因子：衡量市场交易活跃程度"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "volume_change_5d"
        self.description = "5日成交量变化率"
        self.category = "volume"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算5日成交量变化率
        return data['volume'].pct_change(periods=5)


class MACDFactor(Factor):
    """MACD因子：技术分析中常用的趋势跟踪指标"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "macd"
        self.description = "MACD指标"
        self.category = "technical_indicator"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算短期EMA (12天)
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        
        # 计算长期EMA (26天)
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        
        # 计算MACD线（快线与慢线的差值）
        macd_line = ema12 - ema26
        
        return macd_line


class RSIFactor(Factor):
    """RSI因子：相对强弱指数，衡量价格变动的速度和幅度"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "rsi_14d"
        self.description = "14日相对强弱指数"
        self.category = "technical_indicator"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算价格变动
        delta = data['close'].diff()
        
        # 计算上涨和下跌变动
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均收益和损失
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # 计算RSI (Relative Strength Index)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class BollingerBandsFactor(Factor):
    """布林带因子：衡量价格波动范围和超买超卖情况"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "bollinger_band_width"
        self.description = "布林带宽度"
        self.category = "technical_indicator"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算中轨 (20日移动平均线)
        middle_band = data['close'].rolling(window=20).mean()
        
        # 计算标准差
        std = data['close'].rolling(window=20).std()
        
        # 计算上轨和下轨
        upper_band = middle_band + (2 * std)
        lower_band = middle_band - (2 * std)
        
        # 计算布林带宽度，反映价格波动程度
        band_width = (upper_band - lower_band) / middle_band
        
        return band_width


class KDJFactor(Factor):
    """KDJ因子：随机指标，用于判断超买超卖情况"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "kdj_j"
        self.description = "KDJ指标的J值"
        self.category = "technical_indicator"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 计算RSV (Raw Stochastic Value)
        low_9 = data['low'].rolling(window=9).min()
        high_9 = data['high'].rolling(window=9).max()
        rsv = (data['close'] - low_9) / (high_9 - low_9) * 100
        
        # 计算K值和D值
        k = rsv.ewm(com=2).mean()  # 相当于3日移动平均
        d = k.ewm(com=2).mean()    # 相当于3日移动平均
        
        # 计算J值，J值是KDJ指标中最敏感的部分
        j = 3 * k - 2 * d
        
        return j


class OBVFactor(Factor):
    """能量潮因子：通过成交量变化预测价格走势"""
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "obv_change_10d"
        self.description = "10日能量潮变化率"
        self.category = "volume_indicator"
        
    def generate(self, data: pd.DataFrame) -> pd.Series:
        # 初始化OBV (On-Balance Volume)，用float类型避免dtype警告
        obv = pd.Series(0.0, index=data.index, dtype='float64')
        
        # 计算OBV：如果当日收盘价高于前一日，则将当日成交量加入OBV；否则减去
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # 计算OBV变化率
        obv_change = obv.pct_change(periods=10)
        
        return obv_change