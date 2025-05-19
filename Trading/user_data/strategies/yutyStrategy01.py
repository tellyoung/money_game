from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame, Series, DatetimeIndex
import talib.abstract as ta
import pandas as pd
from datetime import datetime
import logging

# 配置日志（通常在策略外部或 __init__ 方法中）
logger = logging.getLogger(__name__)


class yutyStrategy01(IStrategy):
    can_short = True

    timeframe = '5m'
    
    # 策略参数
    stoploss = -0.20  # 初始值，实际在custom_stoploss中动态计算
    trailing_stop = False
    process_only_new_candles = True
    use_custom_stoploss = True
    use_exit_signal = False  # 关闭内置退出信号

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算多周期均线
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        
        # 计算ATR(14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # 计算RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # 计算布林带
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['close']
        
        # 计算ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # 计算Stochastic RSI
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.create_signals(dataframe)
        
        # 多头条件
        dataframe.loc[
            (dataframe['ema5'] > dataframe['ema20']) &  # 短期趋势向上
            (dataframe['rsi'] > 40) &  # RSI 放宽条件
            (dataframe['stoch_k'] < 30) &  # Stochastic RSI 超卖区域
            (dataframe['adx'] > 15) &  # 趋势强度放宽
            (dataframe['close'] > dataframe['bb_lowerband']) &  # 接近布林带下轨
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        
        # 空头条件
        dataframe.loc[
            (dataframe['ema5'] < dataframe['ema20']) &
            (dataframe['rsi'] < 60) &  # RSI 放宽条件
            (dataframe['stoch_k'] > 70) &  # Stochastic RSI 超买区域
            (dataframe['adx'] > 15) &
            (dataframe['close'] < dataframe['bb_upperband']) &  # 接近布林带上轨
            (dataframe['volume'] > 0),
            'enter_short'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 由于使用移动止损，此处保持空退出信号
        dataframe = self.create_signals(dataframe)
        dataframe['exit_long'] = 0  # 不主动触发退出
        dataframe['exit_short'] = 0
        return dataframe

    def create_signals(self, dataframe):
        # 初始化信号列
        for col in ['enter_long', 'enter_short', 'exit_long', 'exit_short']:
            if col not in dataframe:
                dataframe[col] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        动态止损逻辑：
        - 初始止损为 1 倍 ATR
        - 当利润达到 1.5 倍 ATR 时，移动止损到开仓价
        - 当利润达到 2 倍 ATR 时，锁定 50% 利润
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        open_date = trade.open_date_utc.replace(tzinfo=None)
        mask = dataframe['date'] == open_date
        if not dataframe[mask].empty:
            atr_entry = dataframe[mask]['atr'].iloc[0]
        else:
            return 1  # 异常情况返回默认值

        if not trade.is_short:
            price_diff = current_rate - trade.open_rate
            if price_diff >= 2 * atr_entry:
                # 锁定 50% 利润
                return (current_rate - trade.open_rate * 0.5) / current_rate
            elif price_diff >= 1.5 * atr_entry:
                # 移动到保本
                return (current_rate - trade.open_rate) / current_rate
            else:
                # 初始止损
                stop_price = trade.open_rate - 1 * atr_entry
                return (current_rate - stop_price) / current_rate
        else:
            price_diff = trade.open_rate - current_rate
            if price_diff >= 2 * atr_entry:
                return (trade.open_rate - current_rate * 0.5) / current_rate
            elif price_diff >= 1.5 * atr_entry:
                return (trade.open_rate - current_rate) / current_rate
            else:
                stop_price = trade.open_rate + 1 * atr_entry
                return (stop_price - current_rate) / current_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        """
        动态仓位管理：
        - 根据波动率动态调整仓位
        - 高波动行情中增加仓位，低波动行情中减少仓位
        """
        try:
            # 获取当前ATR
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            atr = dataframe['atr'].iloc[-1]
            
            # 动态调整仓位
            atr_factor = min(max(atr / current_rate, 0.01), 0.05)  # ATR占比限制在1%-5%
            adjusted_stake = proposed_stake * atr_factor
            
            return max(min_stake, min(adjusted_stake, max_stake))
        except Exception as e:
            logger.info(f"仓位计算错误: {str(e)}", exc_info=True)
            return proposed_stake
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            }
        ]



