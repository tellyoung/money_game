from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from pandas import DataFrame, Series
import talib.abstract as ta
import numpy as np

class yutyStrategy02(IStrategy):
    can_short = True

    # 策略基础配置
    timeframe = '5m'  # 策略运行的时间周期为5分钟
    process_only_short = False  # 是否只运行空头策略，False表示支持多头和空头
    process_only_long = False  # 是否只运行多头策略，False表示支持多头和空头
    
    # 风险参数
    stoploss = -0.99  # 动态止损占位符，实际止损逻辑在custom_stoploss中实现
    trailing_stop = True  # 启用追踪止损
    trailing_stop_positive = 0.2  # 当价格达到一定盈利时，触发追踪止损
    trailing_only_offset_is_reached = True  # 只有在达到正向偏移时才启用追踪止损
    
    # 策略参数（添加 hyperopt 参数）
    atr_period = IntParameter(10, 30, default=14, space='buy', optimize=True)  # ATR（平均真实波幅）的计算周期
    volatility_threshold = DecimalParameter(0.01, 0.08, default=0.03, space='buy', optimize=True)  # 波动率阈值
    ma_period = IntParameter(5, 20, default=18, space='buy', optimize=True)  # 移动平均线的周期

    # >>>> from me
    h_w = IntParameter(1, 15, default=7, space='buy', optimize=True)
    l_w = IntParameter(1, 15, default=8, space='buy', optimize=True)
    roc_n = IntParameter(6, 18, default=11, space='buy', optimize=True)
    momentum_threshold_long = DecimalParameter(-0.5, 0.5, default=0.0, space='buy', optimize=True)
    momentum_threshold_short = DecimalParameter(-0.5, 0.5, default=0.0, space='buy', optimize=True)
    sell_1 = IntParameter(0, 5, default=1, space='sell', optimize=True)
    sell_2 = IntParameter(1, 8, default=3, space='sell', optimize=True)
    # 添加 hyperopt 参数
    initial_stoploss_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space='sell', optimize=True)  # 初始止损倍数
    trailing_stop_multiplier_long = DecimalParameter(0.5, 1.0, default=0.8, space='sell', optimize=True)  # 多头追踪止损倍数
    trailing_stop_multiplier_short = DecimalParameter(1.0, 1.5, default=1.2, space='sell', optimize=True)  # 空头追踪止损倍数


    # 状态跟踪
    drawdown_blacklist = {}  # 用于记录熔断机制中被禁止交易的交易对
    last_check_date = None  # 记录最后一次检查熔断机制的日期

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算技术指标
        dataframe['ma7'] = ta.SMA(dataframe, timeperiod=int(self.ma_period.value))  # 7周期简单移动平均线
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=int(self.atr_period.value))  # 平均真实波幅，用于衡量市场波动性
        
        # 三线突破信号
        dataframe['prior_high'] = dataframe['high'].rolling(window=int(self.h_w.value)).max().shift(1)  # 过去3根K线的最高价
        dataframe['prior_low'] = dataframe['low'].rolling(window=int(self.l_w.value)).min().shift(1)  # 过去3根K线的最低价
        
        # 趋势强度指标（用于品种轮动）
        dataframe['momentum'] = ta.ROC(dataframe, timeperiod=24 * int(self.roc_n.value))  # 24小时动量指标
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头入场逻辑
        dataframe.loc[ 
            (dataframe['close'] > dataframe['prior_high']) &  # 收盘价高于过去3根K线的最高价
            (dataframe['close'] > dataframe['ma7']) &  # 收盘价高于7周期移动平均线
            (dataframe['atr'] > self.volatility_threshold.value) &  # ATR超过波动率阈值
            (dataframe['momentum'] > self.momentum_threshold_long.value) &  # 动量为正
            (self.allow_trading(metadata)),  # 熔断机制允许交易
            'enter_long'
        ] = 1

        # 空头入场逻辑
        dataframe.loc[
            (dataframe['close'] < dataframe['prior_low']) &  # 收盘价低于过去3根K线的最低价
            (dataframe['close'] < dataframe['ma7']) &  # 收盘价低于7周期移动平均线
            (dataframe['atr'] > self.volatility_threshold.value) &  # ATR超过波动率阈值
            (dataframe['momentum'] < self.momentum_threshold_short.value) &  # 动量为负
            (self.allow_trading(metadata)),  # 熔断机制允许交易
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 基础退出信号（主要退出逻辑在custom_exit中实现）
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        # 分阶段止盈逻辑
        atr = self.get_current_atr(pair)  # 获取当前ATR值
        if trade.is_short:
            price_diff = trade.open_rate - current_rate  # 空头价格变动
        else:
            price_diff = current_rate - trade.open_rate  # 多头价格变动

        if price_diff >= self.sell_1.value * atr:  # 第一阶段止盈：价格变动达到2倍ATR
            return "take_profit_50%", 0.5  # 平仓50%
        elif price_diff >= self.sell_2.value * atr:  # 第二阶段止盈：价格变动达到3倍ATR
            return "take_profit_100%", 1.0  # 平仓100%
        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # 动态止损逻辑
        atr = self.get_current_atr(pair)  # 获取当前ATR值
        
        if trade.is_short:
            return self.calculate_short_stoploss(trade, atr, current_rate)  # 空头止损计算
        else:
            return self.calculate_long_stoploss(trade, atr, current_rate)  # 多头止损计算

    def calculate_long_stoploss(self, trade, atr, current_rate):
        # 多头止损计算
        if len(trade.orders) <= 1:  # 初始止损
            return (trade.open_rate - self.initial_stoploss_multiplier.value * atr) / current_rate
        # 追踪止损
        return (trade.max_rate * self.trailing_stop_multiplier_long.value) / current_rate

    def calculate_short_stoploss(self, trade, atr, current_rate):
        # 空头止损计算
        if len(trade.orders) <= 1:  # 初始止损
            return (trade.open_rate + self.initial_stoploss_multiplier.value * atr) / current_rate
        # 追踪止损
        return (trade.min_rate * self.trailing_stop_multiplier_short.value) / current_rate

    def allow_trading(self, metadata):
        # 熔断机制实现
        pair = metadata['pair']  # 当前交易对
        now = datetime.utcnow()  # 当前时间
        
        # 每日重置检测
        if self.last_check_date != now.date():
            self.drawdown_blacklist = {}  # 清空熔断黑名单
            self.last_check_date = now.date()  # 更新最后检查日期
        
        if pair in self.drawdown_blacklist:  # 如果交易对在黑名单中，禁止交易
            return False
        
        # 计算当日回撤
        trades = Trade.get_trades_proxy(pair=pair)  # 获取交易对的交易记录
        today_trades = [t for t in trades if t.open_date.date() == now.date()]  # 筛选当日交易
        # if len(today_trades) >= 5:  # 每日最多交易3次
        #     return False
        
        if len(today_trades) >= 1:
            drawdown = self.calculate_max_drawdown(today_trades)  # 计算最大回撤
            if drawdown > 0.01:  # 如果回撤超过1%，加入黑名单
                self.drawdown_blacklist[pair] = now
                return False
        return True

    def calculate_max_drawdown(self, trades):
        # 计算最大回撤
        equity_curve = []  # 账户权益曲线
        cumulative = 0  # 累计收益
        for t in sorted(trades, key=lambda x: x.open_date):  # 按交易时间排序
            cumulative += t.calc_profit_ratio(t.close_rate)  # 累加收益
            equity_curve.append(cumulative)
        
        peak = -np.inf  # 初始化峰值
        max_dd = 0  # 初始化最大回撤
        for value in equity_curve:
            if value > peak:
                peak = value  # 更新峰值
            dd = peak - value  # 计算回撤
            if dd > max_dd:
                max_dd = dd  # 更新最大回撤
        return max_dd

    def get_current_atr(self, pair):
        # 获取当前ATR值
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return dataframe['atr'].iat[-1]  # 返回最后一个ATR值

    def version(self) -> str:
        # 返回策略版本号
        return "AdaptiveThreeLineBreak v2.1"