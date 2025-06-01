import qlib
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.task import Task
from qlib.model.trainer import Trainer
from qlib.backtest import backtest
import pandas as pd

# 初始化 Qlib 环境
qlib.init(provider_uri="/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/qlib")

# 数据获取
class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_feather(self.data_path)

# 因子挖掘
class FactorGenerator:
    def __init__(self, data):
        self.data = data

    def generate_factors(self):
        self.data['factor1'] = self.data['close'] / self.data['open']
        self.data['factor2'] = self.data['volume'] / self.data['high']
        return self.data

# 模型训练
class ModelTrainer:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        task = Task(model="gbdt", dataset=self.data)
        trainer = Trainer(task)
        model = trainer.train()
        return model

# 回测
class Backtester:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def run_backtest(self):
        report = backtest(self.model, self.data)
        return report

if __name__ == "__main__":
    # 数据路径
    data_path = "/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_20250101_20250201/ADA_USDT-1h.feather"

    # 数据加载
    loader = DataLoader(data_path)
    raw_data = loader.load_data()

    # 因子生成
    factor_gen = FactorGenerator(raw_data)
    factor_data = factor_gen.generate_factors()

    # 模型训练
    trainer = ModelTrainer(factor_data)
    model = trainer.train_model()

    # 回测
    backtester = Backtester(model, factor_data)
    backtest_report = backtester.run_backtest()

    print("回测报告:", backtest_report)