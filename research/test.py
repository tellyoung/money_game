import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.utils import init_instance_by_config
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
import pandas as pd


# 初始化 Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/crypto_data", region=REG_CN)

# 数据处理
# 使用 Alpha158 数据处理器
data_handler_config = {
    "start_time": "2020-01-01",
    "end_time": "2023-12-31",
    "fit_start_time": "2020-01-01",
    "fit_end_time": "2022-12-31",
    "instruments": "all",
}
handler = Alpha158(**data_handler_config)

# 数据集配置
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": handler,
        "segments": {
            "train": ("2020-01-01", "2022-12-31"),
            "valid": ("2023-01-01", "2023-06-30"),
            "test": ("2023-07-01", "2023-12-31"),
        },
    },
}
dataset = init_instance_by_config(dataset_config)

# 模型训练
# 使用 LightGBM 模型
model = LGBModel(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.0421,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
)

# 训练模型
model.fit(dataset)

# 模型预测
pred = model.predict(dataset)

# 回测部分
# 定义策略
STRATEGY_CONFIG = {
    "topk": 30,
    "n_drop": 3,
    "signal": pred.to_frame("score"),
}
strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

# 执行回测
start_time = pred.index.get_level_values("datetime").min()
end_time = pred.index.get_level_values("datetime").max()
report_normal, positions_normal = backtest_daily(
    start_time=start_time,
    end_time=end_time,
    strategy=strategy_obj,
    executor=None,
    account=1e8,
    benchmark="SH000300",
    exchange_kwargs={
        "freq": "day",
        "limit_threshold": None,
        "deal_price": None,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
    pos_type="Position"
)

# 打印回测结果
print("回测报告:")
print(report_normal)
print("持仓情况:")
print(positions_normal)