import pandas as pd
import qlib
from qlib.data import D

# 读取 .feather 文件
feather_data = pd.read_feather('your_file.feather')

# 假设你的数据中包含 'date' 列，将其转换为 datetime 类型
feather_data['datetime'] = pd.to_datetime(feather_data['date'])

# 假设你的数据中有一个 'symbol' 列作为 instrument 标识
feather_data.rename(columns={'symbol': 'instrument'}, inplace=True)

# 按照 datetime 排序
feather_data = feather_data.sort_values(by='datetime')

# 设置 MultiIndex
feather_data.set_index(['datetime', 'instrument'], inplace=True)

# 初始化 Qlib
qlib.init(provider_uri="/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/qlib/crypto_data", region="cn")

# 假设你要保存的数据列
fields = ['open', 'high', 'low', 'close', 'volume']

# 将数据保存到 Qlib
for instrument, instrument_data in feather_data.groupby(level='instrument'):
    for field in fields:
        D.save_feature(instrument, field, instrument_data[field])