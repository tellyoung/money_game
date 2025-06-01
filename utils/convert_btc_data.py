import pandas as pd

# 定义 Qlib 数据格式转换函数
def convert_to_qlib_format(input_path, output_path):
    # 读取原始数据
    df = pd.read_feather(input_path)

    # 检查原始数据结构
    print("原始数据列:", df.columns)

    # 假设原始数据包含以下列: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    # 转换为 Qlib 格式，重命名列
    df = df.rename(columns={
        'timestamp': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    # 确保时间戳格式正确
    df['datetime'] = pd.to_datetime(df['date'])

    # 设置索引为时间戳
    df = df.set_index('datetime')

    # 保存为 Qlib 格式的 CSV 文件
    df.to_csv(output_path)
    print(f"数据已保存到 {output_path}")

if __name__ == "__main__":
    input_path = "/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/qlib/binance/BTC_USDT-1h.feather"
    output_path = "/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/qlib/binance/BTC_USDT-1h.csv"
    convert_to_qlib_format(input_path, output_path)
