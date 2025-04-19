import pandas as pd

# 定义Feather文件的路径
file_path = '/Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/backtest_results/backtest-result-2025-04-18_23-11-40/backtest-result-2025-04-18_23-11-40_market_change.feather'

try:
    # 读取Feather文件
    df = pd.read_feather(file_path)
    print("成功读取Feather文件！")
    print("数据基本信息：")
    df.info()
    # 显示数据集行数和列数
    rows, columns = df.shape
    if rows < 20:
        # 行数少于20则全部打印
        print("数据全部内容信息：")
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        # 行数多于20则打印数据前几行信息
        print("数据前几行内容信息：")
        print(df.head().to_csv(sep='\t', na_rep='nan'))

except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}。")
except Exception as e:
    print(f"发生未知错误：{e}")