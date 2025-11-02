clear

source activate
conda activate freqtrade

# freqtrade plot-dataframe --strategy open03 \
#     --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/open03/config_open03.json \
#     --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
#     --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Datasets/binance/Vol_top20_202501_202506 \
#     --export-filename /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/backtest_results/backtest-result-2025-06-09_22-26-35.meta.json \
#     -p BTC/USDT


freqtrade plot-profit \
    --export-filename /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/backtest_results/backtest-result-2025-06-14_23-24-23/backtest-result-2025-06-14_23-24-23.json \
    --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/open03/config_open03_spot.json \
    --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
    --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Datasets/binance/Vol_top20_202501_202506 \
    -p RAY/USDT FUN/USDT ADA/USDT \
