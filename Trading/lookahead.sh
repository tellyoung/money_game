clear

source activate
conda activate freqtrade


# ============  ===========================
freqtrade lookahead-analysis \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Datasets/binance/Vol_top20_futrue_202501_202505 \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_spot.json \
--dry-run-wallet 1000 \
--strategy open01 \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--timerange 20250101-20250601 