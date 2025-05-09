clear

source activate
conda activate freqtrade


# ============  ===========================
freqtrade backtesting \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_20250101_20250201 \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_bt.json \
--strategy NewStrategy53_22 \
--strategy-path /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_strategies/NewStrategy53_22 \
--recursive-strategy-search \
--dry-run-wallet 100 \
--breakdown month week \


# --datadir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/data/2024_top10 \


# ============ future ===========================
# freqtrade backtesting \
# --userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
# --datadir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/data/2024_top10 \
# --dry-run-wallet 100 \
# --breakdown month week \
# --config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_future_bt.json \
# --strategy VolatilitySystem \
# --strategy-path /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/freqtrade-strategies/user_data/strategies/futures \
# --recursive-strategy-search

