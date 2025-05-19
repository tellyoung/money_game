clear

source activate
conda activate freqtrade

# ================= IntradayMomentum ===========================
# freqtrade backtesting \
# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_IntradayMomentum.json \
# --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
# --strategy IntradayMomentum \
# --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20200101_20250417 \
# --breakdown year month \
# --strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
# --recursive-strategy-search \
# --timerange 20240101-20250417 \
# --dry-run-wallet 1000 \


# ================= TrendFollowingStrategy ===========================
# freqtrade backtesting \
# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_TrendFollowingStrategy.json \
# --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
# --strategy TrendFollowingStrategy_base \
# --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20200101_20250417 \
# --breakdown month \
# --strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
# --recursive-strategy-search \
# --timerange 20250101-20250417 \



# ==============================================================
# freqtrade backtesting \
# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_futrue_test.json \
# --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
# --strategy yutyStrategy01 \
# --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20200101_20250417 \
# --breakdown month \
# --strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
# --recursive-strategy-search \
# --timerange 20250101-20250417 \
# --enable-protections
# ==============================================================


freqtrade backtesting \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_futrue_test.json \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--strategy yutyStrategy02 \
--breakdown week \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20200101_20250417 \
--timerange 20250101-20250417 


