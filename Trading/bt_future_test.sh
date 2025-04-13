clear

# conda activate freqtrade


freqtrade backtesting \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_IntradayMomentum.json \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--strategy IntradayMomentum \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20250101_20250411 \
--breakdown week \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
# --dry-run-wallet 100 \



# ================= TrendFollowingStrategy ===========================
# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_TrendFollowingStrategy.json \
# --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
# --strategy TrendFollowingStrategy \
