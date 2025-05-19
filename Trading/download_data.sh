clear
source activate
conda activate freqtrade

# freqtrade download-data \
#     --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test.json \
#     --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
#     --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_20250101_20250201 \
# 	--exchange binance \
# 	--prepend --timerange 20250101-20250201 \
# 	--timeframes 1m 5m 15m 1h


# download future data
freqtrade download-data \
    --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_IntradayMomentum.json \
    --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
    --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20250418_20250517 \
	--exchange binance \
	--timerange 20250418-20250517 \
	--timeframes 5m 15m 1h 8h --prepend 
    # 1m 5m 15m 1h
    # --prepend 







