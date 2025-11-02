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
    --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/open03/config_open03_futures.json \
    --userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
    --datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Datasets/binance/Vol_top20_202501_202506_down \
	--exchange binance \
	--timerange 20250610-20250708 \
	--timeframes 5m 15m 1h 4h 8h 1d --prepend 


    # --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_spot.json \





