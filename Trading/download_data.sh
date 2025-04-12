clear

source activate
conda activate freqtrade

freqtrade download-data \
    --config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/notebooks/open_strategy/os-1/config_os1.json \
    --userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
    --datadir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/data/2024_top10 \
	--exchange binance \
	--prepend --timerange 20240101-20241231 \
	--timeframes 1m 5m 15m 1h
# --timeframes 1m 5m 15m 1h \
    # --config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_bt.json \








