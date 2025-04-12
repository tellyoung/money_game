clear

source activate
conda activate freqtrade


freqtrade test-pairlist \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_future_bt.json \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/data/2024 \
--quote USDT \
--print-json 