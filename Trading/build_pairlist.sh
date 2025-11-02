clear

source activate
conda activate freqtrade


freqtrade test-pairlist \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/open03/config_future_open03.json \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--quote USDT \
--print-json 