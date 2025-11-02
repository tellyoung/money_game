clear

source activate
conda activate freqtrade


freqtrade trade \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/open03/config_open03_futures.json \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--strategy open03_630 \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
--db-url sqlite:////Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/db/open03_debug.sqlite


