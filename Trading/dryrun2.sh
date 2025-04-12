clear

source activate
conda activate freqtrade


freqtrade trade \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_TheForce-dr.json \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
--strategy FSupertrendStrategy \
--strategy-path /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/freqtrade-strategies/user_data/strategies/futures \
--recursive-strategy-search \
--db-url sqlite:////Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/DB/FSupertrendStrategy_dryrun2/debug_dryrun.sqlite


