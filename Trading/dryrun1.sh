clear

source activate
conda activate freqtrade


freqtrade trade \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_TheForce-dr.json \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
--strategy TheForce \
--strategy-path /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_strategies/TheForce \
--db-url sqlite:////Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/DB/TheForce_dryrun1/debug_dryrun.sqlite


