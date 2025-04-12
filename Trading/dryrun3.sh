clear

source activate
conda activate freqtrade


freqtrade trade \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_NewStrategy53_22-dr.json \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
--strategy NewStrategy53_22 \
--strategy-path /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_strategies/NewStrategy53_22 \
--recursive-strategy-search \
--db-url sqlite:////Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/DB/NewStrategy53_22_dryrun3/debug_dryrun.sqlite


