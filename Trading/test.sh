#
# /Users/yutieyang/Documents/yuty_projects/freqtrade/user_data
#
source activate
conda activate freqtrade

freqtrade trade \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/user_data/config.json \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/user_data \
--db-url sqlite:////Users/yutieyang/Documents/yuty_projects/freqtrade/user_data/DB/tradesv3.dry_run.sqlite \
--strategy SampleStrategy