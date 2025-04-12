clear

source activate
conda activate freqtrade


# ============  ===========================
freqtrade lookahead-analysis \
--userdir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data \
--datadir /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/data/2025_01_02 \
--dry-run-wallet 100 \
--config /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_data/configs/config_bt.json \
--strategy NewStrategy53_22 \
--strategy-path /Users/yutieyang/Documents/yuty_projects/freqtrade/yuty_space/yuty_strategies/NewStrategy53_22 \
--timerange 20250101-20250228