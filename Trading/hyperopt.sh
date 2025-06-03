clear
source activate
conda activate freqtrade

freqtrade hyperopt \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_spot.json \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Datasets/binance/Vol_top20_futrue_202501_202505 \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
--strategy open01_yuty \
--hyperopt-loss ProfitDrawDownHyperOptLoss \
--epochs 200 \
--spaces buy sell \
--timerange 20250101-20250417 \


# --spaces buy sell roi stoploss trailing trades \

# --hyperopt-loss SharpeHyperOptLoss \
# ProfitDrawDownHyperOptLoss
# MaxDrawDownHyperOptLoss
# MaxDrawDownRelativeHyperOptLoss

# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_TrendFollowingStrategy.json \
# --strategy TrendFollowingStrategy \

# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_IntradayMomentum.json \
# --strategy IntradayMomentum \

