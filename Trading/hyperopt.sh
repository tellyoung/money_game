clear
source activate
conda activate freqtrade

freqtrade hyperopt \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20200101_20250417 \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_futrue_test.json \
--strategy yutyStrategy02 \
--hyperopt-loss MaxDrawDownHyperOptLoss \
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