clear
source activate
conda activate freqtrade

freqtrade hyperopt \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/open03/config_open03_futures.json \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Datasets/binance/Vol_top20_202501_202506 \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
--strategy MS0623 \
--hyperopt-loss ProfitDrawDownHyperOptLoss \
--epochs 100 \
--spaces buy sell \
--timerange 20250101-20250613

# --spaces buy sell roi stoploss trailing trades \

# --hyperopt-loss SharpeHyperOptLoss \
# ProfitDrawDownHyperOptLoss
# MaxDrawDownHyperOptLoss
# MaxDrawDownRelativeHyperOptLoss

# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_TrendFollowingStrategy.json \
# --strategy TrendFollowingStrategy \

# --config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_IntradayMomentum.json \
# --strategy IntradayMomentum \

