clear


freqtrade hyperopt \
--userdir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data \
--datadir /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Data/binance/Vol_top20_futrue_20200101_20250417 \
--strategy-path /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/strategies \
--recursive-strategy-search \
--config /Users/yutieyang/Documents/yuty/yuty_projects/money_game/Trading/user_data/configs/config_test_futrue_IntradayMomentum.json \
--strategy IntradayMomentum \
--hyperopt-loss MaxDrawDownHyperOptLoss \
--epochs 666 \
--spaces buy sell roi stoploss trailing trades \
--timerange 20240101-20250417 \



# --hyperopt-loss SharpeHyperOptLoss \
# ProfitDrawDownHyperOptLoss
# MaxDrawDownHyperOptLoss
# MaxDrawDownRelativeHyperOptLoss