# These options apply to how the script will operate.
script_options:
  # Switch between testnet and mainnet
  # Setting this to False will use REAL funds, use at your own risk
  TEST_MODE: True
  LOG_TRADES: True
  LOG_FILE: 'trades.txt'
  DEBUG: False

  # Set this to true if you are accessing binance from within the United States of America
  # Need to change TLD
  AMERICAN_USER: False


# These options apply to the trading methods the script executes
trading_options:

  # select your base currency to use for trading (trade for example USDT pairs)
  PAIR_WITH: USDT

  # Total amount per trade (your base currency balance must be at least MAX_COINS * QUANTITY)
  # Binance uses a minimum of 10 USDT per trade, add a bit extra to enable selling if the price drops.
  # Recommended: no less than 12 USDT. Suggested: 15 or more.
  QUANTITY: 500

  # List of trading pairs to exclude
  # by default we're excluding the most popular fiat pairs
  FIATS:
    - EURUSDT
    - GBPUSDT
    - JPYUSDT
    - USDUSDT
    - DOWN
    - UP

  # Name of custom tickers list
  TICKERS_LIST: 'tickers.txt'

   # Maximum number of coints to hold
  MAX_COINS: 10

  # the amount of time in MINUTES to calculate the differnce from the current price
  TIME_DIFFERENCE: 1

  # Numer of times to check for TP/SL during each TIME_DIFFERENCE Minimum 1
  RECHECK_INTERVAL: 6

  # the difference in % between the first and second checks for the price.
  CHANGE_IN_PRICE: 0.25

  # define in % when to sell a coin that's not making a profit
  STOP_LOSS: 5

  # define in % when to take profit on a profitable coin
  TAKE_PROFIT: 0.75

  # Use custom tickers.txt list for filtering pairs
  CUSTOM_LIST: True

  # whether to use trailing stop loss or not; default is True
  USE_TRAILING_STOP_LOSS: True

  # when hit TAKE_PROFIT, move STOP_LOSS to TRAILING_STOP_LOSS percentage points below TAKE_PROFIT hence locking in profit
  # when hit TAKE_PROFIT, move TAKE_PROFIT up by TRAILING_TAKE_PROFIT percentage points
  TRAILING_STOP_LOSS: 0.1
  TRAILING_TAKE_PROFIT: 0.3

  # Trading fee in % per trade.
  # If using 0.075% (using BNB for fees) you must have BNB in your account to cover trading fees.
  # If using BNB for fees, it MUST be enabled in your Binance 'Dashboard' page (checkbox).
  TRADING_FEE: 0.075

  SIGNALLING_MODULES:
    - pausebotmod
    - signalsamplemod
    - custsignalmod
  #  - djcommie_signalsell_rsi_stoch
  #  - djcommie_signalbuy_rsi_stoch

