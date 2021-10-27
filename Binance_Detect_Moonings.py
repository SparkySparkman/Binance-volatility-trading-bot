"""
Disclaimer

All investment strategies and investments involve risk of loss.
Nothing contained in this program, scripts, code or repositoy should be
construed as investment advice.Any reference to an investment's past or
potential performance is not, and should not be construed as, a recommendation
or as a guarantee of any specific outcome or profit.

By using this program you accept all liabilities,
and that no claims can be made against the developers,
or others connected with the program.
"""


# use for environment variables
import os
import math
from decimal import Decimal

# use if needed to pass args to external modules
import sys

# used to create threads & dynamic loading of modules
import threading
import multiprocessing
import importlib

# used for directory handling
import glob

#discord needs import request
import requests

# Needed for colorful console output Install with: python3 -m pip install colorama (Mac/Linux) or pip install colorama (PC)
from colorama import init
init()

# needed for the binance API / websockets / Exception handling
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.helpers import round_step_size
from requests.exceptions import ReadTimeout, ConnectionError

# used for dates
from datetime import date, datetime, timedelta
import time

# used to repeatedly execute the code
from itertools import count

# used to store trades and sell assets
import json

# Load helper modules
from helpers.parameters import (
    parse_args, load_config
)

# Load creds modules
from helpers.handle_creds import (
    load_correct_creds, test_api_key
)


# for colourful logging to the console
class txcolors:
    BUY = '\033[92m'
    WARNING = '\033[93m'
    SELL_LOSS = '\033[91m'
    SELL_PROFIT = '\033[32m'
    DIM = '\033[2m\033[35m'
    DEFAULT = '\033[39m'


# print with timestamps
old_out = sys.stdout
class St_ampe_dOut:
    """Stamped stdout."""
    nl = True
    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write(f'{txcolors.DIM}[{str(datetime.now().replace(microsecond=0))}]{txcolors.DEFAULT} {x}')
            self.nl = False
        else:
            old_out.write(x)

    def flush(self):
        pass

sys.stdout = St_ampe_dOut()

# tracks profit/loss each session
global session_profit, trade_wins, trade_losses, sellall, PriceChange, Pending_sum, Pending_perc
global WIN_LOSS_PERCENT, coins_up, coins_down, coins_unchanged

session_profit = 0
bot_started_datetime = datetime.now()
trade_wins = 0
trade_losses = 0
sellall = ''
PriceChange = 0.0
Pending_sum = 0.0
Pending_perc = 0.0
WIN_LOSS_PERCENT = 0.0
coins_up = 0
coins_down = 0
coins_unchanged = 0

def stop_signal_threads():

    try:
        for signalthread in signalthreads:
            print(f'Terminating thread {str(signalthread.name)}')
            signalthread.terminate()
    except:
        pass

def get_price(add_to_historical=True):
    '''Return the current price for all coins on binance'''

    global historical_prices, hsp_head

    initial_price = {}
    prices = client.get_all_tickers()
    for coin in prices:

        if CUSTOM_LIST:
            if any(item + PAIR_WITH == coin['symbol'] for item in tickers) and all(item not in coin['symbol'] for item in FIATS):
                initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}
        else:
            if PAIR_WITH in coin['symbol'] and all(item not in coin['symbol'] for item in FIATS):
                initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}

    if add_to_historical:
        hsp_head += 1

        if hsp_head == RECHECK_INTERVAL:
            hsp_head = 0

        historical_prices[hsp_head] = initial_price

    return initial_price

def print_stats(PriceChange,Pending_sum,Pending_perc,coins_up,coins_down,coins_unchanged):
  if trade_wins+trade_losses > 0:
    Profit_Per_Trade = (session_profit/(trade_wins+trade_losses))
  else:
    Profit_Per_Trade = 0

  print(f'')
  print(f'--------')
  print(f'')
  print(f'Working...')

  # https://algotrading101.com/learn/binance-python-api-guide/
  # Gets all the coin balances on Binance
  # print(client.get_account())

  if TEST_MODE:
    print(f'Mode            : Test mode, {txcolors.SELL_PROFIT}no real money is used')
  else:
    print(f'Mode            : {txcolors.WARNING}You are using REAL money!')

  print(f'Session profit  : {txcolors.SELL_PROFIT if session_profit > 0. else txcolors.SELL_LOSS}{session_profit:.2f}% Est : ${(QUANTITY * session_profit)/100:.2f}, ~{Profit_Per_Trade:.2f}% per trade {txcolors.DEFAULT}')
  print(f'Pending profit  :{txcolors.SELL_PROFIT if Pending_perc > 0. else txcolors.SELL_LOSS} {Pending_perc:.2f}%, {Pending_sum:.2f} USDT{txcolors.DEFAULT}')
  print(f'Overall profit  :{txcolors.SELL_PROFIT if session_profit+Pending_perc > 0. else txcolors.SELL_LOSS} {(session_profit+Pending_perc):.2f}%, {(QUANTITY * (session_profit+Pending_perc))/100:.2f} USDT')
  print(f'Trades total    : {trade_wins+trade_losses}, Wins {trade_wins}, Losses {trade_losses}, Win ratio {WIN_LOSS_PERCENT}%')
  print('Last sell       :',last_sell_datetime)
  print(f'Started         : {bot_started_datetime} | Run time : {datetime.now() - bot_started_datetime}')
  if MAX_COINS > 0:
    print(f'Coins Currently : {len(coins_bought)}/{MAX_COINS} ({float(len(coins_bought)*QUANTITY):g}/{float(MAX_COINS*QUANTITY):g} {PAIR_WITH})')
  else:
    print(f'Coins Currently : {len(coins_bought)}/0 ({float(len(coins_bought)*QUANTITY):g}/0 {PAIR_WITH})')

  print(f'Coin\'s Status   : {txcolors.SELL_PROFIT}Up {coins_up}, {txcolors.SELL_LOSS}Down: {coins_down}{txcolors.DEFAULT}, Unchanged: {coins_unchanged}')
  print(f'Stop Loss       : {STOP_LOSS}%')
  print(f'Take Profit     : {TAKE_PROFIT}%')
  print('Use TSL         :',USE_TRAILING_STOP_LOSS, end = '')
  if USE_TRAILING_STOP_LOSS: print(f', TSL {TRAILING_STOP_LOSS}%, TTP {TRAILING_TAKE_PROFIT}%')
  else: print(f'')
  print('Used sig. mod(s):', end=" ")
  if len(SIGNALLING_MODULES) > 0:
    #for module in SIGNALLING_MODULES:
      #print(module, end=" ")
    print(*tuple(module for module in SIGNALLING_MODULES))
  #print(f'')
  print(f'')
  print(f'--------')
  print(f'')


def wait_for_price():
    '''calls the initial price and ensures the correct amount of time has passed
    before reading the current price again'''

    global historical_prices, hsp_head, volatility_cooloff, WIN_LOSS_PERCENT
    global coins_up,coins_down,coins_unchanged

    volatile_coins = {}
    externals = {}

    coins_up = 0
    coins_down = 0
    coins_unchanged = 0

    WIN_LOSS_PERCENT = 0

    pause_bot()

    if historical_prices[hsp_head]['BNB' + PAIR_WITH]['time'] > datetime.now() - timedelta(minutes=float(TIME_DIFFERENCE / RECHECK_INTERVAL)):

        # sleep for exactly the amount of time required
        time.sleep((timedelta(minutes=float(TIME_DIFFERENCE / RECHECK_INTERVAL)) - (datetime.now() - historical_prices[hsp_head]['BNB' + PAIR_WITH]['time'])).total_seconds())

    # truncating some of the above values to the correct decimal places before printing
    if (trade_wins > 0) and (trade_losses > 0):
        WIN_LOSS_PERCENT = round((trade_wins / (trade_wins+trade_losses)) * 100, 2)
    if (trade_wins > 0) and (trade_losses == 0):
        WIN_LOSS_PERCENT = 100
    #print(f'Wins :  {trade_wins}, Losses :  {trade_losses}, {WIN_LOSS_PERCENT}% ')

    load_settings()

    # retreive latest prices
    get_price()

    if MAX_COINS < 1:
      print(f'')
      print(f'{txcolors.WARNING}MAX_COINS is set to zero or below({MAX_COINS}), no coins will be bought.')
      print(f'{txcolors.WARNING}If you want the bot to buy more coins, set MAX_COINS > {len(coins_bought) if len(coins_bought) > 0. else 0} and save settings file !')

    if MAX_COINS == -1:
      print(f'')
      print(f'{txcolors.WARNING}The bot is set to terminate after all the coins are sold')
      if len(coins_bought) == 0:
        print(f'')
        print(f'{txcolors.WARNING}All the coins are sold, terminating the bot now')
        # stop external signal threads
        stop_signal_threads()
        sys.exit(0)

    # calculate the difference in prices
    for coin in historical_prices[hsp_head]:

        # minimum and maximum prices over time period
        min_price = min(historical_prices, key = lambda x: float("inf") if x is None else float(x[coin]['price']))
        max_price = max(historical_prices, key = lambda x: -1 if x is None else float(x[coin]['price']))

        threshold_check = (-1.0 if min_price[coin]['time'] > max_price[coin]['time'] else 1.0) * (float(max_price[coin]['price']) - float(min_price[coin]['price'])) / float(min_price[coin]['price']) * 100

        # each coin with higher gains than our CHANGE_IN_PRICE is added to the volatile_coins dict if less than MAX_COINS is not reached.
        if threshold_check > CHANGE_IN_PRICE:
            coins_up +=1

            if coin not in volatility_cooloff:
                volatility_cooloff[coin] = datetime.now() - timedelta(minutes=TIME_DIFFERENCE)

            # only include coin as volatile if it hasn't been picked up in the last TIME_DIFFERENCE minutes already
            if datetime.now() >= volatility_cooloff[coin] + timedelta(minutes=TIME_DIFFERENCE):
                volatility_cooloff[coin] = datetime.now()

                if len(coins_bought) + len(volatile_coins) < MAX_COINS: #  or MAX_COINS == 0
                    volatile_coins[coin] = round(threshold_check, 3)
                    print(f'{coin} has gained {volatile_coins[coin]}% within the last {TIME_DIFFERENCE} minute(s), calculating volume in {PAIR_WITH}')

                else:
                    if MAX_COINS > 0:
                      print(f'{txcolors.WARNING}{coin} has gained {round(threshold_check, 3)}% within the last {TIME_DIFFERENCE} minute(s), but you are holding max number of coins{txcolors.DEFAULT}')

        elif threshold_check < CHANGE_IN_PRICE:
            coins_down +=1

        else:
            coins_unchanged +=1

    # Here goes new code for external signalling
    externals = external_signals()
    exnumber = 0

    for excoin in externals:
        if excoin not in volatile_coins and excoin not in coins_bought and \
                (len(coins_bought) + exnumber + len(volatile_coins)) < MAX_COINS:
            volatile_coins[excoin] = 1
            exnumber +=1
            print(f'External signal received on {excoin}, calculating volume in {PAIR_WITH}')

    print_stats(PriceChange,Pending_sum,Pending_perc,coins_up,coins_down,coins_unchanged)

    return volatile_coins, len(volatile_coins), historical_prices[hsp_head]


def external_signals():
    external_list = {}
    signals = {}

    # check directory and load pairs from files into external_list
    signals = glob.glob("signals/*.exs")
    for filename in signals:
        for line in open(filename):
            symbol = line.strip()
            external_list[symbol] = symbol
        try:
            os.remove(filename)
        except:
            if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file{txcolors.DEFAULT}')

    return external_list


def pause_bot():
    '''Pause the script when exeternal indicators detect a bearish trend in the market'''
    global bot_paused, session_profit, hsp_head

    # start counting for how long the bot's been paused
    start_time = time.perf_counter()

    while os.path.isfile("signals/paused.exc"):

        if bot_paused == False:
            print(f'{txcolors.WARNING}Pausing buying due to change in market conditions, stop loss and take profit will continue to work...{txcolors.DEFAULT}')
            bot_paused = True

        # Sell function needs to work even while paused
        coins_sold = sell_coins()
        remove_from_portfolio(coins_sold)
        get_price(True)

        # pausing here
        if hsp_head == 1:
          print(f'')
          print(f'Paused...')
          #Session profit : {session_profit:.2f}% Est : ${(QUANTITY * session_profit)/100:.2f}')
          print_stats(PriceChange,Pending_sum,Pending_perc,coins_up,coins_down,coins_unchanged)
          time.sleep((TIME_DIFFERENCE * 60) / RECHECK_INTERVAL)

    else:
        # stop counting the pause time
        stop_time = time.perf_counter()
        time_elapsed = timedelta(seconds=int(stop_time-start_time))

        # resume the bot and ser pause_bot to False
        if  bot_paused == True:
            print(f'{txcolors.WARNING}Resuming buying due to change in market conditions, total sleep time : {time_elapsed}{txcolors.DEFAULT}')
            bot_paused = False

    return


def convert_volume():
    '''Converts the volume given in QUANTITY from USDT to the each coin's volume'''

    volatile_coins, number_of_coins, last_price = wait_for_price()
    lot_size = {}
    volume = {}

    for coin in volatile_coins:

        # Find the correct step size for each coin
        # max accuracy for BTC for example is 6 decimal points
        # while XRP is only 1
        try:
            info = client.get_symbol_info(coin)
            step_size = info['filters'][2]['stepSize']
            lot_size[coin] = step_size.index('1') - 1

            if lot_size[coin] < 0:
                lot_size[coin] = 0

        except:
            pass

        # calculate the volume in coin from QUANTITY in USDT (default)
        volume[coin] = float(QUANTITY / float(last_price[coin]['price']))

        # define the volume with the correct step size
        if coin not in lot_size:
            volume[coin] = float('{:.1f}'.format(volume[coin]))

        else:
            # if lot size has 0 decimal points, make the volume an integer
            if lot_size[coin] == 0:
                volume[coin] = int(volume[coin])
            else:
                volume[coin] = float('{:.{}f}'.format(volume[coin], lot_size[coin]))

    return volume, last_price

def dropzeros(number):
    mynum = Decimal(number).normalize()
    # e.g 22000 --> Decimal('2.2E+4')
    return mynum.__trunc__() if not mynum % 1 else float(mynum)

def remove_zeros(num):
    nums = list(num)
    indexes = (list(reversed(range(len(nums)))))
    for i in indexes:
        if nums[i] == '0':
            del nums[-1]
        else:
            break
    return "".join(nums)

def buy():
    '''Place Buy market orders for each volatile coin found'''
    volume, last_price = convert_volume()
    orders = {}

    for coin in volume:

        # only buy if the there are no active trades on the coin
        if coin not in coins_bought:
            volume[coin] = math.floor(volume[coin]*10000)/10000
            print(f"{txcolors.BUY}Preparing to buy {volume[coin]} {coin}{txcolors.DEFAULT}")

            if TEST_MODE:
                orders[coin] = [{
                    'symbol': coin,
                    'orderId': 0,
                    'time': datetime.now().timestamp()
                }]

                # Log trade
                if LOG_TRADES:
                    write_log(f"Buy : {volume[coin]} {coin} - {last_price[coin]['price']}")

                continue

            # try to create a real order if the test orders did not raise an exception
            try:
                buy_limit = client.create_order(
                    symbol = coin,
                    side = 'BUY',
                    type = 'MARKET',
                    quantity = volume[coin]
                )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if the position has been placed and return order info
            else:
                orders[coin] = client.get_all_orders(symbol=coin, limit=1)

                # binance sometimes returns an empty list, the code will wait here until binance returns the order
                while orders[coin] == []:
                    print('Binance is being slow in returning the order, calling the API again...')

                    orders[coin] = client.get_all_orders(symbol=coin, limit=1)
                    time.sleep(1)

                else:
                    print('Order returned, saving order to file')

                    # Log trade
                    if LOG_TRADES:
                        write_log(f"Buy : {volume[coin]} {coin} - {last_price[coin]['price']}")


        else:
            print(f'Buy signal detected, but there is already an active trade on {coin}')

    return orders, last_price, volume

def sell_all(msgreason, session_tspl_ovr = False):
    global sell_all_coins, PriceChange

    #msg_discord(f'SELL ALL COINS: {msgreason}')

    # stop external signals so no buying/selling/pausing etc can occur
    stop_signal_threads()

    # sell all coins NOW!
    sell_all_coins = True

    coins_sold = sell_coins(session_tspl_ovr)
    remove_from_portfolio(coins_sold)

    # display final info to screen
    last_price = get_price()
    discordmsg = balance_report(last_price)
    msg_discord(discordmsg)

def sell_all_coins(msg=''):
  global session_tspl_ovr
  with open(coins_bought_file_path, 'r') as f:
    coins_bought = json.load(f)
    total_profit = 0
    total_price_change = 0

    if not TEST_MODE:
      for coin in list(coins_bought):
        sell_coin = client.create_order(
            symbol = coin,
            side = 'SELL',
            type = 'MARKET',
            quantity = coins_bought[coin]['volume'])

        BuyPrice = float(coins_bought[coin]['bought_at'])
        LastPrice = float(sell_coin['fills'][0]['price'])
        profit = (LastPrice - BuyPrice) * coins_bought[coin]['volume']
        PriceChange = float((LastPrice - BuyPrice) / BuyPrice * 100)

        total_profit += profit
        total_price_change += PriceChange

        coins_sold = sell_coins(session_tspl_ovr)
        remove_from_portfolio(coins_sold)

        text_color = txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS
        console_log_text = f"{text_color}Sell: {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} Profit: {profit:.2f} {PriceChange:.2f}%{txcolors.DEFAULT}"
        print(console_log_text)

        if LOG_TRADES:
            timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
            write_log(f"Sell: {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} Profit: {profit:.2f} {PriceChange:.2f}%")

    total_profit += Pending_sum
    total_price_change += Pending_perc
    print(f'')
    print(f'Pending Profit: {Pending_perc}%, {Pending_sum} USDT')
    text_color = txcolors.SELL_PROFIT if total_price_change >= 0. else txcolors.SELL_LOSS
    print(f"Total Profit: {text_color}{total_profit:.2f}{txcolors.DEFAULT}. Total Price Change: {text_color}{total_price_change:.2f}%{txcolors.DEFAULT}")

  with open(coins_bought_file_path, 'r') as f:
    coins_bought = json.load(f)

  #coins_bought = {}
  #os.remove(coins_bought_file_path)


def sell_coins():
    '''sell coins that have reached the STOP LOSS or TAKE PROFIT threshold'''

    global hsp_head, session_profit, trade_wins, trade_losses, Pending_sum, Pending_perc, PriceChange, last_sell_datetime

    Pending_sum = 0.0
    Pending_perc = 0.0

    last_price = get_price(False) # don't populate rolling window
    #last_price = get_price(add_to_historical=True) # don't populate rolling window
    coins_sold = {}
    #print(f'')

    for coin in list(coins_bought):
        # define stop loss and take profit
        TP = float(coins_bought[coin]['bought_at']) + (float(coins_bought[coin]['bought_at']) * coins_bought[coin]['take_profit']) / 100
        SL = float(coins_bought[coin]['bought_at']) + (float(coins_bought[coin]['bought_at']) * coins_bought[coin]['stop_loss']) / 100

        LastPrice = float(last_price[coin]['price'])
        BuyPrice = float(coins_bought[coin]['bought_at'])
        PriceChange = float((LastPrice - BuyPrice) / BuyPrice * 100)
        Pending_perc += PriceChange-(TRADING_FEE*2)
        Pending_sum += (QUANTITY*(PriceChange-(TRADING_FEE*2)))/100

        # check that the price is above the take profit and readjust SL and TP accordingly if trailing stop loss used
        if LastPrice > TP and USE_TRAILING_STOP_LOSS:

            # increasing TP by TRAILING_TAKE_PROFIT (essentially next time to readjust SL)
            coins_bought[coin]['take_profit'] = PriceChange + TRAILING_TAKE_PROFIT
            coins_bought[coin]['stop_loss'] = coins_bought[coin]['take_profit'] - TRAILING_STOP_LOSS
            if DEBUG: print(f"{coin} TP reached, adjusting TP {coins_bought[coin]['take_profit']:.2f}  and SL {coins_bought[coin]['stop_loss']:.2f} accordingly to lock-in profit, , SL {SL}, TP {TP}")
            continue

        # check that the price is below the stop loss or above take profit (if trailing stop loss not used) and sell if this is the case
        if ((LastPrice < SL or LastPrice > TP) and not USE_TRAILING_STOP_LOSS) or (LastPrice < SL and USE_TRAILING_STOP_LOSS):
            print(f"{txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}TP or SL reached, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange-(TRADING_FEE*2):.2f}% Est:${(QUANTITY*(PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}, SL {SL:.2f}, TP {TP:.2f}")

            last_sell_datetime = datetime.now()
            # try to create a real order
            try:

                if not TEST_MODE:
                    sell_coins_limit = client.create_order(
                        symbol = coin,
                        side = 'SELL',
                        type = 'MARKET',
                        quantity = coins_bought[coin]['volume']

                    )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if coin has been sold and create a dict for each coin sold
            else:
                coins_sold[coin] = coins_bought[coin]

                # prevent system from buying this coin for the next TIME_DIFFERENCE minutes
                volatility_cooloff[coin] = datetime.now()

                if (LastPrice+TRADING_FEE) >= (BuyPrice+TRADING_FEE):
                  trade_wins += 1
                else:
                  trade_losses += 1

                # Log trade
                if LOG_TRADES:
                    # adjust for trading fee here
                    profit = ((LastPrice - BuyPrice) * coins_sold[coin]['volume'])* (1-(TRADING_FEE*2))
                    write_log(f"Sell: {coins_sold[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} Profit: {profit:.2f} {PriceChange-(TRADING_FEE*2):.2f}%")
                    session_profit=session_profit + (PriceChange-(TRADING_FEE*2))
            continue

        # no action; print once every TIME_DIFFERENCE
        if hsp_head == 1:
            if len(coins_bought) > 0:
                print(f"TP or SL not yet reached, not selling {coin} for now {BuyPrice} - {LastPrice} : {txcolors.SELL_PROFIT if PriceChange >= 0.0 else txcolors.SELL_LOSS}{PriceChange-(TRADING_FEE*2):.2f}% Est:${(QUANTITY*(PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}, SL {coins_bought[coin]['stop_loss']:.2f}%")

                coins_bought[coin]['take_profit'] = TAKE_PROFIT
                if coins_bought[coin]['stop_loss'] < 0: coins_bought[coin]['stop_loss'] = STOP_LOSS

    if hsp_head == 1 and len(coins_bought) == 0: print(f'Not holding any coins')
    #print(f'')
    return coins_sold


def update_portfolio(orders, last_price, volume):
    '''add every coin bought to our portfolio for tracking/selling later'''
    if DEBUG: print(orders)
    for coin in orders:

        coins_bought[coin] = {
            'symbol': orders[coin][0]['symbol'],
            'orderid': orders[coin][0]['orderId'],
            'timestamp': orders[coin][0]['time'],
            'bought_at': last_price[coin]['price'],
            'volume': volume[coin],
            'take_profit': TAKE_PROFIT,
            'stop_loss': STOP_LOSS,
            }

        # save the coins in a json file in the same directory
        with open(coins_bought_file_path, 'w') as file:
            json.dump(coins_bought, file, indent=4)

        print(f'Order with id {orders[coin][0]["orderId"]} placed and saved to file')


def remove_from_portfolio(coins_sold):
    '''Remove coins sold due to SL or TP from portfolio'''
    for coin in coins_sold:
        coins_bought.pop(coin)

    with open(coins_bought_file_path, 'w') as file:
        json.dump(coins_bought, file, indent=4)


def write_log(logline):
    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
    with open(LOG_FILE,'a+') as f:
        f.write(timestamp + ' ' + logline + '\n')

def remove_external_signals(fileext):
    signals = glob.glob('signals/*.{fileext}')
    for filename in signals:
        for line in open(filename):
            try:
                os.remove(filename)
            except:

                if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file {filename}{txcolors.DEFAULT}')

def load_settings():

    # Load arguments then parse settings
    #mymodule = {}

    # set to false at Start
    global bot_paused
    bot_paused = False

    DEFAULT_CONFIG_FILE = 'config.yml'
    config_file = args.config if args.config else DEFAULT_CONFIG_FILE
    parsed_config = load_config(config_file)

    # Default no debugging
    global DEBUG, TEST_MODE, LOG_TRADES, LOG_FILE, DEBUG_SETTING, AMERICAN_USER, PAIR_WITH, QUANTITY, MAX_COINS, FIATS, TIME_DIFFERENCE, RECHECK_INTERVAL, CHANGE_IN_PRICE, STOP_LOSS, TAKE_PROFIT, CUSTOM_LIST, TICKERS_LIST, USE_TRAILING_STOP_LOSS, TRAILING_STOP_LOSS, TRAILING_TAKE_PROFIT, TRADING_FEE, SIGNALLING_MODULES
    DEBUG = False

    # Load system vars
    TEST_MODE = parsed_config['script_options']['TEST_MODE']
    LOG_TRADES = parsed_config['script_options'].get('LOG_TRADES')
    LOG_FILE = parsed_config['script_options'].get('LOG_FILE')
    DEBUG_SETTING = parsed_config['script_options'].get('DEBUG')
    AMERICAN_USER = parsed_config['script_options'].get('AMERICAN_USER')

    # Load trading vars
    PAIR_WITH = parsed_config['trading_options']['PAIR_WITH']
    QUANTITY = parsed_config['trading_options']['QUANTITY']
    MAX_COINS = parsed_config['trading_options']['MAX_COINS']
    FIATS = parsed_config['trading_options']['FIATS']
    TIME_DIFFERENCE = parsed_config['trading_options']['TIME_DIFFERENCE']
    RECHECK_INTERVAL = parsed_config['trading_options']['RECHECK_INTERVAL']
    CHANGE_IN_PRICE = parsed_config['trading_options']['CHANGE_IN_PRICE']
    STOP_LOSS = parsed_config['trading_options']['STOP_LOSS']
    TAKE_PROFIT = parsed_config['trading_options']['TAKE_PROFIT']
    CUSTOM_LIST = parsed_config['trading_options']['CUSTOM_LIST']
    TICKERS_LIST = parsed_config['trading_options']['TICKERS_LIST']
    USE_TRAILING_STOP_LOSS = parsed_config['trading_options']['USE_TRAILING_STOP_LOSS']
    TRAILING_STOP_LOSS = parsed_config['trading_options']['TRAILING_STOP_LOSS']
    TRAILING_TAKE_PROFIT = parsed_config['trading_options']['TRAILING_TAKE_PROFIT']
    TRADING_FEE = parsed_config['trading_options']['TRADING_FEE']
    SIGNALLING_MODULES = parsed_config['trading_options']['SIGNALLING_MODULES']

    if DEBUG_SETTING or args.debug:
        DEBUG = True


def load_profit(file):
    try:

        with open(file) as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError as fe:
        exit(f'Could not find {file}')

    except Exception as e:
        exit(f'Encountered exception...\n {e}')

if __name__ == '__main__':

    args = parse_args()
    DEFAULT_CREDS_FILE = 'creds.yml'
    DEFAULT_PROFIT_FILE = 'profit.yml'
    mainnet_wait = 10

    load_settings()

    creds_file = args.creds if args.creds else DEFAULT_CREDS_FILE

    # Load creds for correct environment
    parsed_creds = load_config(creds_file)
    access_key, secret_key = load_correct_creds(parsed_creds)

    if DEBUG:
        print(f'loaded config below\n{json.dumps(parsed_config, indent=4)}')
        print(f'Your credentials have been loaded from {creds_file}')


    # Authenticate with the client, Ensure API key is good before continuing
    if AMERICAN_USER:
        client = Client(access_key, secret_key, tld='us')
    else:
        client = Client(access_key, secret_key)

    # If the users has a bad / incorrect API key.
    # this will stop the script from starting, and display a helpful error.
    api_ready, msg = test_api_key(client, BinanceAPIException)
    if api_ready is not True:
       exit(f'{txcolors.SELL_LOSS}{msg}{txcolors.DEFAULT}')

    # Use CUSTOM_LIST symbols if CUSTOM_LIST is set to True
    if CUSTOM_LIST: tickers=[line.strip() for line in open(TICKERS_LIST)]

    # try to load all the coins bought by the bot if the file exists and is not empty
    coins_bought = {}

    global coins_bought_file_path, last_sell_datetime

    last_sell_datetime = "Never"

    # path to the saved coins_bought file
    coins_bought_file_path = 'coins_bought.json'

    # use separate files for testing and live trading
    if TEST_MODE:
        coins_bought_file_path = 'test_' + coins_bought_file_path

    # rolling window of prices; cyclical queue
    historical_prices = [None] * (TIME_DIFFERENCE * RECHECK_INTERVAL)
    hsp_head = -1

    # prevent including a coin in volatile_coins if it has already appeared there less than TIME_DIFFERENCE minutes ago
    volatility_cooloff = {}

    # if saved coins_bought json file exists and it's not empty then load it
    if os.path.isfile(coins_bought_file_path) and os.stat(coins_bought_file_path).st_size!= 0:
        with open(coins_bought_file_path) as file:
                coins_bought = json.load(file)

    print('Press Ctrl-C to stop the script')

    if not TEST_MODE:
        if not args.notimeout: # if notimeout skip this (fast for dev tests)
            print('WARNING: You are using the Mainnet and live funds. Waiting',mainnet_wait,'seconds as a security measure')
            time.sleep(mainnet_wait)
        else:
            print('You are using Test Mode')

    signals = glob.glob("signals/*.exs")
    for filename in signals:
        for line in open(filename):
            try:
                os.remove(filename)
            except:
                if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file {filename}{txcolors.DEFAULT}')

    if os.path.isfile("signals/paused.exc"):
        try:
            os.remove("signals/paused.exc")
        except:
            if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file {filename}{txcolors.DEFAULT}')

    remove_external_signals('buy')
    remove_external_signals('sell')
    remove_external_signals('pause')

    # load signalling modules
    signalthreads = []
    mymodule = {}
    try:
        if len(SIGNALLING_MODULES) > 0:
            for module in SIGNALLING_MODULES:
                print(f'Starting {module}')
                mymodule[module] = importlib.import_module(module)
                # t = threading.Thread(target=mymodule[module].do_work, args=())
                t = multiprocessing.Process(target=mymodule[module].do_work, args=())
                t.name = module
                t.daemon = True
                t.start()

                # add process to a list. This is so the thread can be terminated at a later time
                signalthreads.append(t)

                time.sleep(2)
        else:
            print(f'No modules to load {SIGNALLING_MODULES}')
    except Exception as e:
        print(f'Loading external signals exception: {e}')

    # seed initial prices
    get_price()
    READ_TIMEOUT_COUNT=0
    CONNECTION_ERROR_COUNT = 0
    while True:
        try:
            #print(f'bot_paused while try ',bot_paused)
            if bot_paused == False:
              orders, last_price, volume = buy()
              update_portfolio(orders, last_price, volume)
              coins_sold = sell_coins()
              remove_from_portfolio(coins_sold)
        except ReadTimeout as rt:
            READ_TIMEOUT_COUNT += 1
            print(f'{txcolors.WARNING}We got a timeout error from from binance. Going to re-loop. Current Count: {READ_TIMEOUT_COUNT}\n{rt}{txcolors.DEFAULT}')
        except ConnectionError as ce:
            CONNECTION_ERROR_COUNT +=1
            print(f'{txcolors.WARNING}We got a timeout error from from binance. Going to re-loop. Current Count: {CONNECTION_ERROR_COUNT}\n{ce}{txcolors.DEFAULT}')
        except KeyboardInterrupt as ki:
            # stop external signal threads
            stop_signal_threads()

            # ask user if they want to sell all coins
            print(f'\n\n\n')
            sellall = input(f'{txcolors.WARNING}Program execution ended by user!\n\nDo you want to sell all coins (y/N)?{txcolors.DEFAULT}')
            sellall = sellall.upper()
            if sellall == "Y":
                bot_paused = True
                # sell all coins
                #sell_all_coins('Program execution ended by user!')
                os.system('python sell-remaining-coins.py')
                coins_bought = {}
                print(f'Removing file and resetting session profit : ',coins_bought_file_path)
                if os.path.isfile(coins_bought_file_path):
                  os.remove(coins_bought_file_path)
                  coins_bought = {}
                  #with open(coins_bought_file_path) as file:
                    #coins_bought = json.load(file)


                print(f'Program execution ended by user and all held coins sold !')
                #print(f'Amount of held coins left : ',len(coins_bought))
                print(f'')

            sys.exit(0)
