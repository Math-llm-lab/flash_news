# backtesting_program.py
import backtrader as bt
import datetime
import time, os, json
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
import math


class TradingStrategy(bt.Strategy):
    params = (
        ('sl_level', 1.5),#2.0
        ('tp_level', 1.5),
    )

    def __init__(self, _token, _backtesting_period, precision_qty=4, precision_price=2):
        # Для порівняння ціни на поч та кінець бектестингу
        self.start_price = None
        self.finish_price = None

        self.precision_qty = precision_qty
        self.precision_price = precision_price
        self.backtesting_period = _backtesting_period
        self.token = _token  # Store the token
        self.bollinger = bt.indicators.BollingerBands(self.datas[1], period=20, devfactor=2)
        self.rise_news = {}  # Dictionary to hold rise news dates
        self.fall_news = {}  # Dictionary to hold fall news dates
        self.size_to_buy = 0
        self.potential_stop_loss_price = None

        self.start_balance = self.broker.getvalue()  # Store initial portfolio value

        self.load_sentiment_data()  # Load sentiment data during initialization

    def load_sentiment_data(self):
        """
        Load sentiment data from pre-saved JSON files for the token, specifically rise and fall dates.
        """
        # Parse start and end dates from the backtesting period
        start_date = datetime.strptime(self.backtesting_period[0], '%Y-%m-%d')
        end_date = datetime.strptime(self.backtesting_period[1], '%Y-%m-%d')

        # Define file names based on the token and backtesting period
        file_name_rise = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_rise_dates_{self.token}.json"
        file_name_fall = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_fall_dates_{self.token}.json"
        token_dir = f"./{self.token}"

        # Define file paths
        file_path_rise = os.path.join(token_dir, file_name_rise)
        file_path_fall = os.path.join(token_dir, file_name_fall)

        # Load and round rise dates from JSON file
        try:
            with open(file_path_rise, 'r') as file:
                self.rise_news = [self.round_to_nearest_minute(date) for date in json.load(file)]
                print(f"Loaded and rounded rise dates from {file_path_rise}")
        except FileNotFoundError:
            print(f"[load_sentiment_data] Rise dates file not found: {file_path_rise}")
            self.rise_news = []

        # Load and round fall dates from JSON file
        try:
            with open(file_path_fall, 'r') as file:
                self.fall_news = [self.round_to_nearest_minute(date) for date in json.load(file)]
                print(f"Loaded and rounded fall dates from {file_path_fall}")
        except FileNotFoundError:
            print(f"[load_sentiment_data] Fall dates file not found: {file_path_fall}")
            self.fall_news = []

    def round_to_nearest_minute(self, timestamp):
        """
        Round the given timestamp string to the nearest minute.
        """
        dt = datetime.fromisoformat(timestamp[:-1])  # Remove the 'Z' and convert to datetime
        dt = dt.replace(second=0, microsecond=0)  # Set seconds and microseconds to 0
        if dt.second >= 30:  # If the seconds are 30 or more, round up to the next minute
            dt += timedelta(minutes=1)

        return dt.isoformat() + 'Z'  # Convert back to the desired string format

    def next(self):
        # Get current datetime of the data point
        current_datetime = self.data.datetime.datetime(0)
        current_datetime_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Define start and end dates for the backtesting period
        start_date = datetime.strptime(self.backtesting_period[0], '%Y-%m-%d')
        end_date = datetime.strptime(self.backtesting_period[1], '%Y-%m-%d')

        # If current datetime is outside the backtesting period, skip this data point
        if current_datetime < start_date or current_datetime > end_date:
            return

        # set start_price
        if self.start_price is None:
            self.start_price = self.data.open[0]

        if not self.position:
            pass

        # Execute trade if there's a rise signal
        if current_datetime_str in self.rise_news and not self.position:
            self.execute_trade()

        # Check if we need to close the position based on selling conditions
        if self.position:
            self.check_exit_conditions()

    def execute_trade(self):
        # Calculate the central line of Bollinger Bands using 1-hour data
        central_line = (self.bollinger.lines.top[0] + self.bollinger.lines.bot[0]) / 2

        # Calculate the current position relative to the Bollinger Bands
        price_relative_to_top = (self.data.close[0] - central_line) / (self.bollinger.lines.top[0] - central_line)
        price_relative_to_bot = (self.data.close[0] - central_line) / (self.bollinger.lines.bot[0] - central_line)

        if price_relative_to_top < 0:#-0.3:#0
            return

        # # Print out the relative positions
        # print(f"[TradingStrategy] Current price relative to top Bollinger Band: {price_relative_to_top:.2f}")
        # print(f"[TradingStrategy] Current price relative to bottom Bollinger Band: {price_relative_to_bot:.2f}")

        # Set the stop loss price
        self.potential_stop_loss_price = central_line - self.params.sl_level * abs(
            central_line - self.bollinger.lines.bot[0]
        )
        self.potential_stop_loss_price = round(self.potential_stop_loss_price, self.precision_price)

        # Calculate the amount of tokens we can buy with all available cash
        available_cash = self.broker.getcash()
        token_price = self.data.close[0]  # Current market price

        # Calculate how many tokens we can buy based on available cash and token price
        self.size_to_buy = available_cash / token_price
        self.size_to_buy = math.floor(self.size_to_buy * (10 ** self.precision_qty)) / (10 ** self.precision_qty)

        # Submit market order to buy
        self.buy(size=self.size_to_buy)

    def check_exit_conditions(self):
        # Check if we need to sell based on the Bollinger Bands or fall news
        current_price = self.data.close[0]
        upper_band = self.bollinger.lines.top[0]
        current_datetime = self.data.datetime.datetime(0)
        current_datetime_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Sell if price reaches upper Bollinger Band
        if current_price >= upper_band:
            self.close()

        # Sell if there's a fall news signal
        if current_datetime_str in self.fall_news:
            self.close()

        # Optionally, you could add logic to check for stop loss condition as well
        if current_price <= self.potential_stop_loss_price:
            self.close()

    def notify_order(self, order):
        """
        Notify when an order is executed and display relevant information.
        """
        if order.status in [order.Completed]:  # Only display if the order is fully completed
            if order.isbuy():
                print(
                    f"[TradingStrategy] Executed BUY order for {order.executed.size:.4f} at price {order.executed.price:.2f} "
                    f"on {self.data.datetime.datetime(0)}"
                )
            elif order.issell():
                print(
                    f"[TradingStrategy] Executed SELL order for {order.executed.size:.4f} at price {order.executed.price:.2f} "
                    f"on {self.data.datetime.datetime(0)}"
                )

                # Display balance information after order execution
                cash = self.broker.getcash()
                position_size = self.position.size  # Get current position size
                btc_value = position_size * self.data.close[0]  # Value of held BTC (or asset) in USD

                print(
                    f"[TradingStrategy] After order: Cash Balance: {round(cash, 2)} USD, Position: {position_size:.4f} "
                    f"tokens (Value: {round(btc_value, 2)} USD)\n"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # Notify if the order is canceled, has insufficient margin, or is rejected
            print(f"[TradingStrategy] Order {order.info['name']} was {order.getstatusname()}")

    def close_position(self):
        # Close the position at the current market price if a fall news signal is detected
        self.close(self.entry_order)

    def stop(self):
        # This method is called when the strategy is stopped
        print("[TradingStrategy] Strategy stopped")

        # Get the final price of the last candle
        self.finish_price = self.data.close[0]

        # current_datetime = self.data.datetime.datetime(0)
        # current_datetime_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Get the current value of the portfolio
        end_value = self.broker.getvalue()

        # Calculate final PnL in percentage
        pnl_percent = ((end_value - self.start_balance) / self.start_balance) * 100

        # Calculate the percentage change in price over the period
        price_change_percent = ((
                                            self.finish_price - self.start_price) / self.start_price) * 100 if self.start_price else 0

        # Print final results
        print(f"[TradingStrategy] Final portfolio value: {end_value:.2f}")
        print(f"[TradingStrategy] PnL: {pnl_percent:.2f}%")
        print(
            f"[TradingStrategy] Price change for the period: {price_change_percent:.2f}% (from {self.start_price} to {self.finish_price})")

    @staticmethod
    def add_analyzers(cerebro_):
        cerebro_.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeanalyzer")
        cerebro_.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro_.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=False)
        cerebro_.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn', timeframe=bt.TimeFrame.Days)


class BacktestingProgram:
    def __init__(self,
                 _backtesting_period=None, precision_qty=4, precision_price=2,
                 pair: str = None):
        self.start_price = None
        self.finish_price = None
        self.token = pair  # Store the token as a class variable
        # Dynamically construct the source folder path for each token
        self.source_folder = os.path.join('binance_trading', 'SPOT', f"{self.token}-USDT", '1MIN')

        self.backtesting_period = _backtesting_period
        self.precision_qty = precision_qty
        self.precision_price = precision_price
        self.results = None
        self.final_portfolio_value = None

        # Initialize daily stats and detailed stats
        self.daily_stats = pd.DataFrame()
        self.detailed_stats = {}

    def get_detailed_stats(self):
        return self.detailed_stats  # Return detailed statistics

    def get_daily_stats(self):
        return self.daily_stats  # Return daily statistics

    def get_results(self):
        return self.final_portfolio_value  # Return final portfolio value

    def run(self):
        cerebro = bt.Cerebro()

        # Load and preprocess CSV data
        df = pd.read_csv(os.path.join(self.source_folder, 'data.csv'))

        # Convert from milliseconds to seconds, then to datetime and finally to Backtrader format
        df['startTime'] = pd.to_datetime(df['startTime'] / 1000, unit='s')  # Convert timestamp to datetime
        df['startTime_bt'] = df['startTime'].apply(lambda x: bt.date2num(x))  # Convert to Backtrader's numeric format

        # Apply rounding to prices and volume
        df['open'] = df['open'].round(2)
        df['high'] = df['high'].round(2)
        df['low'] = df['low'].round(2)
        df['close'] = df['close'].round(2)
        df['volume'] = df['volume'].round(8)

        # Save the preprocessed data
        df.to_csv(os.path.join(self.source_folder, 'data_preprocessed.csv'), index=False)

        # Load into Backtrader with correct datetime format handling
        data = bt.feeds.GenericCSVData(
            dataname=os.path.join(self.source_folder, 'data_preprocessed.csv'),
            dtformat='%Y-%m-%d %H:%M:%S',  # Set to match the format in the CSV
            datetime=0,  # Column index for datetime (startTime)
            open=2,  # Column index for open price (skip endTime by shifting indices)
            high=3,  # Column index for high price
            low=4,  # Column index for low price
            close=5,  # Column index for close price
            volume=6,  # Column index for volume
            openinterest=-1,  # No open interest data
            timeframe=bt.TimeFrame.Minutes,  # Set to minutes
            compression=1,  # Compression level for 1-minute data
        )

        cerebro.adddata(data, name='1m')  # Add 1-minute data
        cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=60, name='1h')  # Resample to 1-hour data

        start_balance = 10000
        cerebro.broker.setcash(start_balance)

        cerebro.addstrategy(
            TradingStrategy,
            _backtesting_period=self.backtesting_period,
            precision_qty=self.precision_qty,
            precision_price=self.precision_price,
            _token=self.token  # Pass the token to the strategy
        )

        # Add analyzers
        TradingStrategy.add_analyzers(cerebro)

        self.results = cerebro.run()

        if self.results and isinstance(self.results[0], TradingStrategy):
            strategy_instance = self.results[0]

            # Access analyzers here
            if hasattr(strategy_instance.analyzers, 'tradeanalyzer'):
                self.detailed_stats['TradeAnalyzer'] = strategy_instance.analyzers.tradeanalyzer.get_analysis()
            if hasattr(strategy_instance.analyzers, 'drawdown'):
                self.detailed_stats['DrawDown'] = strategy_instance.analyzers.drawdown.get_analysis()
            if hasattr(strategy_instance.analyzers, 'sharpe'):
                sharpe = strategy_instance.analyzers.sharpe.get_analysis()
                self.detailed_stats['SharpeRatio'] = sharpe.get('sharperatio', None)
            if hasattr(strategy_instance.analyzers, 'timereturn'):
                timereturn = strategy_instance.analyzers.timereturn.get_analysis()
                self.daily_stats = pd.DataFrame(list(timereturn.items()), columns=['Date', 'Return'])
                self.daily_stats['Date'] = pd.to_datetime(self.daily_stats['Date'])
                self.daily_stats.set_index('Date', inplace=True)

        self.final_portfolio_value = cerebro.broker.getvalue()

        strategy_instance = self.results[0]
        self.start_price = strategy_instance.start_price
        self.finish_price = strategy_instance.finish_price

        self.final_portfolio_value = cerebro.broker.getvalue()  # Final portfolio value

    def get_daily_stats(self):
        """
        Retrieve daily statistics, if applicable.
        """
        # Placeholder for actual implementation
        return pd.DataFrame()  # Return an empty DataFrame for now

    def print_backtesting_results(self, file_name=None):
        """
        Prints the backtesting results and optionally saves them to a file.
        If a file_name is provided, results will also be saved to the file.
        """
        # Get final backtest results
        results = self.get_results()

        # Ensure results have been populated
        if not self.results:
            print("[BacktestingProgram] No results to process.")
            return

        strategy_instance = self.results[0]  # Access the first strategy
        position = strategy_instance.getposition(strategy_instance.data)  # Get current position size
        position_size = position.size  # Position size in tokens
        current_price = strategy_instance.data.close[0]  # Current market price
        btc_value = position_size * current_price if position_size else 0  # Value of held tokens in USD
        cash_value = results - btc_value  # Calculate cash value

        # Calculate and format price change
        price_change_percent = ((
                                            self.finish_price - self.start_price) / self.start_price) * 100 if self.start_price else 0

        output = f"\nBacktesting Results:\n"
        output += f"Final Portfolio Value: {round(results, self.precision_price)} (Cash: {round(cash_value, self.precision_price)}, Position: {position_size:.4f} tokens)\n"
        output += f"Price change for the period: {price_change_percent:.2f}% (from {self.start_price} to {self.finish_price})\n"

        # Get detailed statistics from analyzers
        self.detailed_stats = {}
        for strategy in self.results:
            if hasattr(strategy.analyzers, 'tradeanalyzer'):
                self.detailed_stats['TradeAnalyzer'] = strategy.analyzers.tradeanalyzer.get_analysis()
            if hasattr(strategy.analyzers, 'drawdown'):
                self.detailed_stats['DrawDown'] = strategy.analyzers.drawdown.get_analysis()
            if hasattr(strategy.analyzers, 'sharpe'):
                sharpe = strategy.analyzers.sharpe.get_analysis()
                self.detailed_stats['SharpeRatio'] = sharpe.get('sharperatio', None)

        # Print detailed statistics
        output += "\nDetailed Stats:\n"
        trade_analyzer = self.detailed_stats.get('TradeAnalyzer', {})
        if trade_analyzer:
            output += f"Total Trades: {trade_analyzer.get('total', {}).get('total', 0)}\n"
            output += f"Closed Trades: {trade_analyzer.get('total', {}).get('closed', 0)}\n"
            output += f"Gross PnL: {round(trade_analyzer.get('pnl', {}).get('gross', {}).get('total', 0), self.precision_price)}\n"
            output += f"Net PnL: {round(trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0), self.precision_price)}\n"
            output += f"Wins: {trade_analyzer.get('won', {}).get('total', 0)}\n"
            output += f"Losses: {trade_analyzer.get('lost', {}).get('total', 0)}\n"

        # Print Sharpe Ratio
        sharpe_ratio = self.detailed_stats.get('SharpeRatio', None)
        if sharpe_ratio is not None:
            output += f"Sharpe Ratio: {round(sharpe_ratio, 2)}\n"
        else:
            output += "Sharpe Ratio: Not Calculated\n"

        print(output)

        # Save to file if filename is provided
        if file_name:
            with open(file_name, 'a') as f:
                f.write(output)

if __name__ == "__main__":
    tokens = ["BTC", "ETH"]

    for token in tokens:
        backtesting_period = ['2024-09-27', '2024-10-06']

        # Initialize the backtesting program
        backtesting_program = BacktestingProgram(
            _backtesting_period=backtesting_period,
            precision_qty=4,
            precision_price=2,
            pair=token,  # Set the token here
        )

        backtesting_program.run()  # Run the backtesting
        backtesting_program.print_backtesting_results()  # Print the results
