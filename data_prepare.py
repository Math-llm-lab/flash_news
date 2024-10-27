import os
import csv
from datetime import datetime


class TokenDataFinder:
    def __init__(self, aave_lending_folder, binance_trading_folder):
        """
        Initialize the class with the folder paths for Aave lending and Binance trading data.

        :param aave_lending_folder: Path to the folder where Aave lending data is stored.
        :param binance_trading_folder: Path to the folder where Binance trading data is stored.
        """
        self.aave_lending_folder = aave_lending_folder
        self.binance_trading_folder = binance_trading_folder

    def get_data_range(self, file_path, is_binance=False):
        """
        Reads the first and last rows of a CSV file to get the timestamp range (data_from, data_to),
        and converts them to human-readable format.

        :param file_path: Path to the CSV file.
        :param is_binance: Boolean flag indicating whether the file is from Binance (true) or Aave lending (false).
        :return: Dictionary with 'data_from' and 'data_to' based on timestamps, in human-readable format.
        """
        data_range = {'data_from': None, 'data_to': None}

        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Skip headers

                # For the first row
                first_row = next(reader)
                if is_binance:
                    data_range['data_from'] = self.convert_timestamp(first_row[0], is_binance=True)
                else:
                    data_range['data_from'] = self.convert_timestamp(first_row[0], is_binance=False)

                # For the last row
                for last_row in reader:
                    pass
                if is_binance:
                    data_range['data_to'] = self.convert_timestamp(last_row[0], is_binance=True)
                else:
                    data_range['data_to'] = self.convert_timestamp(last_row[0], is_binance=False)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return data_range

    @staticmethod
    def convert_timestamp(timestamp, is_binance=False):
        """
        Converts timestamp to a human-readable format.

        :param timestamp: The timestamp to convert.
        :param is_binance: Boolean flag indicating whether the timestamp is from Binance (true) or Aave lending (false).
        :return: Formatted timestamp as 'YYYY-MM-DD HH:MM:SS'.
        """
        if is_binance:
            # Binance timestamp is in milliseconds
            timestamp = int(timestamp) // 1000
        else:
            timestamp = int(timestamp)  # Aave Lending timestamp is in seconds
        return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    def find_csv_files(self):
        """
        Find CSV files in the Aave lending folder and check if corresponding Binance trading data exists.
        This method constructs dictionaries with paths, data_from, and data_to information.

        :return: A dictionary with token data paths and additional metadata.
        """
        token_paths = {}

        # Search for CSV files in the Aave lending folder
        for root, dirs, files in os.walk(self.aave_lending_folder):
            for file in files:
                if file.endswith(".csv"):
                    # Extract the token name from the CSV file name (e.g., DAI from DAI.csv)
                    token_name = file.split(".")[0].upper()
                    # Extract 'lan' from the folder where the CSV is located
                    lan = root.split(os.sep)[-1]  # Use the last folder in the path as 'lan'
                    aave_path = os.path.join(root, file)
                    data_range = self.get_data_range(aave_path, is_binance=False)

                    # Add the token's Aave lending path and data range to the dictionary
                    token_paths[token_name] = {
                        'aave_lending': {
                            'path': aave_path,
                            'data_from': data_range['data_from'],
                            'data_to': data_range['data_to']
                        },
                        'lan': lan
                    }

        # Check if Binance trading data exists for each token in both SPOT and PERP directories
        for token_name in token_paths.keys():
            # Construct the path to Binance SPOT trading data for this token
            binance_spot_folder = os.path.join(self.binance_trading_folder, 'SPOT', f"{token_name}-USDT", '1MIN')
            binance_spot_file_path = os.path.join(binance_spot_folder, 'data.csv')
            if os.path.exists(binance_spot_file_path):
                data_range = self.get_data_range(binance_spot_file_path, is_binance=True)
                token_paths[token_name]['binance_trading_spot'] = {
                    'path': binance_spot_file_path,
                    'data_from': data_range['data_from'],
                    'data_to': data_range['data_to']
                }

            # Construct the path to Binance PERP trading data for this token
            binance_perp_folder = os.path.join(self.binance_trading_folder, 'PERP', f"{token_name}-USDT", '1MIN')
            binance_perp_file_path = os.path.join(binance_perp_folder, 'data.csv')
            if os.path.exists(binance_perp_file_path):
                data_range = self.get_data_range(binance_perp_file_path, is_binance=True)
                token_paths[token_name]['binance_trading_perp'] = {
                    'path': binance_perp_file_path,
                    'data_from': data_range['data_from'],
                    'data_to': data_range['data_to']
                }

        return token_paths


if __name__ == "__main__":

    # Define the folder paths for Aave Lending and Binance Trading data
    aave_lending_folder = 'aave_lending'
    binance_trading_folder = 'binance_trading'

    # Initialize the TokenDataFinder with the specified folder paths
    finder = TokenDataFinder(aave_lending_folder, binance_trading_folder)

    # Use the find_csv_files method to get paths and date ranges for each token
    token_paths = finder.find_csv_files()

    # Print the results for each token, including paths and date ranges
    for token, paths in token_paths.items():
        print(f"Token: {token}")

        # Print Aave Lending data path and date range
        print(
            f"  Aave Lending Path: {paths['aave_lending']['path']}\n Data From: {paths['aave_lending']['data_from']} Data To: {paths['aave_lending']['data_to']}"
        )

        # Check if Binance Spot Trading data is available and print path and date range
        if 'binance_trading_spot' in paths:
            print(
                f"  Binance Trading Spot Path: {paths['binance_trading_spot']['path']}\n Data From: {paths['binance_trading_spot']['data_from']} Data To: {paths['binance_trading_spot']['data_to']}"
            )
        else:
            print("  Binance Trading Spot Path: Not Found")

        # Print the 'lan' value, which indicates the token's folder for Aave lending data
        print(f"  Lan: {paths.get('lan')}\n")
