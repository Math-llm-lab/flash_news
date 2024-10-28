# Flash News Trading Strategy

## Overview

The **Flash News Trading Strategy** is a Python-based project aimed at leveraging historical financial data, news sentiment analysis, and backtesting to create an effective cryptocurrency trading strategy. The goal is to interact with DeFi primitives and utilize Large Language Models (LLMs) for real-time decision-making in trading.

## Project Structure

The project directory contains the following files and folders:

<br>
├── AAVE                       # Folder for AAVE related data<br>
├── aave_lending               # Folder containing AAVE lending data<br>
├── AVAX                       # Folder for AVAX related data<br>
├── binance_trading            # Folder containing Binance trading data<br>
├── BTC                        # Folder for BTC related data<br>
├── ETH                        # Folder for ETH related data<br>
├── LINK                       # Folder for LINK related data<br>
├── SUSHI                      # Folder for SUSHI related data<br>
├── volatility                  # Folder for volatility data<br>
├── .gitignore                 # Git ignore file<br>
├── api_key.env                # Environment file for API keys (create this file with your API keys)<br>
├── backtesting_program.py      # Script for backtesting trading strategies<br>
├── data_prepare.py            # Script for preparing data<br>
├── flash_news.html            # HTML export of the Jupyter notebook<br>
├── flash_news.ipynb           # Jupyter notebook for the project<br>
├── get_news.py                # Script for collecting news data<br>
├── README.md                  # Project documentation<br>
└── requirements.txt           # List of project dependencies<br>

## Steps Involved

### Step 1: Environment Setup
Set up the environment and install the required dependencies using the `requirements.txt` file. 

### Step 2: Data Preparation
Utilize the `data_prepare.py` script to gather historical data from centralized exchanges (CEX) and decentralized finance (DeFi) platforms. The `TokenDataFinder` class automates the process of locating and extracting time ranges from specified CSV files.

### Step 3: News Analysis
The `get_news.py` module collects relevant news articles and analyzes sentiment related to specified cryptocurrency tokens. The sentiment data is then stored for backtesting.

### Step 4: Backtesting
The `backtesting_program.py` file contains the core logic for the trading strategy, including the trading rules based on news sentiment and Bollinger Bands. It utilizes the Backtrader library for simulating trades and evaluating strategy performance.

### Conclusion
The Flash News Trading Strategy demonstrates the potential for innovative real-time trading algorithms that can react to market dynamics. The findings from this project can lay the groundwork for the development of agents such as **Atlas**, **Ganesh**, or **Apollo** that enhance trading efficiency.

## License
This project is licensed under the MIT License.
