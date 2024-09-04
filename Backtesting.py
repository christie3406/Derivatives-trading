import yfinance as yf
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# Download underlying security historical data
ticker = "SIE.DE"
start_date = '2020-01-01'
end_date = '2024-01-01'
data = yf.download(ticker, start=start_date, end=end_date, interval='1d')


# Initial investment amount
initial_investment = 10000  # Amount of money to start with


# Parameters for strategy
T = 7 / 365  # Days to expiration / time interval for trading
sigma = 0.20  # Assumed constant implied volatility
r = 0.01  # Assumed constant risk-free rate


# Lists to hold portfolio values
covered_call_results = [initial_investment]    # Strategy covered_call_results
bah_results = [initial_investment]    # Buy and hold covered_call_results


# Options typically expire on fridays; the backtester uses weekly intervals that land on friday.
def find_friday():
    fridays = data[data.index.to_series().dt.dayofweek == 4]
    d = 0
    while data.index[d] != fridays.index[0]:
        d += 1
    return d


# Using black-scholes formula to estimate the price of european options, since yfinance does not provide historical option data
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    return option_price


# Prints weekly returns of the strategy and of a simple buy and hold strategy.
def print_results():
    weekly_returns = []
    bah_weekly_returns = []
    # Calculate weekly returns for covered call strategy
    for i in range(1, len(covered_call_results)):
        weekly_return = ((covered_call_results[i] - covered_call_results[i - 1]) / covered_call_results[i - 1])
        weekly_returns.append(weekly_return)
        bah_weekly_return = ((bah_results[i] - bah_results[i - 1]) / bah_results[i - 1])
        bah_weekly_returns.append(bah_weekly_return)

    # Convert to a NumPy array for easier calculations
    weekly_returns = np.array(weekly_returns)
    bah_weekly_returns = np.array(bah_weekly_returns)

    # Calculate average weekly return
    average_weekly_return = np.mean(weekly_returns)
    bah_average_weekly_return = np.mean(bah_weekly_returns)

    # Calculate standard deviation of weekly returns
    std_deviation = np.std(weekly_returns)

    # Find the maximum weekly return
    max_return = np.max(weekly_returns)

    # Find the minimum weekly return
    min_return = np.min(weekly_returns)

    # Annualize the return (optional)
    annualized_return = average_weekly_return * 52

    bah_annualized_return = bah_average_weekly_return * 52

    # Calculate comparison
    advantage = (annualized_return - bah_annualized_return) / annualized_return

    # Print the covered_call_results
    print(f"Average Daily Return: {average_weekly_return:.4f}")
    print(f"Standard Deviation of Daily Returns: {std_deviation:.4f}")
    print(f"Maximum Daily Return: {max_return:.4f}")
    print(f"Minimum Daily Return: {min_return:.4f}")
    print(f"Annualized Return: {annualized_return:.4f}")
    print(f"Advantage compared to buy and hold: {advantage:.4f}")
    print(f"Annualized Return buy and hold: {bah_annualized_return:.4f}")


# Plot the results of different strategies
def plot_results():
    plt.plot(covered_call_results, label='Covered Call Strategy')
    plt.plot(bah_results, label='Buy and Hold Strategy', linestyle='--')
    plt.title('Strategy backtesting')
    plt.xlabel('Weeks')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


# Buy and hold strategy
def buy_and_hold():
    shares_bought = initial_investment // data['Close'][find_friday()]
    current_cash = initial_investment - shares_bought * data['Close'][find_friday()]

    for i in range(find_friday(), len(data) - 5, 5):
        portfolio_value = current_cash + shares_bought * data["Close"][i]
        bah_results.append(portfolio_value)


# Covered call strategy
def covered_call(m):    # (m = 1 -> ATM, m = 1.05 -> OTM, m = 0.95 -> ITM)

    # INITIATION OF STRATEGY
    # Buy as many shares as possible at first friday
    S = data['Close'][find_friday()]
    shares_bought = initial_investment // S

    # Sell an option and collect premium
    K = m * S
    option_price = black_scholes(S, K, T, r, sigma, option_type='call')
    current_cash = initial_investment - (shares_bought * S) + option_price

    # Append current portfolio value to results
    covered_call_results.append(current_cash + (shares_bought * S))

    # REPEAT
    for i in range(find_friday() + 5, len(data) - 5, 5):
        # Check new underlying price
        S = data['Close'][i]

        # If underlying price is above strike, stock is "called away"
        if S > K:
            current_cash += 100 * K
            shares_bought -= 100  # You lose 100 shares of your underlying position
        else:
            # Stock not called away, keep your position
            pass

        # Re-buy underlying security if necessary
        if shares_bought < 100:
            shares_bought += current_cash // S
            current_cash -= shares_bought * S

        # Sell a new option
        K = m * S
        option_price = black_scholes(S, K, T, r, sigma, option_type='call')
        current_cash += option_price

        # Append current portfolio value to results
        portfolio_value = current_cash + (shares_bought * S)
        covered_call_results.append(portfolio_value)


buy_and_hold()
covered_call(1)
plot_results()
print_results()



