import yfinance as yf
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# Download underlying security historical data
ticker = "SIE.DE"
start_date = '2023-01-01'
end_date = '2024-01-01'
data = yf.download(ticker, start=start_date, end=end_date, interval='1d')


# Parameters for strategy
initial_investment = 10000  # Amount of money to start with
number_of_weeks = 4  # Trading interval (1 = weekly, 2 = bi-monthly, etc)
T = (number_of_weeks * 7)/365   # Days till expiration
sigma = 0.25  # Assumed constant implied volatility
r = 0.01  # Assumed constant risk-free rate

# Lists to hold portfolio values
covered_call_results = [initial_investment]    # covered call strategy
protective_put_results = [initial_investment]   # protective put strategy
bull_call_spread_results = [initial_investment]     # bull call spread results
bah_results = [initial_investment]    # Buy and hold


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


# Buy and hold strategy
def buy_and_hold():
    shares_bought = initial_investment // data['Close'][find_friday()]
    current_cash = initial_investment - shares_bought * data['Close'][find_friday()]

    for i in range(find_friday(), len(data), 1):
        portfolio_value = current_cash + shares_bought * data["Close"][i]
        bah_results.append(portfolio_value)


# Covered call strategy
def covered_call(moneyness):    # (moneyness = 1 -> ATM, moneyness = 1.05 -> OTM, moneyness = 0.95 -> ITM)

    # INITIATION OF STRATEGY
    # Buy as many shares as possible in the beginning
    S = data['Close'][0]
    shares_bought = initial_investment // S

    # Sell an option on the first friday and collect premium
    K = moneyness * data["Close"][find_friday()]
    option_price = black_scholes(S, K, T, r, sigma, option_type='call')
    current_cash = initial_investment - (shares_bought * S) + option_price

    # Append current portfolio value to results
    covered_call_results.append(current_cash + (shares_bought * S))

    # Interval of trading days (5/week) between iteration
    d = number_of_weeks * 5

    for i in range(find_friday() + 1, len(data), 1):
        # Check new underlying price
        S = data['Close'][i]

        # If an option expires on the day
        if (i - find_friday()) % d == 0:
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
            K = moneyness * S
            option_price = black_scholes(S, K, T, r, sigma, option_type='call')
            current_cash += option_price

        # Append current portfolio value to results
        portfolio_value = current_cash + (shares_bought * data['Close'][i])
        covered_call_results.append(portfolio_value)

    print_results(covered_call_results)


# Protective Put Strategy
def protective_put(moneyness):  # (moneyness = 1 -> ATM, moneyness = 1.05 -> OTM, moneyness = 0.95 -> ITM)

    # INITIATION OF STRATEGY
    # Buy as many shares as possible in the beginning
    S = data['Close'][0]
    shares_bought = initial_investment // S

    # Buy a put option on the first friday to protect against downside risk
    K = moneyness * data["Close"][find_friday()]
    option_price = black_scholes(S, K, T, r, sigma, option_type='put')
    current_cash = initial_investment - (shares_bought * S) - option_price  # Subtract option premium from cash

    # Append current portfolio value to results
    portfolio_value = current_cash + (shares_bought * S) + option_price  # Adding the value of the put option
    covered_call_results.append(portfolio_value)

    # Interval of trading days (5/week) between iteration
    d = number_of_weeks * 5

    for i in range(find_friday() + 1, len(data), 1):
        # Check new underlying price
        S = data['Close'][i]
        if current_cash <= 0:
            break

        # If an option expires on the day
        if (i - find_friday()) % d == 0:
            # Check if the put option is exercised or expired
            if S < K:
                # Option is in the money; sell stock at strike price
                current_cash += shares_bought * K
                shares_bought = 0  # All stocks sold due to put execution
            else:
                # Option expired worthless, no stock is sold
                pass

            # Re-buy shares and buy a new put option for the next period
            shares_bought = initial_investment // S
            K = moneyness * S  # New strike price for the next put option
            option_price = black_scholes(S, K, T, r, sigma, option_type='put')
            current_cash -= shares_bought * S + option_price  # Subtract stock and option cost

        # Append current portfolio value to results
        portfolio_value = current_cash + (shares_bought * S) + option_price
        protective_put_results.append(portfolio_value)

    print_results(protective_put_results)




# Prints weekly returns of the strategy and of a simple buy and hold strategy.
def print_results(result):
    weekly_returns = []
    # Calculate weekly returns for covered call strategy
    for i in range(1, len(result)):
        weekly_return = ((result[i] - result[i - 1]) / result[i - 1])
        weekly_returns.append(weekly_return)
        bah_weekly_return = ((bah_results[i] - bah_results[i - 1]) / bah_results[i - 1])

    # Convert to a NumPy array for easier calculations
    weekly_returns = np.array(weekly_returns)

    # Calculate average weekly return
    average_weekly_return = np.mean(weekly_returns)

    # Calculate standard deviation of weekly returns
    std_deviation = np.std(weekly_returns)

    # Find the maximum weekly return
    max_return = np.max(weekly_returns)

    # Find the minimum weekly return
    min_return = np.min(weekly_returns)

    # Annualize the return (optional)
    annualized_return = average_weekly_return * 52


    # Print the covered_call_results
    print(f"Average Daily Return: {average_weekly_return:.4f}")
    print(f"Standard Deviation of Daily Returns: {std_deviation:.4f}")
    print(f"Maximum Daily Return: {max_return:.4f}")
    print(f"Minimum Daily Return: {min_return:.4f}")
    print(f"Annualized Return: {annualized_return:.4f}")


# Plot the results of different strategies
def plot_results():
    plt.plot(covered_call_results, label='Covered Call Strategy')
    plt.plot(protective_put_results, label="Protective Put Strategy")
    plt.plot(bull_call_spread_results, label="Bull call spread Strategy")
    plt.plot(bah_results, label='Buy and Hold Strategy', linestyle='--')
    plt.title('Strategy backtesting')
    plt.xlabel('Trading days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


buy_and_hold()
covered_call(moneyness=1)
bull_call_spread(K1=5, K2=10)
plot_results()






