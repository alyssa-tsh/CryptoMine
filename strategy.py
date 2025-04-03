import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_log_returns(predicted_prices):
    """Compute log returns from predicted prices."""
    return np.log(predicted_prices[1:] / predicted_prices[:-1])

def exponential_weights(alpha, n=6):
    """Compute exponentially increasing weights."""
    return np.array([(1 - alpha) * (alpha  i) for i in range(n)])

def equal_weights(n=6, alpha=1):
    """Compute equal weights (average weighting)."""
    return np.full(n, alpha / n)

def weighted_return(returns, weights):
    """Compute weighted sum of returns."""
    return np.dot(weights, returns)

def mse_loss(alpha, returns, true_returns, weighting="exponential"):
    """Calculate MSE for a given alpha."""
    weights = exponential_weights(alpha, len(returns)) if weighting == "exponential" else equal_weights(len(returns), alpha)
    t_new = weighted_return(returns, weights)
    return np.mean((t_new - true_returns)  2)

def optimize_alpha_discrete(returns, true_returns, weighting="exponential"):
    """Find the best alpha by testing discrete values from 0.1 to 0.9."""
    alpha_values = np.arange(0.1, 1.0, 0.1)  # Alpha from 0.1 to 0.9 in steps of 0.1
    best_alpha = min(alpha_values, key=lambda a: mse_loss(a, returns, true_returns, weighting))
    return best_alpha

def generate_trading_signals(predicted_prices, true_returns, threshold=0.01, weighting="exponential"):
    """Generate buy/sell signals using discrete alpha values."""
    returns = compute_log_returns(predicted_prices)
    best_alpha = optimize_alpha_discrete(returns, true_returns, weighting)
    
    # Compute final weighted return
    weights = exponential_weights(best_alpha, len(returns)) if weighting == "exponential" else equal_weights(len(returns), best_alpha)
    final_return = weighted_return(returns, weights)
    
    # Generate trading signals
    signals = np.where(final_return > threshold, "BUY", "SELL")
    
    return best_alpha, final_return, signals

def calculate_logret_proxy(xgb_pred_n, close_t):
  """
  Calculates the Logret_proxy_t+n as defined in the formula.

  Args:
    xgb_pred_n: The XGBoost predicted value at time t+n.
    close_t: The closing price at time t.

  Returns:
    The calculated Logret_proxy_t+n.
  """

  if close_t == 0: 
    return np.nan 

  logret_proxy_t_plus_n = np.log(xgb_pred_n / close_t)
  return logret_proxy_t_plus_n

def calculate_trade_metrics(trade_log):
    """Calculates win rate, average win, and average loss from a trade log."""

    winning_trades = trade_log[trade_log['profit_loss'] > 0]
    losing_trades = trade_log[trade_log['profit_loss'] < 0]

    total_trades = len(trade_log)
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    if total_trades == 0:
        return 0, 0, 0 #Return 0 if there are no trades.

    win_rate = num_wins / total_trades if total_trades > 0 else 0
    avg_win = winning_trades['profit_loss'].mean() if num_wins > 0 else 0
    avg_loss = abs(losing_trades['profit_loss'].mean()) if num_losses > 0 else 0

    return win_rate, avg_win, avg_loss

def calculate_weighted_logret_proxy(logret_proxy_values, weights):
  """
  Calculates the weighted Logret_proxy as defined in the formula.

  Args:
    logret_proxy_values: A list or NumPy array of Logret_proxy values, 
                         corresponding to t+1, t+2, t+5, t+10, t+20, t+30.
    weights: A list or NumPy array of weights (w1, w2, w5, w10, w20, w30).

  Returns:
    The calculated weighted Logret_proxy value.
  """

  if len(logret_proxy_values) != len(weights):
    raise ValueError("The number of Logret_proxy values must match the number of weights.")

  weighted_logret_proxy = np.dot(logret_proxy_values, weights)
  return weighted_logret_proxy

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the Sharpe Ratio."""
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Calculates the Kelly Criterion fraction.

    Args:
        win_rate: The probability of winning a trade (0 to 1).
        avg_win: The average profit of winning trades.
        avg_loss: The average loss of losing trades (should be a positive number).

    Returns:
        The optimal fraction of portfolio to risk (f*), or None if avg_loss is zero.
    """

    if avg_loss == 0:
        return None  # Avoid division by zero

    R = avg_win / avg_loss  # Average win/loss ratio

    if R <= 0:
        return 0 # Avoid negative or zero R values.

    f_star = (win_rate - (1 - win_rate) / R)

    return f_star

def calculate_shares_kelly(portfolio_equity, win_rate, avg_win, avg_loss, current_price, max_risk_fraction = 0.2):
    """
    Calculates the number of shares to trade using the Kelly Criterion, with risk management.

    Args:
        portfolio_equity: The current value of your portfolio.
        win_rate: The probability of winning a trade (0 to 1).
        avg_win: The average profit of winning trades.
        avg_loss: The average loss of losing trades (should be a positive number).
        current_price: The current price of the asset.
        max_risk_fraction: Maximum fraction of portfolio to risk (to prevent overbetting).

    Returns:
        The number of shares to trade (integer), or 0 if Kelly Criterion is invalid.
    """
    kelly_fraction = kelly_criterion(win_rate, avg_win, avg_loss)

    if kelly_fraction is None or kelly_fraction < 0:
        return 0  # Invalid Kelly fraction

    # Apply risk management (limit the fraction)
    fraction_to_use = min(kelly_fraction, max_risk_fraction)

    # Calculate dollar amount to trade
    dollar_amount = portfolio_equity * fraction_to_use

    # Calculate number of shares
    num_shares = int(dollar_amount / current_price)

    return num_shares

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def simulate_trading_action(weighted_logret_proxy, holding_asset):
  """
  Simulates a single-day trading action based on the weighted Logret_proxy and holding status.

  Args:
    weighted_logret_proxy: The calculated weighted Logret_proxy value.
    holding_asset: A boolean indicating whether the algorithm currently holds the asset.

  Returns:
    A string representing the trading action: "BUY", "HOLD", "SELL", or "DO_NOTHING".
  """

  if weighted_logret_proxy > 0 and not holding_asset:
    return "BUY"
  elif weighted_logret_proxy > 0 and holding_asset:
    return "HOLD"
  elif weighted_logret_proxy < 0 and holding_asset:
    return "SELL"
  else:  # weighted_logret_proxy < 0 and not holding_asset
    return "DO_NOTHING"

# Example usage
predicted_prices = np.array([100, 102, 105, 107, 110, 115, 120])  # Example predicted prices
true_returns = np.array([0.02, 0.03, 0.025, 0.027, 0.03, 0.04])  # Example true returns

best_alpha, final_return, signals = generate_trading_signals(predicted_prices, true_returns, threshold=0.01, weighting="exponential")

print(f"Best Alpha: {best_alpha:.1f}")
print(f"Final Weighted Return: {final_return:.4f}")
print(f"Trading Signal: {signals}")