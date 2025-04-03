import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Portfolio:
    def __init__(self, initial_cash= 100000):
        self.cash = initial_cash
        self.positions = {}
        self.trade_history = []
        self.realised_Pnl = 0
        self.unrealised_Pnl = 0
        self.portfolio_value = initial_cash
        
        # Tracking for plotting
        self.dates = []
        self.portfolio_values = []
        self.realised_pnls = []
        self.unrealised_pnls = []

    def buy(self, asset, quantity, price, date):
        # check if enough cash to buy
        if self.cash < price * quantity:
            print(f"Not enough cash to buy {quantity} of {asset} at {price}.")
        
        if asset in self.positions:
            # Get current position details
            current_qty, avg_price, _, _ = self.positions[asset]
            
            # Calculate new average price
            new_avg_price = (avg_price * current_qty + price * quantity) / (current_qty + quantity)
            
            # Update position
            self.positions[asset] = (
                current_qty + quantity,  # Total quantity
                new_avg_price,          # New average price
                price,                  # Current market price
                (price - new_avg_price) * (current_qty + quantity)  # Unrealized PnL
            )
            print(f"Bought {quantity} of {asset} at {price}. New avg price: {new_avg_price:.2f}")
        else:
            # Create new position
            self.positions[asset] = (
                quantity,    # Quantity
                price,       # Average price
                price,       # Current price
                0           # Unrealized PnL (0 for new position)
            )
            print(f"Bought {quantity} of {asset} at {price} (new position)")
        
        self.cash -= price * quantity
        self.trade_history.append(('buy', asset, quantity, price, date))
        self.update_portfolio_value(price, date)
        print(f"Available cash remaining: {self.cash:.2f}")

    def sell(self, asset, quantity, price, date):
        if asset in self.positions:
            current_quantity, avg_price, _, _ = self.positions[asset]
            
            if current_quantity >= quantity:
                # Calculate PnL before modifying position
                pnl = (price - avg_price) * quantity
                
                # Update position
                remaining_quantity = current_quantity - quantity
                if remaining_quantity > 0:
                    self.positions[asset] = (
                        remaining_quantity,
                        avg_price,  # Avg price stays the same
                        price,      # Current market price
                        (price - avg_price) * remaining_quantity  # Unrealized PnL
                    )
                else:
                    del self.positions[asset]  # Remove if position is closed
                
                # Update cash and PnL
                self.cash += price * quantity
                self.realised_Pnl += pnl
                
                # Record trade
                self.trade_history.append(('sell', asset, quantity, price, pnl, date))
                self.update_portfolio_value(price, date)
                
                print(f"Sold {quantity} of {asset} at {price}. Realized PnL: {pnl:.2f}")
                print(f"Available cash remaining: {self.cash:.2f}")
            else:
                print(f"Not enough {asset} to sell.")
        else:
            print(f"No position in {asset} to sell.")
        
    def update_portfolio_value(self, current_price, date):
        # Update all positions with current price and recalculate unrealized PnL
        total_unrealized = 0
        positions_value = 0
        
        for asset in list(self.positions.keys()):
            qty, avg_price, _, _ = self.positions[asset]
            unrealized = (current_price - avg_price) * qty
            self.positions[asset] = (qty, avg_price, current_price, unrealized)
            total_unrealized += unrealized
            positions_value += qty * current_price
        
        # Update portfolio metrics
        self.portfolio_value = self.cash + positions_value
        self.unrealised_Pnl = total_unrealized
        
        # Record the current state
        if hasattr(self, 'dates'):  # Only record if tracking attributes exist
            self.dates.append(date)
            self.portfolio_values.append(self.portfolio_value)
            self.realised_pnls.append(self.realised_Pnl)
            self.unrealised_pnls.append(self.unrealised_Pnl)

# test data
portfolio = Portfolio()

# example trades
portfolio.buy('BTC-USD', 1, 50000, datetime(2024, 1, 1))
portfolio.buy('ETH-USD', 2, 2000, datetime(2024, 1, 2))
portfolio.sell('BTC-USD', 0.5, 55000, datetime(2024, 1, 3))
portfolio.sell('ETH-USD', 1, 2500, datetime(2024, 1, 4))
portfolio.buy('BTC-USD', 0.2, 60000, datetime(2024, 1, 5))
portfolio.sell('ETH-USD', 1, 3000, datetime(2024, 1, 6))
portfolio.buy('BTC-USD', 0.3, 65000, datetime(2024, 1, 7))
# same day transaction
portfolio.sell('BTC-USD', 1, 70000, datetime(2024, 1, 7))

print("Portfolio Value History:")
for date, value in zip(portfolio.dates, portfolio.portfolio_values):
    print(f"{date}: ${value:.2f}")

print("\nTrade History:")
for trade in portfolio.trade_history:
    print(trade)

print("\nPnL:")
for trade in portfolio.trade_history:
    if trade[0] == 'sell':
        print(f"Sold {trade[1]}: PnL ${trade[4]:.2f} (Sold {trade[2]} at ${trade[3]:.2f})")
    else:
        print(f"Bought {trade[1]}: {trade[2]} at ${trade[3]:.2f}")

print("\nPortfolio Positions:")
for asset, (quantity, avg_price) in portfolio.positions.items():
    print(f"{asset}: {quantity} at avg price {avg_price}")
# print the cash
print("\nCash:")
print(f"${portfolio.cash:.2f}")

# plot the portfolio value history  
plt.figure(figsize=(10, 5))
plt.plot(portfolio.dates, portfolio.portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid()
plt.show()

def plot_btc_pnl(portfolio):
    # Extract BTC trades and PnL
    btc_trades = [trade for trade in portfolio.trade_history 
                 if trade[1] == 'BTC-USD' and trade[0] == 'sell']
    
    if not btc_trades:
        print("No BTC sell trades to plot")
        return
    
    dates = [trade[-1] for trade in btc_trades]  # Last element is date
    pnls = [trade[4] for trade in btc_trades]    # PnL is at index 4
    
    plt.figure(figsize=(10, 5))
    
    # Convert dates to proper datetime format for plotting
    date_objects = [pd.to_datetime(date) for date in dates]
    
    plt.plot(date_objects, pnls, marker='o', linestyle='-', color='blue')
    plt.title('Bitcoin Trading PnL')
    plt.xlabel('Date')
    plt.ylabel('PnL ($)')
    
    # Format x-axis dates properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()  # Rotate dates
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_account_pnl(portfolio):
    # Calculate cumulative PnL over time
    dates = []
    cum_pnl = []
    current_pnl = 0
    
    for trade in portfolio.trade_history:
        if trade[0] == 'sell':
            current_pnl += trade[4]  # Add PnL from sell trades
            dates.append(trade[-1])  # Get date
            cum_pnl.append(current_pnl)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dates, cum_pnl, marker='o', linestyle='-', color='green')
    plt.title('Account Cumulative Realized PnL')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_btc_pnl(portfolio)    # plots BTC trading PnL
plot_account_pnl(portfolio) # plots total account PnL