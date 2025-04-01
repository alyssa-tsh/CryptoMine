#MODULE: PLOTTING FUNCTIONS FOR MODEL PERFORMANCE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import plotly.graph_objects as go

def visualize_split(train_data, val_data):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['timestamp'], train_data['close'], label='Train', color='blue')
    plt.plot(val_data['timestamp'], val_data['close'], label='Validation', color='red')
    plt.title("Train, Validation, Test Split")
    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    plt.show()

# Plot BTC price development candlestick chart
def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(x=data['timestamp'], open=data['open'], high=data['high'], low=data['low'], close=data['close'])])
    fig.update_layout(title='BTC/USDT Price Development', xaxis_title='Date', yaxis_title='Price')
    #theme minimal
    fig.update_layout(template='simple_white')
    fig.show()

def plot_actual_vs_predicted(train_data, val_data, test_data, train_preds, val_preds, test_preds, 
                           y_train, y_val, y_test, target_var):

    plt.figure(figsize=(20, 10))
    
    # Plot Training Set
    plt.plot(train_data['timestamp'], y_train[target_var], 
             label='Actual (Train)', color='blue', linewidth=2)
    plt.plot(train_data['timestamp'], train_preds, 'g-', 
             label='Predicted (Train)', linewidth=2)
    
    # Plot Validation Set
    plt.plot(val_data['timestamp'], y_val[target_var], 
             label='Actual (Validation)', color='orange', linewidth=2)
    plt.plot(val_data['timestamp'], val_preds, 'y-', 
             label='Predicted (Validation)', linewidth=2)
    
    # Plot Test Set
    plt.plot(test_data['timestamp'], y_test[target_var], 
             label='Actual (Test)', color='red', linewidth=2)
    plt.plot(test_data['timestamp'], test_preds, 'm-', 
             label='Predicted (Test)', linewidth=2)
    
    # Add vertical dividers between periods
    train_end_date = pd.to_datetime('2025-01-01') 
    val_end_date = pd.to_datetime('2025-02-28')
    last_train_date = train_end_date
    last_val_date = val_end_date
    
    plt.axvline(x=last_train_date, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=last_val_date, color='k', linestyle=':', alpha=0.5)
    
    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('Actual vs Predicted Prices', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()



# def residual_plot(val_pred, test_pred):
#     val_residuals = y_train[targets[0]] - val_pred
#     test_residuals = test_data[targets[0]] - test_pred
#     # Plot residuals for validation set
#     plt.figure(figsize=(12, 6))
#     plt.scatter(val_pred, val_residuals, alpha=0.5)
#     plt.axhline(y=0, color='red', linestyle='--')
#     plt.title("Residual Plot (Validation Set)")
#     plt.xlabel("Predicted Prices")
#     plt.ylabel("Residuals (Actual - Predicted)")
#     plt.show()

#     # Plot residuals for test set
#     plt.figure(figsize=(12, 6))
#     plt.scatter(test_pred, test_residuals, alpha=0.5)
#     plt.axhline(y=0, color='red', linestyle='--')
#     plt.title("Residual Plot (Test Set)")
#     plt.xlabel("Predicted Prices")
#     plt.ylabel("Residuals (Actual - Predicted)")
#     plt.show()
#residual_plot()

# def mse_plot(train_mse, test_mse)


# MODULE: EDA

#Distribution of Each Indicaor 
indicators = ['open', 'high', 'low', 'close', 'volume', 'SMA_5','SMA_10', 'SMA_15', 'EMA_9', 'RSI', 'OBV', 'MFI_10', 'ATR_14']  # Select key indicators
def plot_eda(data):
    # plt.figure(figsize=(12, 6))
    # for i, indicator in enumerate(indicators):
    #     plt.subplot(2, 4, i + 1)
    #     sns.histplot(data[indicator], kde=True, bins=30)
    #     plt.title(indicator)
    # plt.tight_layout()
    # plt.show()

    #Check for Correlations(Feature Redundancy)
    #Some indicators might be highly correlated, making them redundant
    #If correlation > 0.85, remove one of the correlated features.
    # Example: SMA and EMA are usually correlated → Keep only one.
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[indicators].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Technical Indicators")
    plt.show()

    # #Comparing Indicators with closing prices (Plot Trends Over Time)
    # # Check how SMA and EMA interact with price.
    # # Identify points where indicators signal trend reversals.

    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index, data['close'], label='Close Price', color='black')
    # plt.plot(data.index, data['SMA'], label='SMA', linestyle='dashed', color='blue')
    # plt.plot(data.index, data['EMA'], label='EMA', linestyle='dashed', color='red')
    # plt.legend()
    # plt.title("Price with SMA and EMA")
    # plt.show()

    # #Identify Overbought & Oversold Conditions (RSI & MACS)
    # # A. RSI (Relative Strength Index)
    # # RSI > 70 → Overbought (possible sell signal)
    # # RSI < 30 → Oversold (possible buy signal)
    # # Are most RSI values in a normal range (30-70)?
    # # Do RSI peaks align with price reversals?
    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index, data['RSI'], label="RSI", color="purple")
    # plt.axhline(70, linestyle="dashed", color="red", label="Overbought")
    # plt.axhline(30, linestyle="dashed", color="green", label="Oversold")
    # plt.legend()
    # plt.title("RSI Indicator")
    # plt.show()

    # # B. MACD (Moving Average Convergence Divergence)
    # # MACD > Signal Line → Bullish
    # # MACD < Signal Line → Bearish
    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    # plt.plot(data.index, data['MACD_signal'], label='Signal Line', color='red')
    # plt.axhline(0, linestyle="dashed", color="black")
    # plt.legend()
    # plt.title("MACD Indicator")
    # plt.show()

# Plot Validation Set
def plot_val(val_data, y_val, val_preds):
    plt.figure(figsize=(12,6))
    plt.plot(val_data['timestamp'], y_val['return_1'], 
                label='Actual (Validation)', color='orange', linewidth=2)
    plt.plot(val_data['timestamp'], val_preds, 'y--', 
                label='Predicted (Validation)', linewidth=2)
    plt.show()