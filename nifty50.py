import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download historical NIFTY 50 data using yfinance
nifty_data =yf.download('^NSEI', start='2010-01-01', end='2022-01-01')

# Extract relevant columns
nifty_data = nifty_data[['Adj Close']]

# Reset the index to make Date a column
nifty_data = nifty_data.reset_index()

# Feature engineering: Adding a column for days since the start
nifty_data['Days'] = (nifty_data['Date'] - nifty_data['Date'].min()).dt.days

# Split the data into training and testing sets
train, test = train_test_split(nifty_data, test_size=0.2, shuffle=False)

# Define the features (X) and target variable (y) for training
X_train = train[['Days']]
y_train = train['Adj Close']

# Define the features (X) and target variable (y) for testing
X_test = test[['Days']]
y_test = test['Adj Close']

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test['Date'], y_test, label='Actual')
plt.plot(test['Date'], y_pred, label='Predicted', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.title('NIFTY 50 Time Series Forecasting - Linear Regression')
plt.legend()
plt.show()
