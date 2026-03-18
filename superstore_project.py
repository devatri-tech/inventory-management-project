import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("superstore.csv")

print("\n DATA:\n")
print(df.head())

# Convert date
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Sales by category
print("\n SALES BY CATEGORY:\n")
print(df.groupby("Category")["Sales"].sum())

# Top products
print("\n TOP PRODUCTS:\n")
print(df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False))

# Profit by region
print("\n PROFIT BY REGION:\n")
print(df.groupby("Region")["Profit"].sum())

# Sales trend
sales_trend = df.groupby("Order Date")["Sales"].sum()

plt.figure()
sales_trend.plot()
plt.title("Sales Trend")
plt.show()

# Forecast
model = ARIMA(sales_trend, order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=5)

print("\n FORECAST:\n")
print(forecast)

plt.figure()
sales_trend.plot(label="Actual")
forecast.plot(label="Forecast", linestyle="dashed")
plt.legend()
plt.show()