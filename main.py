# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:24:41 2025

@author: Shari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Data/Nat_Gas.csv')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

print("DataFrame shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataFrame info : ")
print(df.info())

print("\nBasic stats: ")
print(df.describe())

# Convert time to datetime
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values('Dates').reset_index(drop=True)

# Prepare data for regression
t = np.arange(len(df))
y = df['Prices'].values
slope, intercept = np.polyfit(t, y, 1)

# Seasonal Component from the last year
seasonal_pattern = y[-12:] - (intercept + slope * t[-12:])

# Forecast for the next 12 months
future_t = np.arange(len(df), len(df) + 12)
future_trend = intercept + slope * future_t

# Repeat the seasonal pattern
future_seasonality = seasonal_pattern

# Combine trend + seasonality
future_prices = future_trend + future_seasonality

# Generate future dates
last_date = df['Dates'].iloc[-1]
future_dates = pd.date_range(
    start=df['Dates'].iloc[-1] + pd.offsets.MonthEnd(1),
    periods=12,
    freq='M'
)

# Generate future DataFrame
future_df = pd.DataFrame({
    'Dates': future_dates,
    'Prices': future_prices
})



# --- Graphing ---



# Create a figure with a grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Bar Chart (top-left)
# Replace 'category_column' and 'value_column' with your actual column names
ax1 = axes[0, 0]
df.groupby('Dates')['Prices'].mean().sort_values().plot(
    kind='barh', ax=ax1, color='skyblue'
)
ax1.set_title('Average Values by Category')
ax1.set_xlabel('Average Value')

# 2. Line Chart (top-right)
# Replace 'date_column' and 'value_column' with your actual column names
ax2 = axes[0, 1]
df.groupby('Dates')['Prices'].sum().plot(
    kind='line', marker='o', ax=ax2, color='green'
)
ax2.set_title('Value Trend Over Time')
ax2.set_ylabel('Total Value')

# 3. Histogram (bottom-left)
# Replace 'numeric_column' with your actual column name
ax3 = axes[1, 0]
sns.histplot(df['Prices'].dropna(), kde=True, ax=ax3, color='purple')
ax3.set_title('Distribution of Values')
ax3.set_xlabel('Value')

# 4. Scatter Plot (bottom-right)
# Replace 'x_column' and 'y_column' with your actual column names
ax4 = axes[1, 1]
sns.scatterplot(
    data=df, 
    x='Dates', 
    y='Prices', 
    # hue='Dates',  # Color points by category
    # size='size_column',     # Vary point size (optional)
    alpha=0.7,
    ax=ax4
)




# --- Future ---



# Create a figure with a grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Bar Chart (top-left)
# Replace 'category_column' and 'value_column' with your actual column names
ax1 = axes[0, 0]
future_df.groupby('Dates')['Prices'].mean().sort_values().plot(
    kind='barh', ax=ax1, color='skyblue'
)
ax1.set_title('Average Values by Category')
ax1.set_xlabel('Average Value')

# 2. Line Chart (top-right)
# Replace 'date_column' and 'value_column' with your actual column names
ax2 = axes[0, 1]
future_df.groupby('Dates')['Prices'].sum().plot(
    kind='line', marker='o', ax=ax2, color='green'
)
ax2.set_title('Value Trend Over Time')
ax2.set_ylabel('Total Value')

# 3. Histogram (bottom-left)
# Replace 'numeric_column' with your actual column name
ax3 = axes[1, 0]
sns.histplot(future_df['Prices'].dropna(), kde=True, ax=ax3, color='purple')
ax3.set_title('Distribution of Values')
ax3.set_xlabel('Value')

# 4. Scatter Plot (bottom-right)
# Replace 'x_column' and 'y_column' with your actual column names
ax4 = axes[1, 1]
sns.scatterplot(
    data=future_df, 
    x='Dates', 
    y='Prices', 
    # hue='Dates',  # Color points by category
    # size='size_column',     # Vary point size (optional)
    alpha=0.7,
    ax=ax4
)

