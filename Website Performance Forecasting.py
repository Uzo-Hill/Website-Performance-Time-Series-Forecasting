#!/usr/bin/env python
# coding: utf-8

# # Multi-Channel Website Performance Forecasting and Engagement Analysis

# ## Introduction

# This project applies time series analysis to website performance data, exploring hourly user traffic, sessions, engagement, and conversions across channels such as Direct, Organic Search, Organic Social, and Referrals. The dataset was obtained from the [Statso Website Performance Case Study](https://statso.io/website-performance-case-study/).
# 
# By uncovering patterns, forecasting trends, and detecting anomalies with techniques such as ARIMA and seasonal decomposition, the analysis provides actionable insights to optimize user experience, channel effectiveness, and strategic decision-making.
# 
# 

# ## Primary Objective:

# - Build and validate time-series forecasting models to predict sessions
# 
# - Analyze engagement patterns and trends over time
# 
# - Identify seasonal patterns and peak performance hours
# 
# - Compare performance across different marketing channels

# In[24]:


import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path[0])


# In[ ]:





# In[ ]:





# In[1]:


# Test all imports
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import scipy
import statsmodels

print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")  
print(f"Statsmodels: {statsmodels.__version__}")

# Test your original problematic imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("✅ All imports successful!")


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ## Data Loading and Initial Inspection
# 

# In[3]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\DATASETS\Time series Dataset\website Performance dataset.csv")


# In[4]:


df.head()


# In[ ]:





# In[5]:


# Display basic information about the dataset

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())


# In[ ]:





# ## Data Cleaning and Transformation

# In[6]:


# Rename column:

df = df.rename(columns={"Session primary channel group (Default channel group)": "Session primary channel group"})


df = df.rename(columns={"Date + hour (YYYYMMDDHH)": "Datetime"})


# In[7]:


df.head()


# In[ ]:





# In[8]:


# Convert the date column to proper datetime format

df['Datetime'] = df['Datetime'].astype(str)
df['datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H')

# Set datetime as index
df.set_index('datetime', inplace=True)


# In[9]:


df.head(2)


# In[10]:


# Check for duplicates

print("Duplicate rows:", df.duplicated().sum())


# In[ ]:





# In[ ]:





# In[33]:


# Check for outliers

# Select numerical columns
num_cols = [
    "Users", "Sessions", "Engaged sessions", 
    "Average engagement time per session", 
    "Engaged sessions per user", "Events per session", 
    "Engagement rate", "Event count"
]

# Set up the figure grid
fig, axes = plt.subplots(len(num_cols), 2, figsize=(12, 4*len(num_cols)))

for i, col in enumerate(num_cols):
    # Distribution (histogram + KDE)
    sns.histplot(data=df, x=col, kde=True, ax=axes[i,0], color="skyblue")
    axes[i,0].set_title(f"Distribution of {col}")
    axes[i,0].set_xlabel("")
    axes[i,0].set_ylabel("Frequency")

    # Boxplot
    sns.boxplot(data=df, x=col, ax=axes[i,1], color="lightcoral")
    axes[i,1].set_title(f"Boxplot of {col}")
    axes[i,1].set_xlabel("")

plt.tight_layout()
plt.show()


# In[ ]:





# In[12]:


# Create additional time-based features

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month
df['week_of_year'] = df.index.isocalendar().week


# In[ ]:





# In[13]:


# Display cleaned data info
print("\nCleaned Dataset Info:")
print(df.info())
print()
print("\nSample of cleaned data:")

df.head()


# In[ ]:





# ## Exploratory Data Analysis

# In[14]:


# Summary statistics

print("Summary Statistics:")
print(df.describe())


# In[ ]:





# In[15]:


# Plot distribution of numerical variables
numerical_cols = ['Users', 'Sessions', 'Engaged sessions', 'Average engagement time per session', 
                 'Engaged sessions per user', 'Events per session', 'Engagement rate', 'Event count']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# - Most activity-based metrics (Users, Sessions, Engaged Sessions, Event Count, Engagement Time, and Events per Session) are right-skewed, meaning most values are low with a few very high outliers. 
#     
# - Engaged Sessions per User is roughly normal, peaking around 0.6–0.7, showing consistent user contribution to engagement.
#     
# - Engagement Rate centers around 0.5–0.6 in a bell-shaped pattern, indicating moderate typical engagement.

# In[16]:


# Channel analysis
plt.figure(figsize=(12, 6))
channel_counts = df['Session primary channel group'].value_counts()
plt.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Marketing Channels')
plt.show()


# In[ ]:





# In[26]:


# Visualization of time seres key metrics:


import matplotlib.pyplot as plt

# Ensure datetime index is in datetime format
df.index = pd.to_datetime(df.index)

plt.figure(figsize=(14, 8))

# Plot Users over time
plt.subplot(3, 1, 1)
plt.plot(df.index, df["Users"], marker="o", linestyle="-", color="tab:blue")
plt.title("Users Over Time")
plt.xlabel("Date")
plt.ylabel("Users")
plt.grid(True)

# Plot Sessions over time
plt.subplot(3, 1, 2)
plt.plot(df.index, df["Sessions"], marker="o", linestyle="-", color="tab:green")
plt.title("Sessions Over Time")
plt.xlabel("Date")
plt.ylabel("Sessions")
plt.grid(True)

# Plot Engagement Rate over time
plt.subplot(3, 1, 3)
plt.plot(df.index, df["Engagement rate"], marker="o", linestyle="-", color="tab:red")
plt.title("Engagement Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Engagement Rate")
plt.grid(True)

plt.tight_layout()
plt.show()


# 
# – User activity fluctuates across the period, with noticeable spikes around mid-April, suggesting traffic surges possibly due to campaigns or external events.
#     
# – Sessions follow a similar trend to users, indicating that session counts are strongly tied to user visits, with peaks aligning with user spikes.
#     
# – Engagement rate is highly variable, with values spread between 0.2 and 0.8, showing inconsistent user engagement patterns despite relatively stable user/session trends.
# 
# 

# In[ ]:





# In[ ]:





# ## Time Series Decomposition

# In[21]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[30]:


# Time Series Decomposition
def clean_decomposition(series, title, period=7):
    """Clean decomposition with error handling"""
    try:
        result = seasonal_decompose(series.dropna(), model='additive', period=min(period, len(series)//2))
        fig = result.plot()
        fig.suptitle(f'Decomposition: {title}', y=1.02, fontweight='bold')
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"⚠️  Could not decompose {title}: {str(e)[:100]}...")
        return False

# Get numerical data
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
daily_df = df[numerical_cols].resample('D').mean()

print(f"📊 Daily data: {daily_df.shape[0]} days available")

# Decompose key metrics
metrics_to_decompose = ['Users', 'Engagement rate']
successful_decompositions = 0

for metric in metrics_to_decompose:
    if metric in daily_df.columns:
        print(f"\n🔍 Analyzing {metric}...")
        if clean_decomposition(daily_df[metric], f'Daily {metric}'):
            successful_decompositions += 1

# Fallback to hourly if daily decomposition fails
if successful_decompositions == 0:
    print("\n🔄 Trying hourly decomposition as fallback...")
    for metric in metrics_to_decompose:
        if metric in df.columns:
            clean_decomposition(df[metric], f'Hourly {metric}', period=24)

# Quick summary
print(f"\n✅ Successful decompositions: {successful_decompositions}/{len(metrics_to_decompose)}")


# - The data shows a stable underlying trend around 40-45 daily users with clear weekly seasonal patterns indicating consistent day-of-week effects on user behavior.
# 
# - The small, random residuals around zero suggest the decomposition effectively captures the main patterns, making this suitable for reliable forecasting.
#     
# - The engagement rate shows a notable declining trend from ~0.51 to ~0.498 in mid-April, followed by a recovery back toward 0.51 by early May.
#     

# In[ ]:





# ## Correlation Analysis

# In[21]:


# Correlation matrix 

correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# - Users, Sessions, and Engaged sessions show very high correlations (0.95-0.99), indicating these core traffic metrics move together, while engagement quality metrics like engagement rate and engaged sessions per user also correlate strongly (0.96).

# In[22]:


# Lag analysis for autocorrelation
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(12, 6))
autocorrelation_plot(daily_df['Users'].dropna())
plt.title('Autocorrelation of Daily Users')
plt.show()


# In[ ]:





# ## Channel-wise Analysis

# In[23]:


# Compare metrics across channels 
channel_col = df.columns[0]  # Assuming first column is the channel
channel_metrics = df.groupby(channel_col)[numerical_cols].mean().round(2)

print("Average Metrics by Channel:")
print(channel_metrics)

# Plot channel performance - select only numerical columns that exist
plot_columns = [col for col in ['Users', 'Sessions', 'Engagement rate', 'Event count'] if col in numerical_cols]

if len(plot_columns) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(plot_columns):
        if i < 4:  # Ensure we don't exceed subplot count
            channel_metrics[col].plot(kind='bar', ax=axes[i], title=f'Average {col} by Channel')
            plt.sca(axes[i])
            plt.xticks(rotation=45)

    # Hide empty subplots if we have less than 4 columns
    for i in range(len(plot_columns), 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
else:
    print("No numerical columns available for channel analysis")


# - Organic Social is the clear leader across all metrics, driving ~70 users and sessions with the highest engagement rate (~0.75) and event count (~450).
#     
# - While Direct traffic provides consistent secondary performance with moderate engagement levels.
# 
# - Despite lower traffic volumes, Organic Video shows strong engagement rates (~0.65)

# In[ ]:





# ## Time-based Pattern Analysis

# In[24]:


# Hourly patterns
hourly_patterns = df.groupby('hour')[numerical_cols].mean()

# Plot only if we have numerical columns
if len(numerical_cols) > 0:
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(numerical_cols[:3], 1):  # Plot first 3 numerical columns
        plt.subplot(2, 2, i)
        hourly_patterns[col].plot()
        plt.title(f'Hourly Pattern of {col}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Value')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[25]:


# Weekly patterns
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_patterns = df.groupby('day_of_week')[numerical_cols].mean()
weekly_patterns.index = weekday_names

if len(numerical_cols) > 0:
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(numerical_cols[:3], 1):  # Plot first 3 numerical columns
        plt.subplot(2, 2, i)
        weekly_patterns[col].plot()
        plt.title(f'Weekly Pattern of {col}')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Value')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:





# ## Stationarity Check

# In[26]:


from statsmodels.tsa.stattools import adfuller

# Check stationarity for key metrics
def check_stationarity(series, title):
    result = adfuller(series.dropna())
    print(f'ADF Statistic for {title}: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    print('\n')

check_stationarity(daily_df['Users'], 'Daily Users')
check_stationarity(daily_df['Engagement rate'], 'Daily Engagement Rate')


# Stationarity Analysis Results
# Daily Users:
# 
# ADF Statistic: -3.856 (more negative than all critical values)
# 
# p-value: 0.0024 (< 0.05)
# 
# Conclusion: Stationary (reject null hypothesis of non-stationarity)
# 
# Daily Engagement Rate:
# 
# ADF Statistic: -4.589 (more negative than all critical values)
# 
# p-value: 0.00014 (< 0.05)
# 
# Conclusion: Stationary (reject null hypothesis of non-stationarity)
# 
# Since both series are stationary, we don't need differencing for our ARIMA models. 

# In[ ]:





# ## Forecasting Preparation

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prepare data for forecasting
forecast_df = daily_df[['Users', 'Sessions', 'Engagement rate']].copy()

# Create lag features for time series forecasting
for lag in range(1, 8):  # 7 days of lag
    forecast_df[f'Users_lag_{lag}'] = forecast_df['Users'].shift(lag)
    forecast_df[f'Engagement_lag_{lag}'] = forecast_df['Engagement rate'].shift(lag)

# Drop rows with NaN values after creating lags
forecast_df = forecast_df.dropna()

# Split into train and test sets
train_size = int(len(forecast_df) * 0.8)
train, test = forecast_df.iloc[:train_size], forecast_df.iloc[train_size:]

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train period: {train.index.min()} to {train.index.max()}")
print(f"Test period: {test.index.min()} to {test.index.max()}")


# In[ ]:





# ## Time Series Forecasting with ACF/PACF

# In[28]:


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[30]:


# Calculate maximum allowed lags for our small dataset
max_lags = min(10, len(daily_df) // 2 - 1)  # Ensure lags < 50% of sample size
print(f"Maximum allowed lags for ACF/PACF: {max_lags}")

# Plot ACF and PACF with appropriate lags for small dataset
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

plot_acf(daily_df['Users'].dropna(), ax=axes[0, 0], lags=max_lags)
axes[0, 0].set_title(f'ACF of Daily Users (lags={max_lags})')

plot_pacf(daily_df['Users'].dropna(), ax=axes[0, 1], lags=max_lags)
axes[0, 1].set_title(f'PACF of Daily Users (lags={max_lags})')

plot_acf(daily_df['Engagement rate'].dropna(), ax=axes[1, 0], lags=max_lags)
axes[1, 0].set_title(f'ACF of Daily Engagement Rate (lags={max_lags})')

plot_pacf(daily_df['Engagement rate'].dropna(), ax=axes[1, 1], lags=max_lags)
axes[1, 1].set_title(f'PACF of Daily Engagement Rate (lags={max_lags})')

plt.tight_layout()
plt.show()


# - Significant autocorrelation at lag 1 in both series indicates strong immediate persistence, where today's values strongly influence tomorrow's
# 
# - Weekly seasonality patterns (lags 7, 14) are not strongly evident, suggesting daily patterns dominate over weekly cycles in this short timeframe
# 
# - Rapid PACF decay after lag 1 suggests an AR(1) process may be sufficient, with minimal need for higher-order autoregressive terms

# In[ ]:





# In[31]:


# Since we have limited data, let's use simpler models
print("\nWith only 21 days of data, we'll use simple models:")

# Simple moving average as baseline
def simple_moving_average(series, window=3):
    return series.rolling(window=window).mean().iloc[-len(test):]

# Naive forecast (last value)
def naive_forecast(series):
    last_value = series.iloc[-1]
    return pd.Series([last_value] * len(test), index=test.index)

# ARIMA model for Users (simple order due to small dataset)
try:
    # For small datasets, use simple ARIMA orders
    model_users = ARIMA(train['Users'], order=(1,0,1))
    model_users_fit = model_users.fit()
    print("ARIMA(1,0,1) model for Users fitted successfully")

    # Forecast Users
    users_forecast = model_users_fit.forecast(steps=len(test))

except Exception as e:
    print(f"Error in ARIMA modeling for Users: {e}")
    print("Using naive forecast for Users")
    users_forecast = naive_forecast(train['Users'])

# ARIMA model for Engagement Rate
try:
    model_engagement = ARIMA(train['Engagement rate'], order=(1,0,1))
    model_engagement_fit = model_engagement.fit()
    print("ARIMA(1,0,1) model for Engagement Rate fitted successfully")

    # Forecast Engagement Rate
    engagement_forecast = model_engagement_fit.forecast(steps=len(test))

except Exception as e:
    print(f"Error in ARIMA modeling for Engagement Rate: {e}")
    print("Using naive forecast for Engagement Rate")
    engagement_forecast = naive_forecast(train['Engagement rate'])

# Also create simple baseline forecasts for comparison
users_ma = simple_moving_average(train['Users'])
engagement_ma = simple_moving_average(train['Engagement rate'])

# Plot forecasts
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(train.index, train['Users'], label='Train', marker='o')
plt.plot(test.index, test['Users'], label='Test', marker='o')
plt.plot(test.index, users_forecast, label='ARIMA Forecast', color='red', linestyle='--', marker='x')
plt.plot(test.index, users_ma, label='Moving Average (3)', color='green', linestyle='--', marker='x')
plt.plot(test.index, naive_forecast(train['Users']), label='Naive Forecast', color='orange', linestyle='--', marker='x')
plt.title('Users Forecast Comparison')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(train.index, train['Engagement rate'], label='Train', marker='o')
plt.plot(test.index, test['Engagement rate'], label='Test', marker='o')
plt.plot(test.index, engagement_forecast, label='ARIMA Forecast', color='red', linestyle='--', marker='x')
plt.plot(test.index, engagement_ma, label='Moving Average (3)', color='green', linestyle='--', marker='x')
plt.plot(test.index, naive_forecast(train['Engagement rate']), label='Naive Forecast', color='orange', linestyle='--', marker='x')
plt.title('Engagement Rate Forecast Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ### Forecast Performance Comparison Across Multiple Models
# 
# **Chart Type:** Multi-model time series forecast comparison  
# **Analysis Period:** 21 days of daily data (April 13 - May 3, 2024)  
# **Models Tested:** ARIMA(1,0,1), 3-Day Moving Average, Naive (Last Value)
# 
# **Key Observations:**
# - **ARIMA shows moderate performance** with some deviation from actual test values, particularly noticeable in the Engagement Rate forecast where it overestimates
# - **Moving Average provides stable predictions** that smooth out short-term fluctuations, performing reasonably well for both metrics
# - **Naive forecast demonstrates baseline performance** - while simple, it captures the general level but misses trend changes
# 
# **Practical Implications:**
# - No single model dramatically outperforms others, suggesting limited predictive signal in the short timeframe
# - The close clustering of forecast lines indicates high uncertainty in predictions with only 21 days of data
# - Moving Average may be the most reliable choice for operational forecasting given its stability
# 
# **Recommendation:** Use Moving Average for short-term planning while collecting more data to improve model selection accuracy.

# In[ ]:





# In[ ]:





# In[34]:


# Calculate evaluation metrics for all models
def calculate_metrics(true, pred, metric_name):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100 if np.all(true != 0) else float('inf')
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ARIMA metrics
users_arima_metrics = calculate_metrics(test['Users'], users_forecast, 'Users ARIMA')
engagement_arima_metrics = calculate_metrics(test['Engagement rate'], engagement_forecast, 'Engagement ARIMA')

# Moving Average metrics
users_ma_metrics = calculate_metrics(test['Users'], users_ma, 'Users MA')
engagement_ma_metrics = calculate_metrics(test['Engagement rate'], engagement_ma, 'Engagement MA')

# Naive metrics
users_naive = naive_forecast(train['Users'])
engagement_naive = naive_forecast(train['Engagement rate'])
users_naive_metrics = calculate_metrics(test['Users'], users_naive, 'Users Naive')
engagement_naive_metrics = calculate_metrics(test['Engagement rate'], engagement_naive, 'Engagement Naive')

# Create comparison table
metrics_comparison = pd.DataFrame({
    'Model': ['ARIMA', 'Moving Average', 'Naive'],
    'Users_MAE': [users_arima_metrics['MAE'], users_ma_metrics['MAE'], users_naive_metrics['MAE']],
    'Users_RMSE': [users_arima_metrics['RMSE'], users_ma_metrics['RMSE'], users_naive_metrics['RMSE']],
    'Users_MAPE': [users_arima_metrics['MAPE'], users_ma_metrics['MAPE'], users_naive_metrics['MAPE']],
    'Engagement_MAE': [engagement_arima_metrics['MAE'], engagement_ma_metrics['MAE'], engagement_naive_metrics['MAE']],
    'Engagement_RMSE': [engagement_arima_metrics['RMSE'], engagement_ma_metrics['RMSE'], engagement_naive_metrics['RMSE']],
    'Engagement_MAPE': [engagement_arima_metrics['MAPE'], engagement_ma_metrics['MAPE'], engagement_naive_metrics['MAPE']]
})

print("MODEL COMPARISON METRICS:")
print(metrics_comparison.to_string(index=False))


# In[ ]:





# ### Model Performance Evaluation Summary
# 
# **Evaluation Metrics:** MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error)  
# **Lower values indicate better performance** for all metrics
# 
# **Key Findings:**
# 
# - **ARIMA emerges as the best performer** for Users forecast with lowest MAE (3.16), RMSE (3.26), and MAPE (7.08%)
# - **Moving Average wins for Engagement Rate** with best MAE (0.019) and RMSE (0.024), outperforming even ARIMA
# - **Naive forecast significantly underperforms** both models across all metrics, confirming more sophisticated approaches add value
# - **MAPE values indicate good accuracy** - all below 16%, with best models achieving 3.9-7.1% error rates
# 
# **Practical Interpretation:**
# - ARIMA reduces user prediction error by **53%** compared to Naive baseline
# - Moving Average reduces engagement rate error by **42%** compared to Naive
# - The 3.9% MAPE for Engagement Rate means predictions are **96% accurate** on average
# 
# **Recommendation:** Use ARIMA for Users forecasting and Moving Average for Engagement Rate prediction based on their respective superior performance.

# ## Simple Exponential Smoothing (Better for Small Datasets)
# 

# In[35]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Simple Exponential Smoothing for Users
try:
    ses_model_users = SimpleExpSmoothing(train['Users']).fit()
    ses_forecast_users = ses_model_users.forecast(len(test))
    users_ses_metrics = calculate_metrics(test['Users'], ses_forecast_users, 'Users SES')

    # Add to comparison
    metrics_comparison.loc[len(metrics_comparison)] = [
        'Exponential Smoothing',
        users_ses_metrics['MAE'], users_ses_metrics['RMSE'], users_ses_metrics['MAPE'],
        np.nan, np.nan, np.nan  # Only for Users for now
    ]
except Exception as e:
    print(f"Error in Exponential Smoothing: {e}")

print("\nUPDATED MODEL COMPARISON:")
print(metrics_comparison.to_string(index=False))


# In[ ]:





#  ## Final Insights for Small Dataset

# In[36]:


print("\n" + "="*60)
print("FINAL INSIGHTS FOR SMALL DATASET (21 DAYS)")
print("="*60)

print(f"\nDataset Size: {len(daily_df)} days")
print(f"Training Period: {len(train)} days")
print(f"Testing Period: {len(test)} days")

print("\nRECOMMENDATIONS FOR SMALL DATASET:")
print("1. Collect more data for better model performance")
print("2. Use simple models (Moving Average, Naive) as benchmarks")
print("3. Consider domain knowledge for seasonal patterns")
print("4. Monitor forecast performance as more data becomes available")
print("5. Use confidence intervals to understand forecast uncertainty")

print("\nNEXT STEPS:")
print("✓ Continue collecting daily data")
print("✓ Re-run analysis monthly to incorporate new data")
print("✓ Consider external variables (weekends, holidays, promotions)")
print("✓ Implement simple monitoring system with moving averages")

# Plot final comparison
best_users_model = metrics_comparison.loc[metrics_comparison['Users_MAE'].idxmin(), 'Model']
best_engagement_model = metrics_comparison.loc[metrics_comparison['Engagement_MAE'].idxmin(), 'Model']

print(f"\nBEST PERFORMING MODELS:")
print(f"Users: {best_users_model} (lowest MAE)")
print(f"Engagement Rate: {best_engagement_model} (lowest MAE)")

# Save results for future comparison
forecast_results = {
    'train_size': len(train),
    'test_size': len(test),
    'users_forecast': users_forecast,
    'engagement_forecast': engagement_forecast,
    'metrics_comparison': metrics_comparison,
    'last_date': daily_df.index.max()
}

print(f"\nAnalysis completed. Ready to incorporate new data as it becomes available.")


# In[ ]:





# ## Project Summary & Insights

# This project analyzed website traffic data from April to May 2024, focusing on user visits, sessions, and engagement across channels like Direct, Organic Search, and Organic Social. 
# Using time series techniques (like trend analysis and forecasting models such as ARIMA), we uncovered patterns in hourly and daily activity to predict future performance and spot opportunities for improvement.
#     
# ## 🎯 Key Findings
# - **Organic Social dominates** with 70+ average users/sessions, driving the highest engagement
# - **Strong correlations** exist between users, sessions, and engagement (0.95-0.99)
# - **Daily patterns outweigh weekly cycles** in the short timeframe analyzed
# - **Both user traffic and engagement rates are predictable** with 90%+ accuracy
# - **Simple models work well** - ARIMA reduced prediction error by 53% compared to naive methods
# 
# ## ⏰ Temporal Patterns
# - Peak activity occurs during midday hours
# - Consistent daily patterns with minimal weekly seasonality
# - Stable overall trends with predictable fluctuations
# 

# In[ ]:





# ## Recommendations

# - Gather More Data: Collect at least 3-6 months of traffic info to improve forecast accuracy and capture longer trends like holidays or seasons.
#     
# - Focus on Top Channels: Invest more in Organic Social for growth, while boosting underperformers like Direct with targeted promotions.
#     
# - Optimize Timing: Schedule content or ads for peak hours (evenings) and days (mid-week) to maximize engagement.
#     
# - Monitor Regularly: Re-run the analysis monthly, adding factors like events or ads, and use simple tools like moving averages for quick checks.
# 
# 

# *Note: These recommendations are based on 28 days of data - accuracy will improve significantly with more historical data collection.*
# 

# In[ ]:





# ✍️ Author
# 
# Uzoh C. Hillary - Data Scientist / Data Analyst
# 
# GitHub: https://github.com/Uzo-Hill
# 
# LinkedIn: http://www.linkedin.com/in/hillaryuzoh

# In[ ]:




