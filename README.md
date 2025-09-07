# 📊 Website Performance Forecasting & Engagement Analysis

## 📌 Project Overview
This project applies **time series analysis** to website performance data, exploring hourly traffic, sessions, engagement, and conversions across marketing channels such as **Direct, Organic Search, Organic Social, and Referrals**.  

The goal is to uncover patterns, forecast trends, detect anomalies, and provide actionable insights for **user experience optimization, channel effectiveness, and strategic decision-making**.

---

## 🎯 Objectives
- Build and validate time-series forecasting models to predict **sessions & engagement**.  
- Identify **seasonal patterns** and peak performance hours.  
- Compare performance across marketing channels.  
- Evaluate forecasting models (ARIMA, Moving Average, Naive, SES) and recommend the best performers.  

---


## 📂 Dataset
- Source: [Statso Website Performance Case Study](https://statso.io/website-performance-case-study/)  
- Timeframe: **April – May 2024** (~28 days of hourly/daily data)  
- Key Features:  
  - Users  
  - Sessions  
  - Engaged Sessions  
  - Average Engagement Time per Session  
  - Engagement Rate  
  - Event Count  
  - Channel Group (Direct, Organic, Referral, etc.)

👉 *![Sample Dataset Screenshot](images/sample_dataset.png)*

---

## 🛠️ Tech Stack
- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`, `sklearn`  
- **Models Used**: ARIMA, Moving Average, Naive Forecast, Simple Exponential Smoothing (SES)

```bash
# Step 1: Activate your environment 
conda activate timeseries_env 

# Step 2: Validate your environment (optional but recommended) 
python -c "import numpy, matplotlib, pandas, seaborn, statsmodels; print('NumPy:', numpy.__version__); print('All imports working!')" 

# Step 3: Start Jupyter 
jupyter notebook
```

---

## 🔎 Data Preparation & Cleaning
- Renamed columns.
- Converted timestamp into **datetime index**.  
- Removed duplicates and checked for missing values (none found).  
- Outlier detection via **boxplots & histograms**.  
- Created time-based features: `hour`, `day_of_week`, `day_of_month`, `week_of_year`.  

```python
# Rename column:

df = df.rename(columns={"Session primary channel group (Default channel group)": "Session primary channel group"})
df = df.rename(columns={"Date + hour (YYYYMMDDHH)": "Datetime"})

# Convert the date column to proper datetime format:
df['Datetime'] = df['Datetime'].astype(str)
df['datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H')

# Set datetime as index:
df.set_index('datetime', inplace=True)

# Check for duplicates:
print("Duplicate rows:", df.duplicated().sum())

# Create additional time-based features:
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month
df['week_of_year'] = df.index.isocalendar().week
```

---
*![Sample Cleaned Dataset](images/sample_dataset.png)*

---


## 📊 Exploratory Data Analysis
- Distribution analysis of key metrics showed **right-skewed activity data**.  
- Channel Analysis revealed **Organic Social** as the top driver of traffic & engagement.  
- Time-series plots highlighted **midday traffic spikes** and **variable engagement rates**.  

👉 *![Distributions](images/distributions.png)*  
👉 *![Channel Analysis](images/channel_analysis.png)*  
👉 *![Time Series Trends](images/time_trends.png)*  

---

## ⏳ Time Series Decomposition
- Weekly seasonality patterns identified for **Users**.  
- Engagement Rate showed a temporary **decline mid-April** before recovery.  

👉 *![Decomposition](images/decomposition.png)*

---

## 📈 Forecasting Models
Models tested:
- **ARIMA(1,0,1)**  
- **3-Day Moving Average**  
- **Naive Forecast (last value)**  
- **Simple Exponential Smoothing (SES)**  

👉 *![ACF/PACF](images/acf_pacf.png)*  
👉 *![Forecast Comparison](images/forecast_comparison.png)*  

---

## 📊 Model Evaluation
Metrics used: **MAE, RMSE, MAPE**  

| Model                  | Users MAE | Users RMSE | Users MAPE | Engagement MAE | Engagement RMSE | Engagement MAPE |
|-------------------------|-----------|------------|------------|----------------|-----------------|-----------------|
| ARIMA                  | 3.16      | 3.26       | 7.08%      | 0.020          | 0.025           | 3.89%           |
| Moving Average (3-Day) | 3.35      | 3.85       | -          | 0.019          | 0.024           | -               |
| Naive Forecast         | 6.77      | 6.85       | 15.20%     | 0.032          | 0.039           | 6.11%           |
| Exponential Smoothing  | 6.76      | 6.83       | 15.18%     | -              | -               | -               |

✅ **Best Models**:  
- **Users** → ARIMA  
- **Engagement Rate** → Moving Average  

---

## 📌 Key Insights
- **Organic Social** drives the highest traffic and engagement.  
- **Daily cycles dominate** over weekly seasonality.  
- ARIMA reduced prediction error by **53%** compared to the naive model.  
- Engagement Rate predictions achieved **96% accuracy** (MAPE ~3.9%).  

---













