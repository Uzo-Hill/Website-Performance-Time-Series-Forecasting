# Website-Performance-Time-Series-Forecasting
Time series analysis and forecasting of website performance data to uncover traffic patterns, detect anomalies, and predict future user engagement. The project applies statistical models (ARIMA, seasonal decomposition) on hourly data from multiple marketing channels to generate actionable business insights and recommendations.


---

## 📌 Project Overview  
This project applies **time series analysis and forecasting** techniques to website performance data. The goal is to uncover traffic and engagement patterns, detect anomalies, and build predictive insights to support **data-driven decision-making** for digital strategy.  

The dataset was obtained from the [Statso Website Performance Case Study](https://statso.io/website-performance-case-study/).  

Key focus areas include:  
- Hourly **user traffic** and **sessions** trends.  
- **Engagement rate** behavior across channels.  
- Identifying **spikes/outliers** in performance.  
- Forecasting future website performance using **ARIMA models**.  

---

## 🛠️ Technical Stack

```python
Python 3.10
NumPy 1.26.4
Pandas 2.1.4
Matplotlib 3.8.2
Seaborn 0.13.0
Statsmodels 0.14.5
Scikit-learn 1.3.2
```

---


## ⚙️ Project Workflow  

1. **Data Import & Cleaning**  
   - Parsed datetime column into proper time index.  
   - Handled missing values and outliers.  
   - Feature engineering (hour, day, week, month).  

   📌 *[Insert sample data screenshot here]*  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution analysis of numerical features.  
   - Time series visualization of key metrics (Users, Sessions, Engagement Rate).  
   - Outlier detection using the **IQR method**.  

   📊 *[Insert plots: distributions, boxplots, time series]*  

3. **Trend & Seasonality Analysis**  
   - Applied **Seasonal Decomposition** to detect trend, seasonality, and residuals.  
   - Identified periodic patterns in traffic and engagement.  

   📊 *[Insert seasonal decomposition chart]*  

4. **Forecasting**  
   - Built forecasting models using **ARIMA/SARIMA**.  
   - Evaluated model performance with **RMSE**.  
   - Generated short-term forecasts for Users, Sessions, and Engagement Rate.  

   📈 *[Insert forecast plots]*  
