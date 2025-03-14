# Pregunta-19
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np

# Step 1: Create the dataset from the provided document
data = {
    'Date': pd.date_range(start='1982-01-01', end='2001-06-01', freq='MS'),
    'GS3M': [
        12.92, 14.28, 13.31, 13.34, 12.71, 13.08, 11.86, 9.00, 8.19, 7.97, 8.35, 8.20, 8.12, 8.39, 8.66, 
        8.51, 8.50, 9.14, 9.45, 9.74, 9.36, 8.99, 9.11, 9.36, 9.26, 9.46, 9.89, 10.07, 10.22, 10.26, 10.53, 
        10.90, 10.80, 10.12, 8.92, 8.34, 8.02, 8.56, 8.83, 8.22, 7.73, 7.18, 7.32, 7.37, 7.33, 7.40, 7.48, 
        7.33, 7.30, 7.29, 6.76, 6.24, 6.33, 6.40, 6.00, 5.69, 5.35, 5.32, 5.50, 5.68, 5.58, 5.75, 5.77, 5.82, 
        5.85, 5.85, 5.88, 6.23, 6.62, 6.35, 5.89, 5.96, 6.00, 5.84, 5.87, 6.08, 6.45, 6.66, 6.95, 7.30, 7.48, 
        7.60, 8.03, 8.35, 8.56, 8.84, 9.14, 8.96, 8.74, 8.43, 8.15, 8.17, 8.01, 7.90, 7.94, 7.88, 7.90, 8.00, 
        8.17, 8.04, 8.01, 7.99, 7.87, 7.69, 7.60, 7.40, 7.29, 6.95, 6.41, 6.12, 6.09, 5.83, 5.63, 5.75, 5.75, 
        5.50, 5.37, 5.14, 4.69, 4.18, 3.91, 3.95, 4.14, 3.84, 3.72, 3.75, 3.28, 3.20, 2.97, 2.93, 3.21, 3.29, 
        3.07, 2.99, 3.01, 2.93, 3.03, 3.14, 3.11, 3.09, 3.01, 3.09, 3.18, 3.13, 3.04, 3.33, 3.59, 3.78, 4.27, 
        4.25, 4.46, 4.61, 4.75, 5.10, 5.45, 5.76, 5.90, 5.94, 5.91, 5.84, 5.85, 5.64, 5.59, 5.57, 5.43, 5.44, 
        5.52, 5.29, 5.15, 4.96, 5.10, 5.09, 5.15, 5.23, 5.30, 5.19, 5.24, 5.12, 5.17, 5.04, 5.17, 5.14, 5.28, 
        5.30, 5.20, 5.07, 5.19, 5.28, 5.08, 5.11, 5.28, 5.30, 5.18, 5.23, 5.16, 5.08, 5.14, 5.12, 5.09, 5.04, 
        4.74, 4.07, 4.53, 4.50, 4.45, 4.56, 4.57, 4.41, 4.63, 4.72, 4.69, 4.87, 4.82, 5.02, 5.23, 5.36, 5.50, 
        5.73, 5.86, 5.82, 5.99, 5.86, 6.14, 6.28, 6.18, 6.29, 6.36, 5.94, 5.29, 5.01, 4.54, 3.97, 3.70, 3.57
    ],
    'GS6M': [
        13.90, 14.81, 13.83, 13.87, 13.13, 13.76, 12.80, 10.51, 9.83, 8.63, 8.80, 8.59, 8.33, 8.65, 8.86, 
        8.78, 8.70, 9.44, 9.85, 10.16, 9.73, 9.39, 9.48, 9.76, 9.56, 9.77, 10.27, 10.47, 11.02, 11.24, 11.27, 
        11.37, 11.19, 10.52, 9.34, 8.76, 8.45, 8.87, 9.45, 8.71, 8.07, 7.46, 7.57, 7.71, 7.64, 7.71, 7.68, 
        7.50, 7.53, 7.47, 6.89, 6.36, 6.47, 6.56, 6.12, 5.79, 5.57, 5.48, 5.64, 5.78, 5.67, 5.83, 5.86, 6.19, 
        6.35, 6.28, 6.05, 6.46, 6.99, 7.04, 6.50, 6.68, 6.56, 6.21, 6.18, 6.50, 6.89, 7.04, 7.35, 7.79, 7.82, 
        7.90, 8.30, 8.70, 8.85, 9.05, 9.39, 9.17, 8.91, 8.38, 8.01, 8.17, 8.16, 8.03, 7.89, 7.81, 7.96, 8.12, 
        8.28, 8.27, 8.19, 8.05, 7.92, 7.77, 7.70, 7.53, 7.39, 7.03, 6.58, 6.19, 6.20, 5.98, 5.87, 6.02, 5.97, 
        5.63, 5.48, 5.26, 4.80, 4.26, 4.01, 4.08, 4.33, 4.00, 3.88, 3.90, 3.38, 3.31, 3.04, 3.13, 3.44, 3.47, 
        3.24, 3.16, 3.15, 3.06, 3.17, 3.29, 3.26, 3.24, 3.15, 3.22, 3.36, 3.34, 3.25, 3.53, 3.92, 4.25, 4.79, 
        4.72, 4.95, 5.08, 5.24, 5.62, 5.98, 6.50, 6.51, 6.31, 6.17, 6.05, 5.93, 5.66, 5.62, 5.65, 5.54, 5.56, 
        5.51, 5.35, 5.13, 4.97, 5.16, 5.27, 5.33, 5.46, 5.52, 5.34, 5.45, 5.32, 5.27, 5.24, 5.31, 5.27, 5.48, 
        5.60, 5.53, 5.34, 5.33, 5.40, 5.30, 5.30, 5.38, 5.45, 5.23, 5.27, 5.25, 5.26, 5.36, 5.32, 5.23, 5.15, 
        4.81, 4.20, 4.59, 4.57, 4.49, 4.61, 4.65, 4.54, 4.75, 5.03, 4.75, 5.09, 5.08, 5.20, 5.43, 5.68, 5.76, 
        6.00, 6.11, 6.07, 6.39, 6.24, 6.27, 6.35, 6.25, 6.32, 6.34, 5.92, 5.15, 4.89, 4.44, 3.99, 3.74, 3.56
    ]
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# a. Plot the two time series
plt.figure(figsize=(12, 6))
plt.plot(df['GS3M'], label='3-Month T-Bill Rate (GS3M)')
plt.plot(df['GS6M'], label='6-Month T-Bill Rate (GS6M)')
plt.title('3-Month and 6-Month Treasury Bill Rates (1982-2001)')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# b. Unit root test (ADF)
def adf_test(series, title=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'ADF Test for {title}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')
    print('Stationary' if result[1] < 0.05 else 'Non-Stationary')
    print()

# Test in levels
adf_test(df['GS3M'], 'GS3M (Levels)')
adf_test(df['GS6M'], 'GS6M (Levels)')

# Test in first differences
df['D_GS3M'] = df['GS3M'].diff()
df['D_GS6M'] = df['GS6M'].diff()
adf_test(df['D_GS3M'], 'GS3M (First Differences)')
adf_test(df['D_GS6M'], 'GS6M (First Differences)')

# c. Cointegration test
coint_result = coint(df['GS3M'], df['GS6M'])
print('Cointegration Test (Engle-Granger):')
print(f'Test Statistic: {coint_result[0]}')
print(f'p-value: {coint_result[1]}')
print(f'Critical Values: {coint_result[2]}')
print('Cointegrated' if coint_result[1] < 0.05 else 'Not Cointegrated')
print()

# e. VAR model (assuming first differences for robustness)
diff_data = df[['D_GS3M', 'D_GS6M']].dropna()
model = VAR(diff_data)
results = model.fit(4)  # 4 lags
print(results.summary())

# Optional: VAR in levels (if cointegrated)
# level_data = df[['GS3M', 'GS6M']]
# model_levels = VAR(level_data)
# results_levels = model_levels.fit(4)
# print(results_levels.summary())
