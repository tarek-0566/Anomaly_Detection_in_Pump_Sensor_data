# Anomaly Detection in Pump Sensor Data

## Project Overview

This project aims to detect anomalies in pump sensor data using various data processing and machine learning techniques. The primary goal is to identify abnormal readings from sensors that might indicate the malfunction or breakdown of the pump.

## Requirements

- Python 3.6+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - seaborn
  - matplotlib
  - statsmodels

Install the required libraries using:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib statsmodels
```

## Dataset

The dataset consists of sensor readings from a pump over time, with a timestamp and various sensor readings. The dataset also includes the status of the machine (e.g., NORMAL, BROKEN, RECOVERING).

## Steps Involved

1. **Data Loading and Preprocessing**
   - Load the dataset using pandas.
   - Remove duplicate entries.
   - Remove columns with all NaN values (e.g., 'sensor_15').
   - Convert the timestamp column to datetime format.
   - Fill missing values with the mean of each column.

2. **Exploratory Data Analysis (EDA)**
   - Calculate the percentage of missing values in each column.
   - Visualize sensor data using box plots.
   - Plot sensor readings for the BROKEN state of the pump.

3. **Time Series Resampling**
   - Resample the dataset to obtain daily averages and standard deviations.

4. **Principal Component Analysis (PCA)**
   - Standardize the dataset.
   - Apply PCA to reduce the dimensionality of the dataset.
   - Visualize the importance of the principal components.

5. **Stationarity Test**
   - Perform the Augmented Dickey-Fuller test on the principal components to check for stationarity.

6. **Autocorrelation Analysis**
   - Plot the Autocorrelation Function (ACF) of the principal components.

7. **Anomaly Detection**
   - Use Isolation Forest to detect anomalies.
   - Use K-Means clustering to identify anomalies based on distance from cluster centroids.
   - Visualize the anomalies detected by both methods.

## Detailed Steps

### Data Loading and Preprocessing
```python
import pandas as pd

# Load dataset
df = pd.read_csv('/content/sensor.csv')

# Remove duplicates
df = df.drop_duplicates()

# Remove 'sensor_15' column
del df['sensor_15']

# Convert timestamp to datetime
df['date'] = pd.to_datetime(df['timestamp'])
del df['timestamp']

# Fill missing values with column means
df = df.fillna(df.mean())
```

### Exploratory Data Analysis (EDA)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate percentage of missing values
def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False) / len(df), columns=['percent'])
    idx = nans['percent'] > 0
    return nans[idx]

# Box plot for sensor_05
sns.boxplot(df['sensor_05'])
plt.show()
```

### Time Series Resampling
```python
# Set date as index
df = df.set_index('date')

# Resample data by daily average and standard deviation
rollmean = df.resample(rule='D').mean()
rollstd = df.resample(rule='D').std()
```

### Principal Component Analysis (PCA)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Standardize and apply PCA
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(df.drop(['machine_status'], axis=1))

# Plot principal components against their inertia
features = range(pca.n_components_)
plt.figure(figsize=(15, 5))
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.title('Importance of the Principal Components based on inertia')
plt.show()
```

### Stationarity Test
```python
from statsmodels.tsa.stattools import adfuller

# Run Augmented Dickey Fuller Test
result = adfuller(principalDf['pc1'])
print(result[1])  # Print p-value
```

### Autocorrelation Analysis
```python
from statsmodels.graphics.tsaplots import plot_acf

# Plot ACF of the first principal component
plot_acf(principalDf['pc1'], lags=20, alpha=0.05)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function of PC1')
plt.show()
```

### Anomaly Detection
#### Using Isolation Forest
```python
from sklearn.ensemble import IsolationForest

# Fit Isolation Forest model
outliers_fraction = 0.13
model = IsolationForest(contamination=outliers_fraction)
model.fit(principalDf.values)

# Predict anomalies
principalDf['anomaly2'] = pd.Series(model.predict(principalDf.values))
df['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df.index)

# Visualize anomalies
a = df.loc[df['anomaly2'] == -1]  # Anomaly
plt.figure(figsize=(18, 6))
plt.plot(df['sensor_12'], color='blue', label='Normal')
plt.plot(a['sensor_12'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
plt.xlabel('Date and Time')
plt.ylabel('Sensor Reading')
plt.title('Sensor_12 Anomalies')
plt.legend(loc='best')
plt.show()
```

#### Using K-Means Clustering
```python
from sklearn.cluster import KMeans

# Fit K-Means model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(principalDf.values)
labels = kmeans.predict(principalDf.values)

# Calculate distance from cluster centroids
def getDistanceByPoint(data, model):
    distance = []
    for i in range(0, len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i] - 1]
        distance.append(np.linalg.norm(Xa - Xb))
    return pd.Series(distance, index=data.index)

# Detect anomalies based on distance
distance = getDistanceByPoint(principalDf, kmeans)
number_of_outliers = int(outliers_fraction * len(distance))
threshold = distance.nlargest(number_of_outliers).min()
principalDf['anomaly1'] = (distance >= threshold).astype(int)
df['anomaly1'] = pd.Series(principalDf['anomaly1'].values, index=df.index)

# Visualize anomalies
a = df.loc[df['anomaly1'] == 1]  # Anomaly
plt.figure(figsize=(18, 6))
plt.plot(df['sensor_12'], color='blue', label='Normal')
plt.plot(a['sensor_12'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
plt.xlabel('Date and Time')
plt.ylabel('Sensor Reading')
plt.title('Sensor_12 Anomalies')
plt.legend(loc='best')
plt.show()
```

## Conclusion

This project demonstrates the process of anomaly detection in pump sensor data using various machine learning techniques. By applying Isolation Forest and K-Means clustering, we can effectively identify abnormal sensor readings that might indicate potential issues with the pump.
