# Urban Heat Island (UHI) Analysis and Prediction Project

# Import necessary libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import folium

# Step 1: Data Collection
# Load satellite data, urban infrastructure, and weather data
# Example: Download datasets from NASA EarthData or other public APIs
# For demonstration, using mock data files
try:
    satellite_data = pd.read_csv('satellite_data.csv')  # Land Surface Temperature (LST) data
    urban_data = pd.read_csv('urban_data.csv')  # Urban infrastructure data
    weather_data = pd.read_csv('weather_data.csv')  # Weather data for the region
except FileNotFoundError:
    print("Ensure all required data files are downloaded and present in the directory.")
    exit()

# Step 2: Data Preprocessing
# Merge datasets on common geographic identifiers
merged_data = pd.merge(satellite_data, urban_data, on="region_id", how="inner")
merged_data = pd.merge(merged_data, weather_data, on="region_id", how="inner")

# Handle missing values
merged_data.fillna(merged_data.mean(), inplace=True)

# Feature engineering
merged_data['population_density'] = merged_data['population'] / merged_data['area_sq_km']
merged_data['green_cover_ratio'] = merged_data['green_area_sq_km'] / merged_data['area_sq_km']

# Step 3: Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(merged_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Visualize temperature vs green cover
sns.scatterplot(x=merged_data['green_cover_ratio'], y=merged_data['temperature'])
plt.title("Green Cover Ratio vs Temperature")
plt.xlabel("Green Cover Ratio")
plt.ylabel("Temperature (°C)")
plt.show()

# Step 4: Clustering for UHI Zones
# Select relevant features for clustering
features = merged_data[['temperature', 'population_density', 'green_cover_ratio']]
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data['UHI_zone'] = kmeans.fit_predict(features)

# Visualize clusters
sns.scatterplot(x=merged_data['population_density'], y=merged_data['temperature'], hue=merged_data['UHI_zone'], palette="viridis")
plt.title("UHI Zones Clustering")
plt.xlabel("Population Density")
plt.ylabel("Temperature (°C)")
plt.show()

# Step 5: Predictive Modeling
# Define input features and target variable
X = merged_data[['population_density', 'green_cover_ratio', 'urban_area_ratio']]
y = merged_data['temperature']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance)

# Step 6: Visualization and Deployment
# Interactive map showing UHI zones
m = folium.Map(location=[merged_data['latitude'].mean(), merged_data['longitude'].mean()], zoom_start=10)
for _, row in merged_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['UHI_zone'] == 2 else ('orange' if row['UHI_zone'] == 1 else 'green'),
        fill=True,
        fill_opacity=0.6,
        popup=f"Temp: {row['temperature']}°C, Pop Density: {row['population_density']}",
    ).add_to(m)
m.save("UHI_zones_map.html")
print("Interactive map saved as 'UHI_zones_map.html'. Open this file in your browser to view the map.")

# Deployment-ready visualization dashboard
# Use Streamlit for building interactive dashboards
# Code omitted for brevity but can be added if required
