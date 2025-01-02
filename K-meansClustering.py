import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from geopy.distance import geodesic

# Load the dataset
file_path = "maritime_data.csv"  # Use the expanded dataset with 1000 points
data = pd.read_csv(file_path)

# Predefined port coordinates for distance calculation
ports = {
    "SUEZ": (29.9668, 32.5498),
    "SIDI KERIR": (31.1561, 29.8652),
    "ROTTERDAM": (51.9225, 4.4792),
    "HAMBURG": (53.5511, 9.9937),
    "SHANGHAI": (31.2304, 121.4737),
    "SINGAPORE": (1.3521, 103.8198),
    "DUBAI": (25.276987, 55.296249),
    "LOS ANGELES": (34.0522, -118.2437),
    "TOKYO": (35.6895, 139.6917),
    "SYDNEY": (-33.8688, 151.2093)
}

# Create a target column for the next destination
data['Next_Destination'] = data['DESTINATION'].shift(-1)

# Drop rows where Next_Destination is NaN (last row has no next destination)
data = data.dropna(subset=['Next_Destination'])

# Feature Engineering: Calculate distance to next destination
def calculate_distance(row):
    if row['Next_Destination'] in ports:
        return geodesic((row['LATITUDE'], row['LONGITUDE']), ports[row['Next_Destination']]).km
    return None

data['Distance_to_Destination'] = data.apply(calculate_distance, axis=1)

# Drop rows with NaN distances (if any)
data = data.dropna(subset=['Distance_to_Destination'])

# Select features and target
features = data[["LATITUDE", "LONGITUDE", "COURSE", "SPEED", "Distance_to_Destination"]]
target = data["Next_Destination"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Use the best model
model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassifier Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

# Analyze feature importance
print("\nFeature Importances:")
for feature, importance in zip(features.columns, model.feature_importances_):
    print(f"{feature}: {importance}")

# Optional: Save the trained model for later use
import joblib
joblib.dump(model, "ship_destination_predictor_with_features.pkl")


#UNCOMMNET EVERYHING BELOW FOR K MEANS CLUSTERING
'''
# Select the features for clustering
features = data[["LATITUDE", "LONGITUDE"]]

# Apply K-means clustering
k = 3  # Choose the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# Add cluster centers to the plot
centers = kmeans.cluster_centers_

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data["LATITUDE"], data["LONGITUDE"], c=data["Cluster"], cmap="viridis", label="Ship Positions")
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Cluster Centers")
plt.title("K-means Clustering of Ship Positions")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend()
plt.show()

# Display cluster centers
print("Cluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: Latitude {center[0]}, Longitude {center[1]}")

# Display the first few rows of the dataset
#print("Dataset loaded successfully:")
#print(data)
'''
