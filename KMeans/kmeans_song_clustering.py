import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
file_path = r"C:\Users\honge\OneDrive\Desktop\KMeans\moststreamed2024.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Convert relevant columns to numeric values
numeric_columns = [
    "Spotify Streams", "Spotify Playlist Count", "Spotify Playlist Reach",
    "Deezer Playlist Count", "Deezer Playlist Reach", "Amazon Playlist Count",
    "Pandora Streams", "Pandora Track Stations", "Soundcloud Streams", 
    "Shazam Counts"
]

# Remove commas and convert to numeric
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# Drop rows with NaN values in relevant numeric columns
df_cleaned = df.dropna(subset=numeric_columns)

# Standardizing the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned[numeric_columns])

# Determine optimal K using the Elbow Method
inertia = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Choose optimal K (based on elbow method)
k_optimal = 4  # Adjust based on elbow plot
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df_cleaned.loc[:, 'Cluster'] = kmeans.fit_predict(X_scaled)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cleaned['PCA1'] = X_pca[:, 0]
df_cleaned['PCA2'] = X_pca[:, 1]

# Plot clusters
plt.figure(figsize=(8, 5))
for cluster in range(k_optimal):
    plt.scatter(
        df_cleaned[df_cleaned['Cluster'] == cluster]['PCA1'],
        df_cleaned[df_cleaned['Cluster'] == cluster]['PCA2'],
        label=f'Cluster {cluster}'
    )

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering of Songs')
plt.legend()
plt.show()

# Display clustered dataset
import ace_tools as tools
tools.display_dataframe_to_user(name='Clustered Songs Data', dataframe=df_cleaned)
