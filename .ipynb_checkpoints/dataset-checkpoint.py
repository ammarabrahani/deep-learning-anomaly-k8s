import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
num_samples = 300

# Generate synthetic Kubernetes metrics (Normal)
cpu_usage = np.random.normal(loc=50, scale=10, size=num_samples)
memory_usage = np.random.normal(loc=50, scale=10, size=num_samples)
network_io = np.random.normal(loc=300, scale=50, size=num_samples)
disk_io = np.random.normal(loc=100, scale=20, size=num_samples)

# Generate anomalies (5% anomalies)
num_anomalies = int(0.05 * num_samples)
anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)

# Introduce anomalies
cpu_usage[anomaly_indices] = np.random.normal(loc=90, scale=5, size=num_anomalies)
memory_usage[anomaly_indices] = np.random.normal(loc=90, scale=5, size=num_anomalies)
network_io[anomaly_indices] = np.random.normal(loc=600, scale=30, size=num_anomalies)
disk_io[anomaly_indices] = np.random.normal(loc=200, scale=10, size=num_anomalies)

# Labels (0: Normal, 1: Anomaly)
labels = np.zeros(num_samples)
labels[anomaly_indices] = 1

# Create DataFrame
data = pd.DataFrame({
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage,
    'network_io': network_io,
    'disk_io': disk_io,
    'label': labels
})

# Save dataset as CSV
data.to_csv("k8_synthetic_dataset.csv", index=False)

print("Synthetic Kubernetes dataset generated and saved as 'k8_synthetic_dataset.csv'")
