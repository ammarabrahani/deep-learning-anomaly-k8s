"""
Project: Deep Q-Learning Based Anomaly Detection for Kubernetes Environment
Author: Ammar Yousuf Abrahani
Date: June 2025

Description:
This script implements a Deep Q-Learning model to detect anomalies in Kubernetes resource usage data.
The dataset used is synthetically generated to simulate realistic CPU, Memory, Network, and Disk usage patterns.

Adapted from: Concepts inspired by:
- https://github.com/gcamfer/Anomaly-ReactionRL
- https://www.tensorflow.org/agents
- Custom implementation written with assistance and guidance.
"""



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load the synthetic Kubernetes dataset
data = pd.read_csv('k8_synthetic_dataset.csv')

# Feature selection
X = data[['cpu_usage', 'memory_usage', 'network_io', 'disk_io']].values
y = data['label'].values.astype(int)

# Split dataset into training and testing (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define Deep Q-Learning Model (Q-Network)
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='linear')  # Q-values for actions: Normal(0) and Anomaly(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# DQL parameters
episodes = 5
epsilon = 0.9  # Initial exploration rate
gamma = 0.95   # Discount factor

# Training Loop
for episode in range(episodes):
    print(f"Episode {episode+1}/{episodes}")
    for i in range(len(X_train)):
        state = np.reshape(X_train[i], [1, X_train.shape[1]])

        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(2)  # Explore
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])  # Exploit

        # Reward assignment
        reward = 1 if action == y_train[i] else -1

        # Predict future state value
        target = reward
        if i < len(X_train) - 1:
            next_state = np.reshape(X_train[i + 1], [1, X_train.shape[1]])
            next_q_values = model.predict(next_state, verbose=0)
            target += gamma * np.amax(next_q_values[0])

        # Update Q-values
        q_values = model.predict(state, verbose=0)
        q_values[0][action] = target

        model.fit(state, q_values, epochs=1, verbose=0)

    # Decay exploration rate
    epsilon *= 0.95
    print(f"Updated epsilon: {epsilon:.4f}")

# Model evaluation on test data
y_pred = []
for state in X_test:
    q_values = model.predict(np.reshape(state, [1, X_test.shape[1]]), verbose=0)
    predicted_action = np.argmax(q_values[0])
    y_pred.append(predicted_action)

# Print detailed classification report
report = classification_report(
    y_test, y_pred, target_names=['Normal', 'Anomaly'], digits=4
)

print("\nFinal Deep Q-Learning Performance:")
print(report)
