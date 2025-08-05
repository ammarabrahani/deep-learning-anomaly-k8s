import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

class RealDeepQLWrapper(BaseEstimator):
    def __init__(self, episodes=1, gamma=0.95, arch="simple"):
        self.episodes = episodes
        self.gamma = gamma
        self.arch = arch
        self.model = None

    def _build_model(self, input_dim, arch="simple"):
        if arch == "simple":
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(32, activation='relu'),
                Dense(2, activation='linear')
            ])
        elif arch == "deep":
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(2, activation='linear')
            ])
        elif arch == "wide":
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(2, activation='linear')
            ])
        else:
            raise ValueError("Unknown architecture type")

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def fit(self, X, y=None):
        print("🚀 Training started...")
        y_simulated = self._simulate_labels(X)
        self.model = self._build_model(X.shape[1], self.arch)

        epsilon = 0.9

        for episode in range(self.episodes):
            print(f"🎯 Episode {episode + 1}/{self.episodes}")
            for i in range(len(X)):
                state = np.reshape(X[i], [1, X.shape[1]])
                if np.random.rand() < epsilon:
                    action = np.random.randint(2)
                else:
                    q_values = self.model.predict(state, verbose=0)
                    action = np.argmax(q_values[0])

                reward = 1 if action == y_simulated[i] else -1

                target = reward
                if i < len(X) - 1:
                    next_state = np.reshape(X[i + 1], [1, X.shape[1]])
                    next_q = self.model.predict(next_state, verbose=0)
                    target += self.gamma * np.amax(next_q[0])

                q_values = self.model.predict(state, verbose=0)
                q_values[0][action] = target
                self.model.fit(state, q_values, epochs=1, verbose=0)

            epsilon *= 0.95

        print("✅ Training completed.")
        return self

    def predict(self, X):
        q_values = self.model.predict(X, verbose=0)
        actions = np.argmax(q_values, axis=1)
        return np.where(actions == 0, 1, -1)

    def _simulate_labels(self, X):
        center = np.median(X, axis=0)
        dists = np.linalg.norm(X - center, axis=1)
        threshold = np.percentile(dists, 85)
        return (dists < threshold).astype(int)

