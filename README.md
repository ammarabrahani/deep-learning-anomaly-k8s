
# Deep Learning-based Anomaly Detection

## 🧠 Project Overview

This research project explores the application of deep learning models, particularly **Autoencoders**, **Multi-Layer Perceptrons (MLP)**, and **Deep Q-Learning (DQL)**, for anomaly detection. The models are benchmarked against classical unsupervised methods such as **Isolation Forest**, and **Local Outlier Factor**

The project includes:
- Evaluation on Scikit-learn datasets (e.g., blobs, moons)
- Application to the **Kaggle Credit Card Fraud Detection** dataset
- Custom implementation of a Deep Q-Learning anomaly detection model
- Comparison of macro F1-score across models and datasets
- Visualization of decision boundaries

---

## 🗂️ Project Structure

```
NCI-Research-Project/
│
├── code/
│   ├── autoencoder_anomaly_detection.ipynb        # Autoencoder on Scikit-learn
│   ├── mlp_classifier_imbalanced_dataset.ipynb    # MLP on imbalanced Scikit-learn
│   ├── real_dql_model.py                          # Custom DQL implementation
│   ├── comparison_with_real_dql.png               # Final model comparison visualization
│   ├── creditcard_autoencoder_eval.ipynb          # Deep Autoencoder on credit card data
│   └── README.md                                   # Project documentation
│
├── Notebook/
│   └── data/
│       └── raw/
│           └── creditcard.csv (NOT INCLUDED in repo)
```

---

## 📊 Dataset: Credit Card Fraud Detection

We use the **Credit Card Fraud Detection** dataset from Kaggle:

🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Due to GitHub's 100MB file limit, the dataset is not included.  
After downloading, place it here:

```
Notebook/data/raw/creditcard.csv
```

---

## ⚙️ Models Used

- **Autoencoder**: Deep symmetric architecture with dropout and regularization
- **MLP Classifier**: Simple feedforward model for balanced data comparison
- **Deep Q-Learning**: Custom agent-based anomaly detector
- **Classical Models**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - SGDOneClassSVM

---

## 📈 Evaluation Metrics

- **Macro Average F1-Score**
- **Classification Report**
- **Reconstruction Error (MSE)**
- **Decision Boundary Visualizations**

---

## 🧪 How to Run

Make sure you have the required libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

Run each notebook individually depending on the model and dataset.

---

## 🛡️ Disclaimer

This repository is intended for **academic and research purposes** only.  
Do not reuse without proper citation of:
- The Kaggle dataset
- TensorFlow/Keras
- scikit-learn
- Any third-party GitHub or Kaggle tutorials referenced in headers

---

## 📬 Contact

For questions or contributions, please open an issue or contact:

**Ammar Abrahani**  
[GitHub](https://github.com/ammarabrahani/deep-learning-anomaly-k8s)

