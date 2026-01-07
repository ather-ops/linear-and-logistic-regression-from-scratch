import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("student_scores_csv")
print("Original Data:")
print(df.head())
print("=" * 80)

# -----------------------------
# Features & Target
# -----------------------------
X1 = df["study_hours"].values
X2 = df["sleep_hours"].values
X3 = df["attendance"].values
Y  = df["final_score"].values

# -----------------------------
# Feature Scaling (Normalization)
# -----------------------------
X1 = X1 / X1.max()
X2 = X2 / X2.max()
X3 = X3 / X3.max()

# -----------------------------
# Initialize Parameters
# -----------------------------
m1 = m2 = m3 = b = 0.0
lr = 0.00001
epochs = 2000
n = len(Y)

loss_history = []

# -----------------------------
# Training Loop (Gradient Descent)
# -----------------------------
for epoch in range(epochs):
    y_pred = m1*X1 + m2*X2 + m3*X3 + b
    error = Y - y_pred

    loss = np.mean(error ** 2)
    loss_history.append(loss)

    dm1 = -(2/n) * np.sum(X1 * error)
    dm2 = -(2/n) * np.sum(X2 * error)
    dm3 = -(2/n) * np.sum(X3 * error)
    db  = -(2/n) * np.sum(error)

    m1 -= lr * dm1
    m2 -= lr * dm2
    m3 -= lr * dm3
    b  -= lr * db

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# -----------------------------
# Final Parameters
# -----------------------------
print("=" * 80)
print("Trained Model Parameters:")
print(f"m1 (study): {m1:.4f}")
print(f"m2 (sleep): {m2:.4f}")
print(f"m3 (attendance): {m3:.4f}")
print(f"b (bias): {b:.4f}")

# -----------------------------
# Loss Visualization
# -----------------------------
plt.plot(loss_history, color="green")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# -----------------------------
# Prediction (New Student)
# -----------------------------
study = 7
sleep = 8
attendance = 85

study /= df["study_hours"].max()
sleep /= df["sleep_hours"].max()
attendance /= df["attendance"].max()

predicted_score = m1*study + m2*sleep + m3*attendance + b
print("=" * 80)
print(f"Predicted Final Score: {predicted_score:.2f}")

