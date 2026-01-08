import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- DATA ----------------
data = {
    "income":        [20000,25000,30000,40000,50000,60000,70000,80000,90000,100000],
    "credit_score":  [450,480,520,600,650,700,720,750,780,820],
    "loan_amount":   [300000,250000,230000,180000,160000,150000,180000,130000,120700,100000],
    "approved":      [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# ---------------- FUNCTIONS ----------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def standardize(X):
    return (X - np.mean(X)) / np.std(X)

# ---------------- FEATURES ----------------
X1 = standardize(df["income"].values)
X2 = standardize(df["credit_score"].values)
X3 = standardize(df["loan_amount"].values)
Y  = df["approved"].values

# ---------------- MODEL INIT ----------------
m1 = m2 = m3 = b = 0.0
lr = 0.1
epochs = 1500
n = len(Y)

loss_history = []

# ---------------- TRAINING ----------------
for epoch in range(epochs):
    z = m1*X1 + m2*X2 + m3*X3 + b
    y_pred = sigmoid(z)

    loss = log_loss(Y, y_pred)
    loss_history.append(loss)

    dm1 = (1/n) * np.sum(X1 * (y_pred - Y))
    dm2 = (1/n) * np.sum(X2 * (y_pred - Y))
    dm3 = (1/n) * np.sum(X3 * (y_pred - Y))
    db  = (1/n) * np.sum(y_pred - Y)

    m1 -= lr * dm1
    m2 -= lr * dm2
    m3 -= lr * dm3
    b  -= lr * db

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Log Loss {loss:.4f}")

# ---------------- LOSS CURVE ----------------
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Loan Approval Training")
plt.grid(True)
plt.show()

# ---------------- PREDICTION ----------------
z = m1*X1 + m2*X2 + m3*X3 + b
probabilities = sigmoid(z)

threshold = 0.6
predicted_class = (probabilities >= threshold).astype(int)

# ---------------- CONFUSION MATRIX ----------------
TP = TN = FP = FN = 0
for actual, pred in zip(Y, predicted_class):
    if actual==1 and pred==1: TP+=1
    elif actual==0 and pred==0: TN+=1
    elif actual==0 and pred==1: FP+=1
    elif actual==1 and pred==0: FN+=1

print("\nConfusion Matrix")
print(f"TN:{TN} FP:{FP}")
print(f"FN:{FN} TP:{TP}")

accuracy  = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall    = TP/(TP+FN)
fpr       = FP/(FP+TN)

print("\nMetrics")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("FPR      :", fpr)
