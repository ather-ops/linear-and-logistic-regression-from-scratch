import numpy as np
import matplotlib.pyplot as plt

# Example predicted probabilities
probabilities = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95])
Y = np.array([0, 0, 1, 1, 1, 1])

# ROC calculation
thresholds = np.linspace(0, 1, 100)
tpr_list = []
fpr_list = []

for t in thresholds:
    preds = (probabilities >= t).astype(int)

    TP = np.sum((preds == 1) & (Y == 1))
    TN = np.sum((preds == 0) & (Y == 0))
    FP = np.sum((preds == 1) & (Y == 0))
    FN = np.sum((preds == 0) & (Y == 1))

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    tpr_list.append(TPR)
    fpr_list.append(FPR)

# Plot ROC
plt.figure(figsize=(6,5))
plt.plot(fpr_list, tpr_list, label="ROC Curve", color="blue")
plt.plot([0,1],[0,1],'--',color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
