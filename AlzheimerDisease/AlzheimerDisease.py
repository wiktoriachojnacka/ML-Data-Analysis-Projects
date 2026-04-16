import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# Load data
data = pd.read_csv(
    './data/alzheimer.csv',
    sep=';',
    decimal=',',
    skipinitialspace=True,
    na_values=['', 'NA', 'NaN']
)

print('********** DATA QUALITY CHECK *********')

# 1. Data types
print("1. Data types:")
print(data.info())
print("Conclusion: All columns are numeric")

# 2. Missing values
print("2. Missing values:")
print(data.isna().sum())
print("Conclusion: No missing values in the dataset")

# 3. Value ranges
print("3. Value ranges (min/max):")
print(data.describe())
for col in data.columns:
    print(col, data[col].min(), data[col].max())
print("Conclusion: Age 60-90, BMI 15.01-39.99, AlcoholConsumption 0.0-20.0, "
      "PhysicalActivity 0h-10h, CholesterolTotal 150.09-299.9")

# 4. Duplicates
print("4. Number of duplicates:", data.duplicated().sum())
duplicates = data[data.duplicated()]
print(duplicates)
print("Conclusion: No duplicates in the dataset")

# 5. Value distributions - histograms
print("5. Value distribution analysis - histograms:")
data.hist(figsize=(12, 12))
plt.show()
print("Conclusion: Most patients are around 90 years old. Gender distribution is "
      "roughly balanced. Most respondents are non-smokers (~1500), no family history "
      "of Alzheimer's (~1600), no memory complaints (~1700), no behavioral problems "
      "(~1900), no difficulty completing tasks (~1900). Diagnosis: ~1400 negative, "
      "~800 positive. BMI, AlcoholConsumption, PhysicalActivity, CholesterolTotal "
      "and ADL show similar uniform distributions.")

# 6. Correlation heatmap
print("6. Correlation heatmap:")
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# 7. Outlier check
print("7. Outlier check:")
print(data.describe())

print("\nFeature selection conclusion: Selected predictors: Age, MemoryComplaints, "
      "BehavioralProblems, ADL (notable correlation with Diagnosis), plus "
      "FamilyHistoryAlzheimers and DifficultyCompletingTasks (low statistical "
      "correlation but strong medical justification - genetic factors and daily "
      "task difficulties significantly increase Alzheimer's risk). Excluded: "
      "Gender, BMI, Smoking, AlcoholConsumption, PhysicalActivity, CholesterolTotal, "
      "Forgetfulness - no significant relationship with Diagnosis.")

print("\n******************** DATA PREPARATION FOR MODELING **********************")

# Set seed as the integer mean of dataset indices
seed = int(np.mean(data.index))
print("seed:", seed)

# Train/test split
train, test = train_test_split(
    data,
    test_size=0.3,
    random_state=seed
)
print("train size:", len(train), "test size:", len(test))

print('\n************************* FEATURE PREPARATION *******************************')

features = ["Age", "MemoryComplaints", "BehavioralProblems", "ADL",
            "FamilyHistoryAlzheimers", "DifficultyCompletingTasks"]
X_train = train[features]
y_train = train["Diagnosis"]

X_test = test[features]
y_test = test["Diagnosis"]

print('\n************************* MLP MODEL *******************************')

# Standardize features (fit on train, transform on test to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation='relu',
    max_iter=500,
    random_state=seed
)
mlp.fit(X_train_scaled, y_train)

print('\n************************* RANDOM FOREST MODEL *******************************')

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=seed,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print('\n************************* MODEL EVALUATION *******************************')

# Predictions
y_train_pred_mlp = mlp.predict(X_train_scaled)
y_test_pred_mlp = mlp.predict(X_test_scaled)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)


def metrics(model, y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = f1_score(y_true, y_pred)

    print("\nModel:", name)
    print("Confusion matrix:\n", cm)
    print("Accuracy: %.3f" % accuracy)
    print("Specificity: %.3f" % specificity)
    print("Sensitivity (recall): %.3f" % sensitivity)
    print("F1-score: %.3f" % f1)
    return accuracy, specificity, sensitivity, f1


print("MLP metrics")
metrics(mlp, y_train, y_train_pred_mlp, "MLP - train")
metrics(mlp, y_test, y_test_pred_mlp, "MLP - test")

print("\nRandom Forest metrics")
metrics(rf, y_train, y_train_pred_rf, "Random Forest - train")
metrics(rf, y_test, y_test_pred_rf, "Random Forest - test")

print("\n************************* ROC CURVES AND AUC *******************************")

# MLP probabilities for class 1
y_test_proba_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_test_proba_mlp)
auc_mlp = auc(fpr_mlp, tpr_mlp)

# Random Forest probabilities for class 1
y_test_proba_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {auc_mlp:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()


print("\nFinal conclusions: The MLP model achieved AUC = 0.776, effectively "
      "distinguishing Alzheimer's patients from healthy individuals. Random Forest "
      "reached AUC = 0.748, performing slightly worse than MLP but still showing "
      "meaningful predictive power. Both ROC curves lie clearly above the diagonal "
      "(random guessing baseline), confirming that the models learned real patterns "
      "from the data rather than producing random predictions.")