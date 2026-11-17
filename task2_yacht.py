
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

# Load the Yacht Hydrodynamics dataset
folder = Path(__file__).parent
file_path = folder / "yacht_hydrodynamics.data"

data = pd.read_csv(file_path, sep=r"\s+", header=None)

# Input: Froude number (column 6)
# Output: Residuary resistance (column 7)
X = data[[5]]
y = data[6]

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Linear Regression
linear_model = LinearRegression()

linear_scores = cross_val_score(
    linear_model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

linear_rmse = -linear_scores

# Model 2: Quadratic Regression
quadratic_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("regression", LinearRegression())
])

quadratic_scores = cross_val_score(
    quadratic_model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

quadratic_rmse = -quadratic_scores

# Display results
print("YACHT HYDRODYNAMICS REGRESSION")
print("--------------------------------")

print("\nLinear Regression RMSE:")
for i, score in enumerate(linear_rmse, start=1):
    print(f"Fold {i}: {score:.4f}")

print(f"Mean Linear RMSE: {np.mean(linear_rmse):.4f}")

print("\nQuadratic Regression RMSE:")
for i, score in enumerate(quadratic_rmse, start=1):
    print(f"Fold {i}: {score:.4f}")

print(f"Mean Quadratic RMSE: {np.mean(quadratic_rmse):.4f}")

# Compare both models
print("\nMODEL COMPARISON")
print("--------------------------------")

if np.mean(quadratic_rmse) < np.mean(linear_rmse):
    print("Quadratic Regression performs better.")
else:
    print("Linear Regression performs better.")

# Save results in notes file
notes_path = folder / "lab9_notes.txt"

with open(notes_path, "a", encoding="utf-8") as file:
    file.write("\n\nTASK 2: YACHT HYDRODYNAMICS\n")
    file.write("--------------------------------\n")

    file.write("Linear Regression RMSE:\n")
    for i, score in enumerate(linear_rmse, start=1):
        file.write(f"Fold {i}: {score:.4f}\n")

    file.write(f"Mean Linear RMSE: {np.mean(linear_rmse):.4f}\n")

    file.write("\nQuadratic Regression RMSE:\n")
    for i, score in enumerate(quadratic_rmse, start=1):
        file.write(f"Fold {i}: {score:.4f}\n")

    file.write(f"Mean Quadratic RMSE: {np.mean(quadratic_rmse):.4f}\n")

    if np.mean(quadratic_rmse) < np.mean(linear_rmse):
        file.write("\nConclusion: Quadratic Regression performs better.\n")
    else:
        file.write("\nConclusion: Linear Regression performs better.\n")

print("\nResults saved in lab9_notes.txt")
