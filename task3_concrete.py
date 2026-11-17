
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------
# 1. Load dataset and remove duplicates
# --------------------------------

folder = Path(__file__).parent
file_path = folder / "concrete_strength.csv"

concrete = pd.read_csv(file_path)

concrete = concrete.drop_duplicates()

target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

print("CONCRETE STRENGTH DATASET")
print("-------------------------")
print("Total rows after removing duplicates:", len(concrete))
print("Number of input features:", X.shape[1])

# --------------------------------
# 2. Split data into training and testing
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------
# 3. Create five-fold cross-validation
# --------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# --------------------------------
# 4. Linear Regression baseline
# --------------------------------

linear_model = LinearRegression()

baseline_scores = cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

baseline_rmse = -baseline_scores
baseline_mean = np.mean(baseline_rmse)

print("\nLINEAR REGRESSION BASELINE")
print("-------------------------")

for i, score in enumerate(baseline_rmse, start=1):
    print(f"Fold {i} RMSE: {score:.4f}")

print(f"Mean Baseline CV RMSE: {baseline_mean:.4f} MPa")

# --------------------------------
# 5. Create Polynomial + Scaling + Ridge pipeline
# --------------------------------

pipeline = Pipeline([
    ("polynomial", PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# Parameters to test
parameters = {
    "polynomial__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

# --------------------------------
# 6. Tune model using GridSearchCV
# --------------------------------

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    return_train_score=False
)

search.fit(X_train, y_train)

# --------------------------------
# 7. Display three highest-ranked settings
# --------------------------------

results = pd.DataFrame(search.cv_results_)

results = results.sort_values("rank_test_score")

print("\nTOP THREE MODEL SETTINGS")
print("-------------------------")

top_three = results.head(3)

for index, row in top_three.iterrows():
    degree = row["param_polynomial__degree"]
    alpha = row["param_ridge__alpha"]
    rmse = -row["mean_test_score"]
    rank = row["rank_test_score"]

    print(f"\nRank: {rank}")
    print(f"Polynomial Degree: {degree}")
    print(f"Ridge Alpha: {alpha}")
    print(f"Mean CV RMSE: {rmse:.4f} MPa")

# --------------------------------
# 8. Evaluate the best model on test data
# --------------------------------

best_model = search.best_estimator_

predictions = best_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
test_mae = mean_absolute_error(y_test, predictions)

print("\nBEST MODEL")
print("-------------------------")
print("Best Parameters:", search.best_params_)
print(f"Best Mean CV RMSE: {-search.best_score_:.4f} MPa")

print("\nTEST RESULTS")
print("-------------------------")
print(f"Test RMSE: {test_rmse:.4f} MPa")
print(f"Test MAE: {test_mae:.4f} MPa")

# --------------------------------
# 9. Compare baseline and tuned model
# --------------------------------

print("\nMODEL COMPARISON")
print("-------------------------")
print(f"Baseline CV RMSE: {baseline_mean:.4f} MPa")
print(f"Tuned CV RMSE: {-search.best_score_:.4f} MPa")

if -search.best_score_ < baseline_mean:
    print("Tuned model performs better than the baseline.")
else:
    print("Baseline model performs better.")

# --------------------------------
# 10. Plot actual vs predicted values
# --------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.7, label="Predictions")

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction (y = x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = folder / "task3_concrete.png"
plt.savefig(plot_path, dpi=300)

plt.show()

print("\nPlot saved as task3_concrete.png")

# --------------------------------
# 11. Save results in notes file
# --------------------------------

notes_path = folder / "lab9_notes.txt"

with open(notes_path, "a", encoding="utf-8") as file:

    file.write("\n\nTASK 3: CONCRETE STRENGTH MODEL\n")
    file.write("--------------------------------\n")

    file.write(f"Rows after duplicate removal: {len(concrete)}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")

    file.write(f"Baseline CV RMSE: {baseline_mean:.4f} MPa\n")
    file.write(f"Tuned CV RMSE: {-search.best_score_:.4f} MPa\n")

    file.write(f"Best parameters: {search.best_params_}\n")
    file.write(f"Test RMSE: {test_rmse:.4f} MPa\n")
    file.write(f"Test MAE: {test_mae:.4f} MPa\n")

    file.write("\nTop three settings:\n")

    for index, row in top_three.iterrows():
        file.write(
            f"Degree: {row['param_polynomial__degree']}, "
            f"Alpha: {row['param_ridge__alpha']}, "
            f"Mean CV RMSE: {-row['mean_test_score']:.4f} MPa\n"
        )

    file.write(
        "\nDegree-2 interaction example: Cement multiplied by Water.\n"
    )

    file.write(
        "Scaling is inside the pipeline to prevent data leakage. "
        "Each CV training fold learns its own scaling values.\n"
    )

print("Task 3 results also saved in lab9_notes.txt")
