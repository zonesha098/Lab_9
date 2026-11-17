
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
homes = pd.read_csv("real_estate.csv")

print("Total transactions:", len(homes))

# Step 2: Convert distance into units of 100 metres
homes["distance_100m"] = (
    homes["X3 distance to the nearest MRT station"] / 100
)

# Step 3: Select input and output
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Step 4: Create Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X, y)

# Step 6: Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 7: Predict price at 500 metres
new_distance = pd.DataFrame({
    "distance_100m": [5]
})

prediction = model.predict(new_distance)

print("Predicted price at 500 m:", prediction[0])

# Step 8: Sort distances for plotting
sorted_X = X.sort_values("distance_100m")

# Step 9: Plot observations and regression line
plt.figure(figsize=(9, 6))

plt.scatter(
    X["distance_100m"],
    y,
    alpha=0.6,
    label="Actual observations"
)

plt.plot(
    sorted_X["distance_100m"],
    model.predict(sorted_X),
    color="red",
    linewidth=2,
    label="Regression line"
)

plt.xlabel("Distance to nearest MRT station (100 metres)")
plt.ylabel("House price per unit area")
plt.title("Housing Prices vs Distance to MRT Station")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 10: Save graph
plt.savefig("task1_housing.png", dpi=300)

plt.show()
