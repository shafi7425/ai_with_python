import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

df = pd.read_csv("weight-height.csv")
X = df[['Height']].values
y = df['Weight'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

rmse = math.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Weight = {model.coef_[0]:.4f} * Height + {model.intercept_:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R² = {r2:.4f}")

x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure(figsize=(6, 4))
plt.scatter(X, y, s=8, alpha=0.3, label="Data")
plt.plot(x_line, y_line, color="red", linewidth=2, label="Linear fit")
plt.title("Weight vs. Height — Linear Regression")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.tight_layout()

plt.savefig("ex2_linear_fit.png", dpi=150)
plt.show()  
