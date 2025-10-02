import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("50_Startups.csv")

print("Problem 2: Profit Prediction")
print("Variables:", df.columns.tolist())
print("Correlation matrix:")
print(df.corr(numeric_only=True))

y = df["Profit"]
X = df.drop(columns=["Profit"])
X = pd.get_dummies(X, columns=["State"], drop_first=True)

chosen = ["R&D Spend","Marketing Spend"]
print("Chosen variables:", chosen)

for c in chosen:
    plt.scatter(df[c], y)
    plt.xlabel(c)
    plt.ylabel("Profit")
    plt.title(c + " vs Profit")
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X[chosen], y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

pred_tr = model.predict(X_train)
pred_te = model.predict(X_test)

print("Train RMSE:", round(mean_squared_error(y_train, pred_tr, squared=False),2))
print("Test RMSE:", round(mean_squared_error(y_test, pred_te, squared=False),2))
print("Train R2:", round(r2_score(y_train, pred_tr),3))
print("Test R2:", round(r2_score(y_test, pred_te),3))

"""
Here I loaded the 50_Startups dataset and checked the correlation. 
I noticed R&D Spend and Marketing Spend had the strongest relation with Profit. 
I plotted them and the graphs looked close to linear, so I used them as predictors. 
Then I split the data 80/20 and trained a linear regression model. 
Finally, I calculated RMSE and R2 to measure the performance. 
The model fits well and shows good R2 on both train and test.
"""
