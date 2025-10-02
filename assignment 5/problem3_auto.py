import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

auto = pd.read_csv("Auto.csv", na_values=["?","NA"])
auto = auto.dropna()

y = auto["mpg"]
X = auto.drop(columns=["mpg","name","origin"])
X = X.apply(pd.to_numeric, errors="coerce").dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 3, 30)
ridge_scores = []
lasso_scores = []

ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))])

for a in alphas:
    ridge.set_params(model__alpha=a)
    ridge.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))
    lasso.set_params(model__alpha=a)
    lasso.fit(X_train, y_train)
    lasso_scores.append(lasso.score(X_test, y_test))

plt.semilogx(alphas, ridge_scores, label="Ridge")
plt.semilogx(alphas, lasso_scores, label="Lasso")
plt.xlabel("alpha")
plt.ylabel("R2 on test")
plt.legend()
plt.show()

best_ridge_alpha = alphas[np.argmax(ridge_scores)]
best_lasso_alpha = alphas[np.argmax(lasso_scores)]

print("Problem 3: Car mpg")
print("Best Ridge alpha:", round(best_ridge_alpha,4), "R2:", round(max(ridge_scores),3))
print("Best Lasso alpha:", round(best_lasso_alpha,4), "R2:", round(max(lasso_scores),3))

"""
For the Auto dataset, I set mpg as the target and used all numeric features except name and origin. 
I split the data 80/20 and tried Ridge and Lasso regression with different alpha values. 
I plotted R2 against alpha and picked the best alpha based on highest test R2. 
Ridge kept stable scores while Lasso dropped faster. 
This shows regularization strength affects the performance and the plot helps to find the best balance.
"""
