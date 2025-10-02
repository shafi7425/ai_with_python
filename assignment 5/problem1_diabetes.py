import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base = ["bmi","s5"]
m0 = LinearRegression().fit(X_train[base], y_train)
r2_base = r2_score(y_test, m0.predict(X_test[base]))

best_var = None
best_r2 = -999
for v in X.columns:
    if v not in base:
        m = LinearRegression().fit(X_train[base+[v]], y_train)
        r2 = r2_score(y_test, m.predict(X_test[base+[v]]))
        if r2 > best_r2:
            best_r2 = r2
            best_var = v

m1 = LinearRegression().fit(X_train[base+[best_var]], y_train)
r2_one = r2_score(y_test, m1.predict(X_test[base+[best_var]]))

m_all = LinearRegression().fit(X_train, y_train)
r2_all = r2_score(y_test, m_all.predict(X_test))

print("Problem 1: Diabetes")
print("Base R2:", round(r2_base,3))
print("Next variable added:", best_var, "R2:", round(r2_one,3))
print("All variables R2:", round(r2_all,3))

"""
How I Solve?
I first tested the model using only bmi and s5. 
Then I checked which other feature improved the R2 the most when added. 
The best variable was chosen this way and the model performed a bit better. 
Finally, I tested with all the variables and saw R2 improved more, but not drastically. 
So more variables help but the gain is smaller after a point.
"""
