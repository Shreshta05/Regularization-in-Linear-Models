# Assignment 2: Regularization in Linear Models
# Student Marks Prediction

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

#create a dataset
data = {
    'study_hours': [
        2, 5, 1, 4, 6, 3, 5, 2, 7, 4,
        6, 1, 5, 3, 4, 2, 6, 7, 3, 5,
        4, 6, 2, 5, 1, 7, 3, 4, 6, 2
    ],
    'attendance_percent': [
        60, 85, 50, 75, 90, 70, 80, 65, 95, 78,
        88, 45, 82, 68, 76, 55, 89, 92, 72, 84,
        74, 87, 58, 81, 48, 94, 69, 77, 86, 62
    ],
    'internal_marks': [
        12, 18, 10, 15, 19, 14, 17, 13, 20, 16,
        18, 9, 17, 14, 16, 11, 19, 20, 15, 18,
        16, 19, 12, 17, 10, 20, 14, 16, 18, 13
    ],
    'practice_tests': [
        1, 4, 0, 3, 5, 2, 4, 1, 6, 3,
        5, 0, 4, 2, 3, 1, 5, 6, 2, 4,
        3, 5, 1, 4, 0, 6, 2, 3, 5, 1
    ],
    'final_marks': [
        45, 78, 40, 65, 88, 55, 75, 48, 92, 68,
        85, 38, 76, 54, 66, 42, 89, 94, 58, 80,
        64, 86, 44, 74, 39, 93, 56, 67, 84, 46
    ]
}

df = pd.DataFrame(data)

labels = ['study_hours', 'attendance_percent', 'internal_marks', 'practice_tests']

#separate feature(x) and target(y)
X = df.drop('final_marks', axis=1)
y = df['final_marks']

#split the data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create and train models with alphas(regularization strength)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.5),
    "Elastic Net": ElasticNet(alpha=0.5, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"{name:<18} "
          f"Train MSE: {mean_squared_error(y_train, train_pred):.2f} | "
          f"Test MSE: {mean_squared_error(y_test, test_pred):.2f}")

#plot train vs test error( i chose ridge)
alpha_values = [0.01, 0.1, 0.5, 1, 5]
train_errors = []
test_errors = []

for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(alpha_values, train_errors, marker='o', label='Train Error')
plt.plot(alpha_values, test_errors, marker='o', label='Test Error')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Train vs Test Error')
plt.legend()
plt.savefig('TrainVsTestErrorRidge.png', dpi = 150, bbox_inches = 'tight')
plt.show()

#coefficient shrinkage path - ridge
ridge_coefs = []

for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    ridge_coefs.append(model.coef_)

plt.figure(figsize=(8, 5))
for i in range(len(ridge_coefs[0])):
    plt.plot(alpha_values, [coef[i] for coef in ridge_coefs], marker='*', alpha= 0.7, label=labels[i])

plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression: Coefficient Shrinkage Path')
plt.axhline(0, linestyle='--', color='black')
plt.legend()
plt.savefig('CoeffShrinkagePathRidge.png', dpi = 150, bbox_inches = 'tight')
plt.show()

#coefficient shrinkage path - lasso
lasso_coefs = []

for alpha in alpha_values:
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    lasso_coefs.append(model.coef_)

plt.figure(figsize=(8, 5))
for i in range(len(lasso_coefs[0])):
    plt.plot(alpha_values, [coef[i] for coef in lasso_coefs], marker='*', label=labels[i])

plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression: Coefficient Shrinkage Path')
plt.axhline(0, linestyle='--', color='black')
plt.legend()
plt.savefig('CoeffShrinkagePathLasso.png', dpi = 150, bbox_inches = 'tight')
plt.show()

##coefficient shrinkage path - elasticNet
elastic_coefs = []

for alpha in alpha_values:
    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X_train, y_train)
    elastic_coefs.append(model.coef_)

plt.figure(figsize=(8, 5))
for i in range(len(elastic_coefs[0])):
    plt.plot(alpha_values, [coef[i] for coef in elastic_coefs], marker='*', label=labels[i])

plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net: Coefficient Shrinkage Path')
plt.axhline(0, linestyle='--', color='black')
plt.legend()
plt.savefig('CoeffShrinkagePathElasticNet.png', dpi = 150, bbox_inches = 'tight')
plt.show()