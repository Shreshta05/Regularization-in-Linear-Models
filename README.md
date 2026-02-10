Regularization in linear models!

This repository contains an implementation of multiple linear regression models to predict students’ final marks and analyze the effect of regularization techniques. It compares Linear, Ridge, Lasso, and Elastic Net regression to understand how regularization reduces overfitting, controls model complexity, and improves generalization performance.

Conceptual Requirements!

Linear Regression : 

Q: What is the loss function used in Linear Regression?

Answer:
Linear Regression uses Mean Squared Error (MSE) as its loss function. MSE calculates the average of the squared differences between the actual values and the predicted values. Squaring the error ensures that positive and negative errors do not cancel each other out and that larger errors are penalized more heavily. The goal of the model during training is to minimize this loss.

Loss = 1/𝑛 * ∑(Y - y)^2

Here n = No.of data points, Y = Actual value, y = Predicted value, ∑ = sum of overall points

Q: What term is added to the loss?

Answer:
The loss consists only of the prediction error measured by MSE.

Q: What does that term mathematically encourage or discourage?

Answer:
Because there is no penalty term, the loss only encourages the model to reduce prediction error. It does not discourage large coefficient values, meaning the model is free to assign very high weights to features if that helps reduce training error.

Q: How does it affect model complexity and generalization?

Answer:
Since there is no control on coefficient size, the model can become complex and sensitive to noise in the training data. This often results in overfitting, where the model performs well on training data but generalizes poorly to unseen data.


Ridge Regression :

Q: What is the loss function used in Ridge Regression?

Answer:
Ridge Regression modifies the Linear Regression loss by adding an L2 regularization term, which penalizes large coefficients.

Loss = 1/𝑛 * ∑(Y− 𝑦)^2 + 𝛼 * (sum of squared weights)

Here n = No.of data points, Y = Actual value, y = Predicted value, ∑ = sum of overall points, 𝛼 = regularization strength 

Q: What term is added to the loss?

Answer:
An L2 penalty, which is the sum of the squared values of the model coefficients, is added to the loss.

Q: What does that term mathematically encourage or discourage?

Answer:
The L2 penalty encourages coefficients to remain small because squaring large coefficients increases the penalty significantly. It discourages very large weights but does not force coefficients to become zero, so all features remain part of the model.

Q: How does it affect model complexity and generalization?

Answer:
By shrinking coefficients, Ridge Regression reduces model complexity and prevents the model from relying too heavily on any single feature. This reduces overfitting and improves generalization, especially when all features are relevant.


Lasso Regression :

Q: What is the loss function used in Lasso Regression?

Answer:
Lasso Regression adds an L1 regularization term to the MSE loss.

Loss = 1/𝑛 * ∑(Y− 𝑦)^2 + 𝛼 * (sum of absolute weights)

Q: What term is added to the loss?

Answer:
An L1 penalty, which is the sum of the absolute values of the coefficients, is added.

Q: What does that term mathematically encourage or discourage?

Answer:
The L1 penalty encourages sparsity by strongly penalizing small coefficients. As a result, some coefficients are pushed exactly to zero, effectively discouraging less important features and removing them from the model.

Q: How does it affect model complexity and generalization?

Answer:
By removing unnecessary features, Lasso significantly reduces model complexity and improves interpretability. This can improve generalization, but if the regularization is too strong, it may remove important features and reduce performance.

Elastic Net Regression :

Q: What is the loss function used in Elastic Net?

Answer:
Elastic Net combines both L1 and L2 penalties with the MSE loss.

Loss = 1/𝑛 * ∑(Y− 𝑦)^2 + (λ) + [ 𝛼 (|B1| + |B2| + ----- + |Bn|) (1+𝛼) ] / 2 * (B1^2 + B2^2 + --- + Bn^2)

Q: What term is added to the loss?

Answer:
A combination of L1 (absolute value) and L2 (squared value) regularization terms is added.

Q: What does that term mathematically encourage or discourage?

Answer:
The L1 part encourages feature selection by pushing some coefficients to zero, while the L2 part encourages coefficient shrinkage and stability. Together, they discourage both unnecessary features and excessively large coefficients.

Q: How does it affect model complexity and generalization?

Answer:
Elastic Net controls model complexity in a balanced way. It avoids removing too many features like Lasso and avoids keeping all features without selection like Ridge. This makes it especially effective when features are correlated and leads to improved generalization.

Analysis & Hypothesis!

Q: Why does regularization improve test performance?

Answer: 
Regularization improves test performance because it controls model complexity. In standard linear regression, the model tries to minimize training error without any restriction on coefficient size. This can cause the model to fit noise in the training data, leading to overfitting. Regularization adds a penalty term to the loss function that discourages large coefficients. By limiting how much the model can rely on any single feature, regularization helps the model learn more general patterns. This is reflected in the results where regularized models show lower test MSE compared to plain linear regression, indicating better generalization.

Q: Why does Ridge keep all features but shrink them?

Answer:
Ridge regression uses an L2 penalty, which penalizes the square of the coefficients. Mathematically, shrinking a coefficient reduces the penalty smoothly, but setting it exactly to zero does not provide a special advantage. As a result, Ridge reduces the magnitude of all coefficients without removing any feature completely. This behavior is visible in the coefficient shrinkage path plots, where all coefficients move closer to zero as alpha increases but none disappear. Ridge is therefore suitable when all features are informative and should be retained with reduced influence.

Q: Why does Lasso remove features entirely?

Answer:
Lasso regression uses an L1 penalty, which penalizes the absolute value of coefficients. This penalty makes it mathematically favorable for some coefficients to become exactly zero. As alpha increases, less important features are completely removed from the model. This behavior is clearly shown in the Lasso coefficient shrinkage plot, where some coefficient lines reach zero and stay there. By removing features, Lasso reduces model complexity through feature selection, which can improve interpretability and sometimes generalization.

Q: Why does Elastic Net behave differently from both Ridge and Lasso?

Answer:
Elastic Net combines both L1 and L2 penalties, so it inherits properties from both Ridge and Lasso. The L1 part encourages feature selection, while the L2 part stabilizes coefficient shrinkage and prevents overly aggressive feature removal. This makes Elastic Net behave differently from Ridge, which keeps all features, and Lasso, which may remove too many features. In the Elastic Net shrinkage plot, coefficients shrink smoothly like Ridge, but some may approach or reach zero like Lasso. This balanced behavior is especially useful when features are correlated.

Q: Which model performed best on your dataset and why?

Answer:
Based on the numerical results and plots, Ridge Regression performed best on this dataset. It achieved the lowest test Mean Squared Error, indicating the best generalization performance. The Train vs Test MSE plot shows that Ridge effectively reduces overfitting by increasing bias slightly while significantly lowering variance. Since all features in the dataset contribute meaningful information, shrinking coefficients rather than removing features entirely resulted in better performance than Lasso or Elastic Net.

Numerical Results!

The performance of each model was evaluated using Mean Squared Error (MSE) on both training and testing datasets. The results are shown below:

| Model             | Train MSE | Test MSE |
| ----------------- | --------- | -------- |
| Linear Regression | 3.47      | 5.61     |
| Ridge Regression  | 4.15      | 2.93     |
| Lasso Regression  | 3.82      | 5.14     |
| Elastic Net       | 6.74      | 3.74     |

Interpretation of Results!

Linear Regression shows a lower training error but a higher test error, indicating overfitting. Ridge Regression achieves the lowest test MSE, demonstrating the best generalization performance. Lasso Regression removes some features, which increases test error in this dataset. Elastic Net provides a balance between Ridge and Lasso but does not outperform Ridge. These results confirm that coefficient shrinkage without feature elimination is most effective for this dataset.
