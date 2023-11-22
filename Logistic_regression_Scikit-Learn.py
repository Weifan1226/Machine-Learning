import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
y = np.array([0, 0, 0, 1, 1, 1])

"""
基于X,y进行的训练,下面对X_test进行预测
按道理说应该有y_test但是我没设置新的所以就用y代替了。
"""

X_test = np.array([[0.1, 1.0], [2,3], [1.9, 3.2], [2, 2.2], [5, 6], [3, 4]])

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)

y_pred = lr_model.predict(X_test)  

print("Prediction on training set:", y_pred)

print("Accuracy on training set:", lr_model.score(X_test, y))