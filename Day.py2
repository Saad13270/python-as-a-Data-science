#Problem Statement on Logistic regression::
"""In an e-commerce  company, the  management wants to 
predict a customer will purchase a high-value product
on their age, time spent on the website, and wheter
they have added item to their cart. The goal is to 
optimise marketing startegies by tarinng potential
customers more effective, thereby incrase sales
and revenue"""


import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = np.array([[25, 30, 0], [30, 40, 1], [20, 35, 0], [30, 45, 1]])
y = np.array([0, 1, 0, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(f"Model Accuracy {accuracy}")

user_age = float(input("Enter customer age :"))
user_time_spend = float(input("Enter time spend :"))
add_in_cart = float(input("Enter 1 if added to cart, 0 for not: "))

user_data = np.array([float(user_age), float(user_time_spend), add_in_cart], dtype=float).reshape(1, -1)
prediction = model.predict(user_data)

if prediction[0] == 1:
    print("The customer is likely to purchase")
else:
    print("The customer is unlikely to purchase")

