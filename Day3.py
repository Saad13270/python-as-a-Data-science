# problem On SVC::

""" A telecommunication company want to reduce customer churn by identification
customer at risk of leaving . They have historical data on customer 
behaviour and want to build a model which customer are most liekly to churn"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Age": [23, 12, 34, 56, 45, 76, 45, 80, 56],
    "monthly_charges": [23, 43, 45, 12, 34, 55, 67, 75, 80],
    "Churn": [0, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print(df)

x = df[["Age", "monthly_charges"]]  
y = df["Churn"]  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

svc_model = SVC(kernel="linear", C=1.0)
svc_model.fit(x_train, y_train)

y_predict = svc_model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
report = classification_report(y_test, y_predict)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

user_age = float(input("Enter user age: "))
user_monthly_charges = float(input("Enter user monthly charges: "))
user_input = np.array([[user_age, user_monthly_charges]])
prediction = svc_model.predict(user_input)

if prediction[0] == 1:
    print("The customer is more likely to leave.")
else:
    print("The customer is unlikely to leave.")

