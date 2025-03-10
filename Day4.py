# Problem on KNeighborsClassifier::
'''A retail company wants to predict customer purchasing behavior based 
on their age, salary, and past purchase history. The company aims to 
use K-Nearest Neighbors (KNN) algorithm to classify customers into 
potential buying groups to personalize marketing strategies. This 
predictive model will help the company understand and target specific
customer segments more effectively, thereby increasing sales and customer 
satisfaction.'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier
data=np.array([23,50000,1], [56,70000,2], [34,53000,1],
               [43,40000,1],[27,90000,3],[60,88000,3])
behaviour_of_customer=np.array([1,0,1,1,0,1])#0 low, 1 medium, 2 high
x_train, x_test, y_train, y_test=train_test_split(data, behaviour_of_customer, test_size=0.2, random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
accuracy=knn.score(x_test, y_test)
print(f"Accuracy of our model is {accuracy} :")
age = float(input("Enter your age: "))
salary = float(input("Enter your salary: "))
num_products = int(input("Enter the number of products you purchase: "))
user_data = (age, salary, num_products)
print(f"\nUser Data (Tuple): {user_data}")
user_input_sacled=scaler.transform(user_data)
knn.predict(user_input_sacled)



