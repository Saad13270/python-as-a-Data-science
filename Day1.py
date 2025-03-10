# problem 1
# predict student final exam score based on the no of hour they study
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Data = {"Hour_study": [2, 3, 4, 5, 6, 7, 8, 9], 
        "Exam_score": [56, 59, 71, 80, 84, 89, 90, 95]}

df = pd.DataFrame(Data)
print(df)

x = df[["Hour_study"]]
y = df[["Exam_score"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

user_input = float(input("Enter the number of hours you study: "))
predict_model = model.predict([[user_input]])

print(f"Predicted exam score is {predict_model[0][0]:.2f}")

