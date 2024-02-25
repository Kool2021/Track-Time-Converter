import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import datetime

data_path = 'times.csv'
time_data = pd.read_csv(data_path)
#print(time_data.head())
#sns.scatterplot(x = '800_Time', y = '1600_Time', data = time_data)

x = time_data[['800_Time']]
y = time_data[['1600_Time']]

linear = LinearRegression()
linear.fit(x, y)
print(linear.coef_)
print(linear.intercept_)
y_pred = linear.predict(x)
plt.plot(x, y_pred, color='red')
plt.scatter(x, y)
plt.xlabel('800 Time')
plt.ylabel('1600 Time')
plt.show()

time = int(input("Enter your 800 time: "))
expected_secs = time*linear.coef_[0][0] + linear.intercept_[0]
expected = str(datetime.timedelta(seconds=expected_secs))
print(f"Your 1600 time should be {expected}")
actual = int(input("What is your actual 1600 time? "))
if actual > expected_secs:
    print(f"Your score is {(actual-expected_secs)**2}, you should be a middle distance runner")
else:
    print(f"Your score is {(expected_secs-actual)**2 * -1}, you should be a distance runner")

""" x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
plt.plot(x, y)
plt.show()

print(x)
print(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}") """

