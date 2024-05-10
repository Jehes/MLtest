from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv('F:\scripts\main_additional\wine.csv')
y = df['statistics']
x = df.drop('statistics', axis=1)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=1000)
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

train_mean = mean_squared_error(y_train, y_lr_train_pred)
train_r2 = r2_score(y_train, y_lr_train_pred)

test_mean = mean_squared_error(y_test, y_lr_test_pred)
test_r2 = r2_score(y_test, y_lr_test_pred)
