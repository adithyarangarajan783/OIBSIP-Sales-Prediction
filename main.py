import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Advertising.csv")
x = dataset.iloc[:,1:4]
y = dataset.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.8)
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print('mean_squared_error : ', mean_squared_error(Y_test, Y_pred))
print("mean_absolute_error: ", mean_absolute_error(Y_test,Y_pred))
