import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

HouseDF = pd.read_csv('D:\Machine_learning\kc_house_data (1).csv')
HouseDF.head()
HouseDF.columns
X = HouseDF[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y = HouseDF['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
c = lm.intercept_
c
m = lm.coef_
m
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Giá Trị Thực Tế')
plt.ylabel('Giá Trị Dự Đoán')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Tính toán MSE và RMSE cho tập kiểm tra
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Tính toán MAE cho tập kiểm tra
mae = mean_absolute_error(y_test, predictions)

# Hiển thị các chỉ số lỗi
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

from sklearn.metrics import r2_score

# Dự đoán trên tập kiểm tra
predictions = lm.predict(X_test)

# Tính R-squared
r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2:.4f}')

new_data = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2],
    'sqft_living': [2500],
    'sqft_lot': [5000],
    'floors': [2],
    'waterfront': [0],
    'view': [1],
    'condition': [3],
    'grade': [7],
    'sqft_above': [1800],
    'sqft_basement': [700],
    'yr_built': [1995],
    'yr_renovated': [0],
    'zipcode': [98178],
    'lat': [47.5208],
    'long': [-122.233],
    'sqft_living15': [1690],
    'sqft_lot15': [7503]
})

# Dự đoán giá cho dữ liệu mới
predicted_price = lm.predict(new_data)

# Hiển thị kết quả dự đoán
print(f"Dự đoán giá của ngôi nhà mới là: ${predicted_price[0]:,.2f}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dự đoán trên tập huấn luyện
train_predictions = lm.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

# Dự đoán trên tập kiểm tra
test_predictions = lm.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Tập Huấn Luyện:")
print(f"MSE: {train_mse}, RMSE: {train_rmse}, MAE: {train_mae}, R²: {train_r2}")

print("\nTập Kiểm Tra:")
print(f"MSE: {test_mse}, RMSE: {test_rmse}, MAE: {test_mae}, R²: {test_r2}")