import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Doc du lieu train
data = pd.read_csv('D:\Machine_learning\kc_house_data (1).csv')

# Chon features (Dung den ky thuat chon loc dac trung de chon)
features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']

# Split dataset sang X va Y
X = data[features]
y = data["price"]

#Split X,y -> X_train,y_train,X_valid,y_valid
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2, random_state=0) #random_state giup lay du lieu moi lan train giong nhau

#Training Model
from sklearn.linear_model import Lasso

reg_lasso = Lasso(alpha= 1.0)
reg_lasso.fit(X_train,y_train)



y_preds = reg_lasso.predict(X_valid)

print("Gia nha du doan: ",y_preds)
# DF= pd.DataFrame({ 'y': y_valid,'y_preds': y_preds})
# print(DF)

# Sai số huấn luyện trên tập train
print(f"R-square: {reg_lasso.score(X_train, y_train):.4f}" )

# Hệ số hồi qui và hệ số chặn
print("He so hoi quy: ", reg_lasso.coef_)
print("He so chan: ", reg_lasso.intercept_)

plt.figure(figsize=(10, 6))
plt.scatter(y_valid, y_preds, color='blue', label='Dự đoán')
plt.xlabel('Giá Trị Thực Tế')
plt.ylabel('Giá Trị Dự Đoán')
plt.title('Biểu đồ phân tán giữa giá trị thực tế và giá trị dự đoán')

plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', label='Dự đoán Hoàn Hảo')

plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dự đoán giá trị trên tập huấn luyện
y_train_preds = reg_lasso.predict(X_train)

# Dự đoán giá trị trên tập xác thực
y_valid_preds = reg_lasso.predict(X_valid)

# Tính các chỉ số hiệu suất trên tập huấn luyện
train_mse = mean_squared_error(y_train, y_train_preds)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_preds)
train_r2 = r2_score(y_train, y_train_preds)

# Tính các chỉ số hiệu suất trên tập xác thực
valid_mse = mean_squared_error(y_valid, y_valid_preds)
valid_rmse = np.sqrt(valid_mse)
valid_mae = mean_absolute_error(y_valid, y_valid_preds)
valid_r2 = r2_score(y_valid, y_valid_preds)

# In các chỉ số hiệu suất
print("Tập Huấn Luyện:")
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"R-squared (R²): {train_r2:.4f}")

print("\nTập Xác Thực:")
print(f"Mean Squared Error (MSE): {valid_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {valid_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {valid_mae:.2f}")
print(f"R-squared (R²): {valid_r2:.4f}")

# So sánh hiệu suất
print("\nSo sánh:")
print(f"Sự khác biệt MSE (Huấn Luyện - Xác Thực): {train_mse - valid_mse:.2f}")
print(f"Sự khác biệt RMSE (Huấn Luyện - Xác Thực): {train_rmse - valid_rmse:.2f}")
print(f"Sự khác biệt MAE (Huấn Luyện - Xác Thực): {train_mae - valid_mae:.2f}")
print(f"Sự khác biệt R² (Huấn Luyện - Xác Thực): {train_r2 - valid_r2:.4f}")

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
predicted_price = reg_lasso.predict(new_data)

# Hiển thị kết quả dự đoán
print(f"Dự đoán giá của ngôi nhà mới là: ${predicted_price[0]:,.2f}")