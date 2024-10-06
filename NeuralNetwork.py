import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

data = pd.read_csv('D:\Machine_learning\kc_house_data (1).csv')
features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']
target = 'price'

data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled[:5]

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42, learning_rate_init=0.01)
mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

print("Giá thực tế:", y_test[:5].values)
print("Giá dự đoán:", y_pred[:5])

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

plt.scatter(y_test, y_pred)
plt.xlabel('Giá Trị Thực Tế')
plt.ylabel('Giá Trị Dự Đoán')
plt.title('Biểu đồ phân tán giữa giá trị thực tế và giá trị dự đoán')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.show()

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
predicted_price = mlp.predict(new_data)

# Hiển thị kết quả dự đoán
print(f"Dự đoán giá của ngôi nhà mới là: ${predicted_price[0]:,.2f}")

# Dự đoán trên tập huấn luyện và tập kiểm tra
y_train_preds = mlp.predict(X_train_scaled)
y_test_preds = mlp.predict(X_test_scaled)

# Tính toán các chỉ số hiệu suất trên tập huấn luyện
train_mse = mean_squared_error(y_train, y_train_preds)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_preds)
train_r2 = r2_score(y_train, y_train_preds)

# Tính toán các chỉ số hiệu suất trên tập kiểm tra
test_mse = mean_squared_error(y_test, y_test_preds)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_preds)
test_r2 = r2_score(y_test, y_test_preds)

# In các chỉ số hiệu suất
print("Tập Huấn Luyện:")
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"R-squared (R²): {train_r2:.4f}")

print("\nTập Kiểm Tra:")
print(f"Mean Squared Error (MSE): {test_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"R-squared (R²): {test_r2:.4f}")

# So sánh hiệu suất
print("\nSo sánh:")
print(f"Sự khác biệt MSE (Huấn Luyện - Kiểm Tra): {train_mse - test_mse:.2f}")
print(f"Sự khác biệt RMSE (Huấn Luyện - Kiểm Tra): {train_rmse - test_rmse:.2f}")
print(f"Sự khác biệt MAE (Huấn Luyện - Kiểm Tra): {train_mae - test_mae:.2f}")
print(f"Sự khác biệt R² (Huấn Luyện - Kiểm Tra): {train_r2 - test_r2:.4f}")

