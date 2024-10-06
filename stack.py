import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('D:\Machine_learning\kc_house_data (1).csv')
features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = data[features]
y = data['price']

print(X)
print(y)
from sklearn.model_selection import train_test_split
X_base, X_meta, y_base, y_meta = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_base,y_base,train_size=0.8, random_state=0)

from sklearn.linear_model import Lasso

base_lasso = Lasso(alpha = 10000.0)
base_lasso.fit(X_train,y_train)
base_lasso_preds = base_lasso.predict(X_valid)
# Vẽ biểu đồ phân tán giữa y thực tế và y dự đoán
plt.scatter(y_valid, base_lasso_preds, color='blue', label='Giá trị dự đoán')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=2, label='Đường y = y_pred')

# Thêm tiêu đề và nhãn
plt.title("So sánh giá trị thực tế và dự đoán Lasso")
plt.xlabel("Giá trị thực tế (y)")
plt.ylabel("Giá trị dự đoán (y_pred)")
plt.legend()

# Hiển thị biểu đồ
plt.show()

# In kết quả R-square
print(f"R-square: {base_lasso.score(X_train, y_train):.4f}")

from sklearn.linear_model import LinearRegression
base_linear = LinearRegression()
base_linear.fit(X_train, y_train)
base_linear_preds = base_linear.predict(X_valid)
# Vẽ biểu đồ phân tán giữa y thực tế và y dự đoán
plt.scatter(y_valid, base_linear_preds, color='blue', label='Giá trị dự đoán')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=2, label='Đường y = y_preds')

# Thêm tiêu đề và nhãn
plt.title("So sánh giá trị thực tế và dự đoán Linear")
plt.xlabel("Giá trị thực tế (y)")
plt.ylabel("Giá trị dự đoán (y_pred)")
plt.legend()

# Hiển thị biểu đồ
plt.show()

# In kết quả R-square
print(f"R-square: {base_linear.score(X_train, y_train):.4f}")
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

base_MLP = MLPRegressor(hidden_layer_sizes=(100, 50),activation='relu',solver='adam',max_iter=1000,random_state=0)
base_MLP.fit(X_train,y_train)
base_MLP_preds = base_MLP.predict(X_valid)
# Vẽ biểu đồ phân tán giữa y thực tế và y dự đoán
plt.scatter(y_valid, base_MLP_preds, color='blue', label='Giá trị dự đoán')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=2, label='Đường y = y_preds')

# Thêm tiêu đề và nhãn
plt.title("So sánh giá trị thực tế và dự đoán MLPRegressor ")
plt.xlabel("Giá trị thực tế (y)")
plt.ylabel("Giá trị dự đoán (y_pred)")
plt.legend()

# Hiển thị biểu đồ
plt.show()

# In kết quả R-square
print(f"R-square: {base_MLP.score(X_train, y_train):.4f}")

valid_pred_lasso = base_lasso.predict(X_meta)
valid_pred_linear = base_linear.predict(X_meta)
valid_pred_MLP = base_MLP.predict(X_meta)


# Tao du lieu train cho mo hinh chinh(mo hinh meta)
stacked_predictions = np.column_stack((base_lasso_preds, base_linear_preds, base_MLP_preds)) #Tao ma tran bang ket hop du doan tu 3 mo hinh co so tren tap X_valid -> dung de huan luyen meta
stacked_valid_predictions = np.column_stack((valid_pred_lasso, valid_pred_linear, valid_pred_MLP)) #Tao ma tran tu cac du doan cua base tren tap X_meta

meta_model = Lasso(alpha = 50000.0)
meta_model.fit(stacked_predictions, y_valid) #Huan luyen mo hinh sao cho y du doan khop vs y thuc te nhat

meta_preds = meta_model.predict(stacked_valid_predictions)

# Vẽ biểu đồ phân tán giữa y thực tế và y dự đoán
plt.scatter(y_meta, meta_preds, color='blue', label='Giá trị dự đoán')
plt.plot([y_meta.min(), y_meta.max()], [y_meta.min(), y_meta.max()], 'k--', lw=2, label='Đường y = y_preds')

# Thêm tiêu đề và nhãn
plt.title("So sánh giá trị thực tế và dự đoán")
plt.xlabel("Giá trị thực tế (y)")
plt.ylabel("Giá trị dự đoán (y_pred)")
plt.legend()

# Hiển thị biểu đồ
plt.show()

# In kết quả R-square
print(f"R-square: {meta_model.score(stacked_predictions, y_valid):.4f}" )

DF= pd.DataFrame({ 'y': y_meta,'y_preds': meta_preds})
print(DF)
from sklearn.metrics import mean_squared_error

# Evaluate the model on the training set (X_valid)
train_preds = meta_model.predict(stacked_predictions)  # Predictions on training data
train_mse = mean_squared_error(y_valid, train_preds)
train_r2 = meta_model.score(stacked_predictions, y_valid)

# Evaluate the model on the test set (X_meta)
test_mse = mean_squared_error(y_meta, meta_preds)
test_r2 = meta_model.score(stacked_valid_predictions, y_meta)

# Print the results
print(f"Training set performance:")
print(f"R-squared (train): {train_r2:.4f}")
print(f"Mean Squared Error (train): {train_mse:.4f}\n")

print(f"Test set performance:")
print(f"R-squared (test): {test_r2:.4f}")
print(f"Mean Squared Error (test): {test_mse:.4f}\n")

# Check if overfitting occurs by comparing train vs test performance
if train_r2 - test_r2 > 0.1:
    print("The model might be overfitting (higher training performance compared to test performance).")
else:
    print("The model is not significantly overfitting.")
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
valid_pred_lasso = base_lasso.predict(new_data)
valid_pred_linear = base_linear.predict(new_data)
valid_pred_MLP = base_MLP.predict(new_data)

stacked_valid_predictions = np.column_stack((valid_pred_lasso, valid_pred_linear, valid_pred_MLP))
# Dự đoán giá cho dữ liệu mới
predicted_price = meta_model.predict(stacked_valid_predictions)

# Hiển thị kết quả dự đoán
print(f"Dự đoán giá của ngôi nhà mới là: ${predicted_price[0]:,.2f}")

