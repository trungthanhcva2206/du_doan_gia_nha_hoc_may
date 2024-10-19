# Dự đoán giá nhà bằng các phương pháp học máy
Dự đoán giá nhà dựa vào các model được xây dựng bằng các thuật toán Hồi Quy Tuyến Tính, Lasso, Neutral Network và kết hợp các thuật toán bằng kỹ thuật Stacking.

# Changelogs
## V1.0.0
- Mô hình dự đoán giá nhà bằng thuật toán hồi quy tuyến tính
- Mô hình dự đoán giá nhà bằng thuật toán lasso
- Mô hình dự đoán giá nhà bằng thuật toán Neutral NetWork
- Mô hình dự đoán giá nhà bằng cách kết hợp 3 thuật toán bằng kỹ thuật Stacking
# Link demo chương trình
- https://flask-app-deploy-hsqd.onrender.com
# Link trên Kaggle
- [Code Model Hồi Quy Tuyến Tính](https://www.kaggle.com/code/hunganh72/linear)
- [Code Model Lasso](https://www.kaggle.com/code/hunganh72/lasso)
- [Code Model Mạng Nơ-ron](https://www.kaggle.com/code/hunganh72/neural)
- [Code Model Kết hợp 3 Thuật Toán bằng Stacking](https://www.kaggle.com/code/hunganh72/stack-final)
# Setup 
- C1: Clone code từ git xuống
- C2: Có thể tải từ phần release
- Lưu ý: Nhớ chỉnh lại đường dẫn đọc file dữ liệu
```bash
data = pd.read_csv('D:\Machine_learning\kc_house_data (1).csv')
```
- Sử dụng pip để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt

