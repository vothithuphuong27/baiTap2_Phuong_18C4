import math
from collections import Counter
import numpy as np

class Point:
    value = []
    label = None
    distance = None

    def __init__(self, value, label):
        self.value = value
        self.label = label
    # tính toán khoảng cách sử dụng công thức euclid

    def euclidean_distance(self, test):
        acc = 0  # dùng để lưu trữ tổng
        for i in range(0, len(self.value)):
            acc += math.pow(test.value[i] - self.value[i], 2)  # tính tổng
        self.distance = math.sqrt(acc)  # lưu trữ giá trị khoảng cách

ex_2_dataset = np.array([
    Point([4, 3], 1),
    Point([3, 7], 1),
    Point([7, 4], 1),
    Point([4, 1], 1),
    Point([6, 5], 1),
    Point([5, 6], 1),
    Point([3, 7], 1),
    Point([6, 2], 1),
    Point([4, 6], 0),
    Point([4, 4], 0),
    Point([5, 8], 0),
    Point([7, 8], 0),
    Point([7, 6], 0),
    Point([4, 10], 0),
    Point([9, 7], 0),
    Point([5, 4], 0),
    Point([8, 5], 0),
    Point([6, 6], 0),
    Point([7, 4], 0),
    Point([8, 8], 0),
])

    # huấn luyện mô hình
def training(data, test):
    data_copy = data.copy()
    for point in data_copy:
        point.euclidean_distance(test)
    return data_copy

# tìm ra k láng giềng gần nhất
def get_k_nearest_neighbors(data, k):
    data_copy = data.copy()  # sao chép mảng tránh ghi đè
    # sắp xếp mảng tăng dần theo khoảng cách
    dt = data_copy.tolist()
    dt.sort(key=lambda x: x.distance)
    return np.array(dt[:k])  # trả về k phần tử đầu tiên của mảng

# tìm ra nhãn dự đoán
def get_predicted_label(data):
    labels = []  # mảng lưu trữ những nhãn xuất hiện trong dữ liệu
    for point in data:
        labels.append(point.label)  # thêm nhãn vào mảng
    # sử dụng Counter để đếm số lần xuất hiện của nhãn
    counted = Counter(labels)
    common = counted.most_common()  # trả vễ nhãn có số lần xuất hiện nhiều nhất
    if(common):  # kiểm tra sự tồn tại nhãn xuất hiện nhiều nhất tránh trường hợp mảng rỗng
        return common[0][0]  # trả vễ nhãn có số lần xuất hiện nhiều nhất
    return None  # nếu mảng rỗng trả về None

def main_ex_2_a():
    val = []
    # n = int(input("Nhập vào số lượng thuộc tính muốn kiểm tra: "))
    for i in range(0, 2):
        ele = float(input("Nhập vào thuộc tính " + str(i + 1) + ":"))
        val.append(ele)
    data = training(ex_2_dataset, Point(val, None))
    k = input("Nhập vào giá trị k: ")
    knn = get_k_nearest_neighbors(data, int(k))
    label = get_predicted_label(knn)
    print("Nhãn là:", label)

main_ex_2_a()
