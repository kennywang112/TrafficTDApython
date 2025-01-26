import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def average_encoded_label(data):
    # 轉換為 NumPy 陣列，保持物件型態
    data = np.array(data, dtype=object)
    
    # 移除 NaN 值
    clean_data = data[~pd.isnull(data)]
    
    # 如果沒有數據，返回 NaN
    if clean_data.size == 0:
        return np.nan

    # 判斷是否為字串型資料
    if all(isinstance(x, str) for x in clean_data):
        # 如果是字串型，使用 LabelEncoder
        le = LabelEncoder()
        encoded_labels = le.fit_transform(clean_data)
        # 計算標籤的平均值
        average_value = np.mean(encoded_labels)
    elif all(isinstance(x, (int, float)) for x in clean_data):
        # 如果是數值型，直接計算平均值
        average_value = np.mean(clean_data)
    else:
        raise ValueError("Data contains mixed types or unsupported types.")
    
    return average_value

def most_common_encoded_label(data):
    
    most_common_item = Counter(data).most_common(1)[0][0]
    
    return most_common_item

def sum_of_data(data):
    # 計算輸入資料的總和
    return np.nansum(data)

def rotate_z(points, theta):
    """
    以 Z 軸為旋轉軸，旋轉點雲。
    :param points: 點的列表或數組，形狀為 (n, 3)，每一行是 [x, y, z]
    :param theta: 旋轉角度（以弧度為單位）
    :return: 旋轉後的點的數組，形狀為 (n, 3)
    """
    # 定義 Z 軸旋轉矩陣
    theta = np.deg2rad(theta) # 將角度轉換為弧度
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
    ])
    
    # 點雲與旋轉矩陣相乘
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

def rotate_x(points, theta):
    """
    以 X 軸為旋轉軸，旋轉點雲。
    :param points: 點的列表或數組，形狀為 (n, 3)，每一行是 [x, y, z]
    :param theta: 旋轉角度（以弧度為單位）
    :return: 旋轉後的點的數組，形狀為 (n, 3)
    """
    # 定義 X 軸旋轉矩陣
    theta = np.deg2rad(theta) # 將角度轉換為弧度
    rotation_matrix = np.array([
        [1, 0,             0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    # 點雲與旋轉矩陣相乘
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

def rotate_y(points, theta):
    """
    以 Y 軸為旋轉軸，旋轉點雲。
    :param points: 點的列表或數組，形狀為 (n, 3)，每一行是 [x, y, z]
    :param theta: 旋轉角度（以弧度為單位）
    :return: 旋轉後的點的數組，形狀為 (n, 3)
    """
    # 定義 Y 軸旋轉矩陣
    theta = np.deg2rad(theta) # 將角度轉換為弧度
    rotation_matrix = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # 點雲與旋轉矩陣相乘
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

# Polygon 使用
def update_outliers(label_0, label_0_outliers, label_1, label_1_outliers, label_out):
    # 檢查 label_0_outliers 是否在 label_1 中有連接
    for idx, row in label_0_outliers.iterrows():
        connected = False
        for _, row1 in label_1.iterrows():
            if set(row["ids"]) & set(row1["ids"]): # 交集
                connected = True
                break
        if connected:
            # 從 label_0_outliers 移除，并添加到 label_0
            label_0_outliers = label_0_outliers.drop(idx)
            label_0 = pd.concat([label_0, pd.DataFrame([row])], ignore_index=True)

    # 檢查 label_1_outliers 是否在 label_0 中有連接
    for idx, row in label_1_outliers.iterrows():
        connected = False
        for _, row0 in label_0.iterrows():
            if set(row["ids"]) & set(row0["ids"]): # 交集
                connected = True
                break
        if connected:
            # 從 label_1_outliers 移除，并添加到 label_1
            label_1_outliers = label_1_outliers.drop(idx)
            label_1 = pd.concat([label_1, pd.DataFrame([row])], ignore_index=True)
            
    outliers = pd.concat([label_out, label_0_outliers, label_1_outliers], ignore_index=True)

    return label_0, label_1, outliers