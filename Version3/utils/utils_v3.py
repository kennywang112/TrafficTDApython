import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import mode, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

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

def split_death_injury(data):
    # Initialize lists to store death and injury counts
    deaths = []
    injuries = []
    
    # Loop over each item in the data
    for item in data:
        # Split the item by ';'
        parts = item.split(';')
        # For deaths, remove non-numeric characters and convert to integer
        deaths.append(int(''.join(filter(str.isdigit, parts[0]))))
        # For injuries, remove non-numeric characters and convert to integer
        injuries.append(int(''.join(filter(str.isdigit, parts[1]))))
    
    # Return a DataFrame with the extracted data
    return pd.DataFrame({'死亡': deaths, '受傷': injuries})

def process_age_speed(input_data):
    # 年齡分組
    bins_age = [0, 14, 24, 34, 44, 54, 64, 74, float('inf')]
    labels_age = ['未滿15歲', '15~24', '25~34', '35~44', '45~54', '55~64', '65~74', '75+']
    input_data['當事者事故發生時年齡'] = pd.cut(input_data['當事者事故發生時年齡'], bins=bins_age, labels=labels_age, right=False)

    bins_speed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, float('inf')]
    labels_speed = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '101-110', '110+']

    input_data['速限-第1當事者'] = pd.cut(input_data['速限-第1當事者'], bins=bins_speed, labels=labels_speed, right=False).astype(str)

    return input_data

# 定義函數，按月份進行下採樣
def downsample_by_month_simple(A1, A2, sampling_ratio, total_ratio, random_state=42):
    A1_downsampled = pd.DataFrame()
    A2_downsampled = pd.DataFrame()

    months = sorted(set(A1['發生月份']).intersection(A2['發生月份']))  # 確保月份匹配

    for month in months:
        # 提取該月份的資料
        A1_month = A1[A1['發生月份'] == month]
        A2_month = A2[A2['發生月份'] == month]

        # 計算該月份目標數量
        A1_target = int(len(A1_month) * sampling_ratio)
        A2_target = int(A1_target / total_ratio)
        print(A1_target, A2_target)

        # 下採樣
        A1_sampled = resample(A1_month, replace=False, n_samples=A1_target, random_state=random_state)
        A2_sampled = resample(A2_month, replace=False, n_samples=A2_target, random_state=random_state)

        # 合併到最終結果
        A1_downsampled = pd.concat([A1_downsampled, A1_sampled])
        A2_downsampled = pd.concat([A2_downsampled, A2_sampled])

    return A1_downsampled.reset_index(drop=True), A2_downsampled.reset_index(drop=True)

def get_unique_ids(input_data):
    unique_ids = set()
    for ids_list in input_data['ids']:
        unique_ids.update(ids_list)
    return list(unique_ids)