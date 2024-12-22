import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import mode, chi2_contingency
from sklearn.preprocessing import LabelEncoder

def compare_categorical_features(full_0, full_1):
    # 添加來源標籤以區分兩個資料表
    full_0['source'] = 'full_0'
    full_1['source'] = 'full_1'
    
    # 合併兩個資料表
    combined = pd.concat([full_0, full_1], ignore_index=True)
    
    # 初始化結果字典
    chi2_results = {}
    
    # 遍歷每個分類特徵
    for col in full_0.columns:
        if col == 'source':  # 跳過來源標籤
            continue
        
        # 建立列聯表
        contingency_table = pd.crosstab(combined[col], combined['source'])
        
        # 進行卡方檢定
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # 保存結果
        chi2_results[col] = {'chi2': chi2, 'p_value': p_value}
    
    # 將結果轉換為 DataFrame 以便排序
    result_df = pd.DataFrame.from_dict(chi2_results, orient='index')
    result_df = result_df.sort_values(by='p_value')  # 按 P 值排序，越小越顯著
    
    return result_df

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

def rotate_z(points, theta):
    """
    以 Z 軸為旋轉軸，旋轉點雲。
    :param points: 點的列表或數組，形狀為 (n, 3)，每一行是 [x, y, z]
    :param theta: 旋轉角度（以弧度為單位）
    :return: 旋轉後的點的數組，形狀為 (n, 3)
    """
    # 定義 Z 軸旋轉矩陣
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