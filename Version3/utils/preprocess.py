from sklearn.utils import resample
import pandas as pd

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

def preprocess(input_data, target, lst=False):
    
    if lst:
        select_lst = lst
    else:
        select_lst = [
            # 月份是為了篩選每個月2萬筆
            '發生月份',

            '天候名稱', '光線名稱', 
            '道路類別-第1當事者-名稱', '速限-第1當事者', 
            '路面狀況-路面鋪裝名稱', '路面狀況-路面狀態名稱', '路面狀況-路面缺陷名稱',
            '道路障礙-障礙物名稱', '道路障礙-視距品質名稱', '道路障礙-視距名稱',
            '號誌-號誌種類名稱', '號誌-號誌動作名稱',
            '車道劃分設施-分道設施-快車道或一般車道間名稱', '車道劃分設施-分道設施-快慢車道間名稱', '車道劃分設施-分道設施-路面邊線名稱',
            '當事者屬-性-別名稱', '當事者事故發生時年齡',
            '保護裝備名稱', '行動電話或電腦或其他相類功能裝置名稱',
            '肇事逃逸類別名稱-是否肇逃',
            '死亡受傷人數',

            # 大類別
            '道路型態大類別名稱', '事故位置大類別名稱',
            '車道劃分設施-分向設施大類別名稱',
            '事故類型及型態大類別名稱', '當事者區分-類別-大類別名稱-車種', '當事者行動狀態大類別名稱',
            '車輛撞擊部位大類別名稱-最初', '車輛撞擊部位大類別名稱-其他',

            # 兩個欄位只有兩個觀察值不同
            '肇因研判大類別名稱-主要',
            # '肇因研判大類別名稱-個別',
        ]
        
    # 篩選到第一個順位，因為注重的是單次事故的情況
    main_data = input_data[input_data['當事者順位'] == 1].reset_index(drop=True, inplace=False)
    sample_data = main_data[main_data['發生月份'] < 11]
    selected_data = sample_data[select_lst]
    
    # 將資料分出死亡和受傷，合併到原本的資料後去除多餘的死亡受傷人數
    split_death_injury_data = split_death_injury(selected_data['死亡受傷人數'])
    full_data = pd.concat([selected_data, split_death_injury_data], axis=1)
    # 補齊缺失值
    full_data[select_lst] = full_data[select_lst].fillna('未紀錄')
    # 速限範圍
    full_data = full_data[(full_data['速限-第1當事者'] < 200) &
                      (full_data['當事者事故發生時年齡'] < 100) &
                      (full_data['當事者事故發生時年齡'] > 0)]
    full_data.drop(columns=['死亡受傷人數'], inplace=True)

    if target=='駕駛' or target=='汽車' or target=='機車':

        if target == '駕駛':
            # 排除當事者類別為 '人'
            full_data = full_data[full_data['當事者區分-類別-大類別名稱-車種'] != '人']

        elif target == '汽車':
            # 排除當事者行動狀態為 '人', '機車', 或 '慢車'
            full_data = full_data[~full_data['當事者區分-類別-大類別名稱-車種'].isin(['人', '機車', '慢車'])]

        elif target == '機車':
            # 篩選當事者行動狀態為 '機車' 或 '慢車'
            full_data = full_data[full_data['當事者區分-類別-大類別名稱-車種'].isin(['機車', '慢車'])]
            # 嚴重影響MCA和Mapper表現
            full_data = full_data[full_data['道路型態大類別名稱'] != '平交道']

        # 篩選離群資料(影響MCA的因子得分)
        full_data = full_data[(full_data['肇因研判大類別名稱-主要'] != '非駕駛者') &
                    # 非車輛駕駛人因素和車的狀態相反，不適合分析，且數量很少
                    (full_data['肇因研判大類別名稱-主要'] != '無(非車輛駕駛人因素)') &
                    # 該資料的子類別都是"尚未發現肇事因素"，因此對於Mapper分析無意義
                    (full_data['肇因研判大類別名稱-主要'] != '無(車輛駕駛者因素)') &
                    # 未紀錄的資料量很少，不適合分析
                    (full_data['行動電話或電腦或其他相類功能裝置名稱'] != '未紀錄') &
                    (full_data['車輛撞擊部位大類別名稱-最初'] != '未紀錄')]

    elif target=='行人':
        # 篩選行人的資料
        full_data = full_data[(full_data['當事者區分-類別-大類別名稱-車種'] == '人')]

        # 篩選離群資料(影響MCA的因子得分)
        full_data = full_data[(full_data['行動電話或電腦或其他相類功能裝置名稱'] != '未紀錄') &
                    (full_data['行動電話或電腦或其他相類功能裝置名稱'] != '不明')]

    elif target=='全部':
        pass
        
    return full_data

def process_other(A1, A2, downsample=False):
    
    if downsample:
        # 下採樣資料，專用在ForMatrix檔案
        sampling_ratio = downsample  # 下採樣比例，根據A1 和 A2 原始數據量比例調整
        total_ratio = len(A1) / len(A2) # 保留 A1/A2 的比例
        downsampled_A1, downsampled_A2 = downsample_by_month_simple(A1, A2, sampling_ratio, total_ratio)
        rbind_data = pd.concat([downsampled_A1, downsampled_A2], axis=0, ignore_index=True)
    else:
        rbind_data = pd.concat([A1, A2], axis=0, ignore_index=True)

    rbind_data.drop(columns=['發生月份'], inplace=True)
    
    # 處理年齡和速限
    rbind_data = process_age_speed(rbind_data)
    death = rbind_data['死亡']
    rbind_data.drop(['死亡', '受傷'], axis=1, inplace=True)
    
    # 唯一值處理
    columns_to_drop = []
    for column in rbind_data.columns:
        if rbind_data[column].nunique() == 1:  # 檢查唯一值數量是否等於 1
            columns_to_drop.append(column)
    print(columns_to_drop)
    rbind_data.drop(columns=columns_to_drop, inplace=True)
    
    # Dummy
    rbind_data["速限-第1當事者"] = rbind_data["速限-第1當事者"].astype(str)
    dummy_data = pd.get_dummies(rbind_data)
    print('dummy_data:', dummy_data.shape)
    mapper_numpy = dummy_data.to_numpy()
    
    return mapper_numpy, rbind_data, dummy_data, death