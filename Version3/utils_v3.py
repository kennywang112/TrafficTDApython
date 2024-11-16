import pandas as pd
from scipy.stats import chi2_contingency

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