import pandas as pd
from scipy.stats import chi2_contingency

def chi_compare(c0, c1):
    dict_0 = {}
    dict_1 = {}
    
    for i in range(c1.shape[1] - 1):
        dict_0[c0.columns[i]] = c0.iloc[:, i].value_counts()

    for i in range(c1.shape[1] - 1): 
        dict_1[c1.columns[i]] = c1.iloc[:, i].value_counts()
        
    pvalue_lst = [] 
    for i in range(c1.shape[1] - 1):
        combined_df = pd.concat([dict_0[c0.columns[i]], dict_1[c1.columns[i]]], axis=1).fillna(0)
        combined_df.columns = ['cluster1', 'cluster2']
        # print(combined_df)

        # 將 DataFrame 轉換為列聯表
        contingency_table = combined_df.values
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        if p > 0.05:
            print(f"{c1.columns[i]} p值: {p} 不可分群")
            # continue
        else:
            print(f"{c1.columns[i]} p值: {p} 可分群")
            
    return pvalue_lst

def table(colnames, full_0, full_1, full_12):
    
    combined_df = pd.concat([full_0[colnames].value_counts(normalize = True), 
                             full_1[colnames].value_counts(normalize = True),
                             full_12[colnames].value_counts(normalize = True)
                            ],
                            axis=1).fillna(0)

    combined_df.columns = ['cluster1', 'cluster2', 'cluster12']
    
    return combined_df
    
def add_count(input_data, count):

    input_data['count'] = 0

    for key, value in count.items():
        input_data.loc[key, 'count'] = value

    return input_data

def get_count_dict(input_data):
    count = {}
    for key_to_search in input_data['node']:
        ids_to_search = input_data[input_data['node'] == key_to_search]['ids'].values[0]
        for num in ids_to_search:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
                
    return count