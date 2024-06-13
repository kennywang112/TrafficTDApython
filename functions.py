import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

from scipy import stats

def preprocess(input_data, select_lst, sample = 592):
    sample_data = input_data[input_data['當事者順位'] == 1].reset_index(drop=True, inplace=False)#.sample(sample).reset_index(drop=True)
    dataA = sample_data[select_lst]
    
    death_injury_data = split_death_injury(dataA['死亡受傷人數'])
    dist_df = pd.concat([dataA, death_injury_data], axis=1)
    dist_df.drop(columns=['死亡受傷人數'], inplace=True)
    
    return dist_df, sample_data

def process_data(input_data):
    # Create a copy of the data to avoid modifying the original dataframe
    input_cp = input_data.copy()
    
    # Iterate over each column in the dataframe
    for column in input_cp.columns:
        # Check if the column is not numeric
        if not pd.api.types.is_numeric_dtype(input_cp[column]):
            # Convert the column to 'category' if it's not already
            input_cp[column] = input_cp[column].astype('category')
            # Convert categories to integers
            input_cp[column] = input_cp[column].cat.codes
        # Fill NA/NaN values with 0
        input_cp[column] = input_cp[column].fillna(0)
        
    return input_cp

def process_age(input_data):
    
    bins = [0, 18, 30, 45, 65, float('inf')]
    labels = [0, 1, 2, 3, 4]

    # 分段處理
    input_data['當事者事故發生時年齡'] = pd.cut(input_data['當事者事故發生時年齡'], bins=bins, labels=labels, right=False)

    return input_data

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

# 使用cluster內出現頻率最多的作為顏色
def most_frequent_nonan(data):
    data = np.array(data, dtype=float)  # 指定float，處裡NaN
    # 移除 NaN 值
    clean_data = data[~np.isnan(data)]
    
    # 如果沒有數據，返回NaN
    if clean_data.size == 0:
        return np.nan
    
    # 使用 Counter 来找出出现最多的元素
    counter = Counter(clean_data)
    most_frequent = counter.most_common(1)
    
    # 確保獲取元素
    if most_frequent:
        return most_frequent[0][0]  # 獲取元素而非整個元組
    else:
        return np.nan
    
# Multinomial logistic regression
def get_clusterN_logit(cluster_data):
    scaler = StandardScaler()
    
    c0_for_lm = process_data(cluster_data)
    c0_for_lm_X = pd.DataFrame(scaler.fit_transform(c0_for_lm), columns = c0_for_lm.columns)
    c0_for_lm_y = cluster_data['死亡']
    
    return c0_for_lm_X, c0_for_lm_y

def get_logit_data(cX, cY, select_lst_LR):
    X = cX[select_lst_LR]
    
    logit_reg = LogisticRegression(penalty='l2', C=1, solver='saga', multi_class='multinomial')
    logit_reg.fit(X, cY)

    coef = logit_reg.coef_[0]
    intercept = logit_reg.intercept_[0]

    # Calculate the standard errors of the coefficients
    # This requires the design matrix (X), the coefficients (coef), and the number of observations (n)
    n = X.shape[0]
    p = X.shape[1]

    # Calculate the predicted probabilities
    pred_probs = logit_reg.predict_proba(X)

    # Construct the diagonal weight matrix for each observation
    W = np.diagflat(pred_probs[:, 1] * (1 - pred_probs[:, 1]))

    # Calculate the matrix product X.T @ W @ X
    matrix_product = X.T @ W @ X

    # Check the rank of the matrix to see if it's full rank
    matrix_rank = np.linalg.matrix_rank(matrix_product)
    full_rank = matrix_product.shape[0]

    # If the matrix is not full rank, apply a regularization term to make it invertible
    if matrix_rank < full_rank:
        reg_term = np.eye(full_rank) * 1e-5
        matrix_product += reg_term

    # Calculate the covariance matrix of the coefficients
    cov_matrix = np.linalg.inv(matrix_product)

    # The standard errors of the coefficients are the square roots of the diagonal elements
    se = np.sqrt(np.diag(cov_matrix))

    # Calculate the Wald statistics
    wald_stats = coef / se

    # Calculate the p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(wald_stats)))

    # Create a DataFrame to display the coefficients and their p-values
    coef_df = pd.DataFrame({
        'coefficients': coef,
        'standard_error': se,
        'wald_statistics': wald_stats,
        'p_value': p_values
    }, index=X.columns)

    # Sort the DataFrame by p_value
    coef_df_sorted = coef_df.sort_values(by='p_value')
    
    return coef_df_sorted

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def get_calinski_from_db(input_data, eps): 
    X = input_data.iloc[:, 3:6]

    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    input_data['label'] = labels
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    if len(set(labels)) != 1:
        score = metrics.calinski_harabasz_score(X, labels)
        silhouette_score_value = silhouette_score(X, labels)
    else:
        score = -1
        silhouette_score_value = -1
        
    return score, input_data, db, labels, n_clusters_, silhouette_score_value, unique_labels, colors