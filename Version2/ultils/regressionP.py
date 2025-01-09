import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from functions import process_data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Multinomial logistic regression
def get_clusterN_logit(cluster_data, lst):
    scaler = StandardScaler()
    
    c0_for_lm = process_data(cluster_data)
    c0_for_lm_X = pd.DataFrame(scaler.fit_transform(c0_for_lm), columns=c0_for_lm.columns).reset_index(drop=True, inplace=False)
    c0_for_lm_y = cluster_data.apply(lambda row: 1 if row['死亡'] != 0 else 2, axis=1).reset_index(drop=True)
    c0_for_lm_X = c0_for_lm_X[lst]
    
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


"""
合併兩個比標籤好的資料，使用MLR後計算該分類是否顯著
"""

def pval(fullA, fullB, lst_regression) :  
    full_logit_X = pd.concat([fullA, fullB], axis=0)
    full_logit_X = get_clusterN_logit(full_logit_X, lst_regression)[0]
    full_logit_X.reset_index(drop=True, inplace=True)

    new_cluster0_y = pd.DataFrame([0] * fullA.shape[0], columns=['label'])
    new_cluster1_y = pd.DataFrame([1] * fullB.shape[0], columns=['label'])
    full_logit_y = pd.concat([new_cluster0_y, new_cluster1_y], axis=0)
    full_logit_y.reset_index(drop=True, inplace=True)

    p = get_logit_data(full_logit_X, full_logit_y['label'], lst_regression)

    p['feature'] = p.index
    
    return full_logit_X, full_logit_y, p

def calculate_proportions(full, category_column):
    # 計算受傷比例
    grouped1 = full.groupby([category_column, '受傷']).size().unstack(fill_value=0)
    total_count1 = grouped1.sum(axis=1)
    proportions1 = grouped1.div(total_count1, axis=0) * 100
    proportions1 = proportions1.round(2)  # 四捨五入到小數點後兩位
    proportions1.columns = [f'受傷{i}' for i in range(grouped1.shape[1])]  # 更新列名稱

    # 計算死亡比例
    grouped2 = full.groupby([category_column, '死亡']).size().unstack(fill_value=0)
    total_count2 = grouped2.sum(axis=1)
    proportions2 = grouped2.div(total_count2, axis=0) * 100
    proportions2 = proportions2.round(2)  # 四捨五入到小數點後兩位
    proportions2.columns = [f'死亡{i}' for i in range(grouped2.shape[1])]  # 更新列名稱

    # 合併兩個 DataFrame
    final_df = proportions1.join(proportions2)
    final_df['總數'] = total_count1
    # 重置索引以將 category_column 作為一個普通列
    final_df.reset_index(inplace=True)

    return final_df