from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

def get_train_test_data(input_data):
    input_data['y'] = input_data['死亡'].apply(lambda x: 1 if x >= 1 else 0)
    
    new_input_data = input_data.drop(columns=['死亡'], inplace=False)
    
    X = new_input_data.drop(columns=['y'])
    y = new_input_data['y']

    return X, y

def logistic_cm_gridsearch(X, y, random_state=42, n_jobs=12):
    # 数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # SMOTE+ENN 数据平衡处理
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

    # 对测试集进行 RUS 处理，平衡类别
    min_class_count = min(sum(y_test == 0), sum(y_test == 1))
    rus_test = RandomUnderSampler(sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=random_state)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    # 模型训练和超参数优化
    model = LogisticRegression(solver='saga', max_iter=10000, random_state=random_state)
    parameters = {
        'penalty': ['l2', 'l1'],  # 正则化方式
        'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度
    }
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
    grid_search.fit(X_resampled_train, y_resampled_train)
    best_model = grid_search.best_estimator_

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)

    # 对测试集预测概率
    y_proba = best_model.predict_proba(X_resampled_test)[:, 1]

    # 返回值调整为与 logistic_cm 一致
    return y_resampled_test, y_proba, np.arange(len(y_resampled_test))

def linear_svc_cm_gridsearch(X, y, random_state=42, n_jobs=12):
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # 資料平衡處理
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

    # 測試集重新平衡
    min_class_count = min(sum(y_test == 0), sum(y_test == 1))
    rus_test = RandomUnderSampler(sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=random_state)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    # 建立線性支持向量機模型
    model = LinearSVC(random_state=random_state, max_iter=500000)

    # 超參數範圍
    parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'loss': ['hinge', 'squared_hinge']
    }

    # 使用 GridSearchCV 找最佳參數
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
    grid_search.fit(X_resampled_train, y_resampled_train)
    best_model = grid_search.best_estimator_

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)

    # 預測測試集
    decision_scores = best_model.decision_function(X_resampled_test)

    # 返回與其他函数一致的输出格式
    return y_resampled_test, decision_scores, np.arange(len(y_resampled_test))

def xgboost_cm_gridsearch(X, y, random_state=42, n_jobs=12):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    smote = SMOTE(random_state=random_state, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

    min_class_count = min(sum(y_test == 0), sum(y_test == 1))
    rus_test = RandomUnderSampler(sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=random_state)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    model = XGBClassifier(eval_metric='logloss', random_state=random_state)
    parameters = {
        'n_estimators': [50, 100, 200, 400], # 樹的数量
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'colsample_bytree': [0.8, 1.0],  # 每棵樹使用的特征采样比例
    }
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
    grid_search.fit(X_resampled_train, y_resampled_train)
    best_model = grid_search.best_estimator_

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)

    y_proba = best_model.predict_proba(X_resampled_test)[:, 1]

    return y_resampled_test, y_proba, np.arange(len(y_resampled_test))

def logistic_cm_kfold(X, y, k=5, random_state=42, n_jobs=12):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    y_true_all = []
    y_proba_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Oversampling + undersampling
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        enn = EditedNearestNeighbours(n_neighbors=3)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
        X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

        # Model and GridSearch
        model = LogisticRegression(solver='saga', max_iter=10000, random_state=random_state)
        # 超參數範圍
        parameters = {
            # l2為Ridge，l1為Lasso，l1會將不重要的特徵權重變為0，l2不會
            'penalty': ['l2', 'l1'],
            # 大C傾向完全擬和，可能導致過擬和；小C傾向正則化，可能增加訓練誤差但增加泛化能力
            'C': [0.01, 0.1, 1, 10, 100]
        }
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X_resampled_train, y_resampled_train)
        best_model = grid_search.best_estimator_

        print(f"Best parameters for this fold: {grid_search.best_params_}")

        # Prediction and probability
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Store metrics and results
        y_true_all.extend(y_test)
        y_proba_all.extend(y_proba)
        original_indices.extend(test_index)

    return np.array(y_true_all), np.array(y_proba_all), np.array(original_indices)

def linear_svc_kfold(X, y, k=5, random_state=42, n_jobs=12):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # 存真實以及decision score方便後續對混淆的計算以及AUC
    y_true_all = []
    decision_scores_all = []
    # 這裡存儲的是原始資料的索引，方便後續將decision score直接對應到原始資料做拓樸
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 資料平衡處理
        smote = SMOTE(random_state=42, k_neighbors=3)
        enn = EditedNearestNeighbours(n_neighbors=3)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
        X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

        # 建立線性支持向量機模型
        model = LinearSVC(random_state=random_state, max_iter=500000)

        # 超參數範圍
        parameters = {
            # 大C傾向完全擬和，可能導致過擬和；小C傾向正則化，可能增加訓練誤差但增加泛化能力
            'C': [0.01, 0.1, 1, 10, 100],
            # 目標是最大化分類邊界的距離，hinge是線性SVM的損失函數，squared_hinge是hinge的平方
            'loss': ['hinge', 'squared_hinge']
        }

        # 使用 GridSearchCV 找最佳參數
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X_resampled_train, y_resampled_train)
        best_model = grid_search.best_estimator_

        print(f"Best parameters for this fold: {grid_search.best_params_}")

        # 預測測試集
        decision_scores = best_model.decision_function(X_test)

        # 存儲測試結果
        y_true_all.extend(y_test)
        decision_scores_all.extend(decision_scores)
        original_indices.extend(test_index)

    return np.array(y_true_all), np.array(decision_scores_all), np.array(original_indices)
