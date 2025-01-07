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

def get_train_test_data(input_data):
    input_data['y'] = input_data['死亡'].apply(lambda x: 1 if x >= 1 else 0)
    
    new_input_data = input_data.drop(columns=['受傷', '死亡'], inplace=False)
    
    X = new_input_data.drop(columns=['y'])
    y = new_input_data['y']

    return X, y

def logistic_cm_gridsearch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    # print("Original train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

    smote = SMOTE(random_state=42, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

    # print("Resampled train class distribution:", dict(zip(*np.unique(y_resampled_train, return_counts=True))))

    min_class_count = min(sum(y_test == 0), sum(y_test == 1))
    rus_test = RandomUnderSampler(sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=42)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    # print("Resampled test class distribution:", dict(zip(*np.unique(y_resampled_test, return_counts=True))))

    model = LogisticRegression(solver='saga', max_iter=10000)
    parameters = {
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_resampled_train, y_resampled_train)
    best_model = grid_search.best_estimator_

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)
    
    # y_pred = best_model.predict(X_resampled_test)
    y_proba = best_model.predict_proba(X_resampled_test)[:, 1]


    return y_resampled_test, y_proba

def linear_svc_cm_gridsearch(X, y):
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    # print("Original train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

    # 資料平衡處理
    smote = SMOTE(random_state=42, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

    # print("Resampled train class distribution:", dict(zip(*np.unique(y_resampled_train, return_counts=True))))

    # 測試集重新平衡
    min_class_count = min(sum(y_test == 0), sum(y_test == 1))
    rus_test = RandomUnderSampler(sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=42)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    # print("Resampled test class distribution:", dict(zip(*np.unique(y_resampled_test, return_counts=True))))

    # 建立線性支持向量機模型
    model = LinearSVC(random_state=42, max_iter=100000)

    # 超參數範圍
    parameters = {
        'C': [0.1, 1, 10, 100],
        'loss': ['hinge', 'squared_hinge']
    }

    # 使用 GridSearchCV 找最佳參數
    grid_search = GridSearchCV(model, parameters, cv=10, scoring='accuracy')
    grid_search.fit(X_resampled_train, y_resampled_train)
    best_model = grid_search.best_estimator_

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)

    # 預測測試集
    # y_pred = best_model.predict(X_resampled_test)
    decision_scores = best_model.decision_function(X_resampled_test)
    print(decision_scores)

    return y_resampled_test, decision_scores

def logistic_cm_kfold(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    y_true_all = []
    y_proba_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Oversampling + undersampling
        smote = SMOTE(random_state=42, k_neighbors=3)
        enn = EditedNearestNeighbours(n_neighbors=3)
        smote_enn = SMOTEENN(smote=smote, enn=enn)
        X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

        # Model and GridSearch
        model = LogisticRegression(solver='saga', max_iter=10000)
        parameters = {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1]
        }
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
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

def linear_svc_kfold(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    y_true_all = []
    decision_scores_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 資料平衡處理
        smote = SMOTE(random_state=42, k_neighbors=3)
        enn = EditedNearestNeighbours(n_neighbors=3)
        smote_enn = SMOTEENN(smote=smote, enn=enn)
        X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

        # 建立線性支持向量機模型
        model = LinearSVC(random_state=42, max_iter=100000)

        # 超參數範圍
        parameters = {
            'C': [0.1, 1, 10, 100],
            'loss': ['hinge', 'squared_hinge']
        }

        # 使用 GridSearchCV 找最佳參數
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
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
