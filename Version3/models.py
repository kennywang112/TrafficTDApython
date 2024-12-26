from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

def logistic_cm_gridsearch(X, y, threshold=0.5):
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
    threshold = threshold
    y_pred = (y_proba >= threshold).astype(int)
    print(y_proba)

    conf_matrix = confusion_matrix(y_resampled_test, y_pred)
    accuracy = accuracy_score(y_resampled_test, y_pred)

    precision = precision_score(y_resampled_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_resampled_test, y_pred, average=None)
    f1 = f1_score(y_resampled_test, y_pred, average=None)

    metrics_df = pd.DataFrame({
        'Label': [f'Class_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    return metrics_df, accuracy, conf_matrix, y_resampled_test, y_proba

def linear_svc_cm_gridsearch(X, y, threshold=0.5):
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
    threshold = threshold  # 舉例，根據需要調整
    y_pred = (decision_scores >= threshold).astype(int)

    # 計算評估指標
    conf_matrix = confusion_matrix(y_resampled_test, y_pred)
    accuracy = accuracy_score(y_resampled_test, y_pred)

    precision = precision_score(y_resampled_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_resampled_test, y_pred, average=None)
    f1 = f1_score(y_resampled_test, y_pred, average=None)

    # 整理成表格
    metrics_df = pd.DataFrame({
        'Label': [f'Class_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    return metrics_df, accuracy, conf_matrix, y_resampled_test, decision_scores