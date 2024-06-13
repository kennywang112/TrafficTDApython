import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

def svc_cm_with_grid_search(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # Set up the parameter grid
    param_grid = {
        'C': [1],
        'multi_class': ['ovr', 'crammer_singer']
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(LinearSVC(penalty='l2', dual=True, fit_intercept=True, random_state=42, max_iter=100000), param_grid, cv=5, scoring='accuracy')
    
    # Fit the grid search to the data
    grid_search.fit(X_resampled, y_resampled)
    
    # Extract the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate confusion matrix and accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Prepare detailed performance metrics
    cm_df = pd.DataFrame(conf_matrix, index=[f'Actual_{i}' for i in range(conf_matrix.shape[0])], columns=[f'Predicted_{i}' for i in range(conf_matrix.shape[1])])
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    metrics = {
        'Label': [f'Class_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df, accuracy, conf_matrix

def logistic_cm_gridsearch(X, y):
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用RandomOverSampler來平衡資料
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # 建立邏輯回歸模型並使用GridSearchCV來找到最佳參數
    model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000)
    parameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10]
    }
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_resampled, y_resampled)

    # 使用最佳模型進行預測
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 生成並打印混淆矩陣和各項度量指標
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    metrics_df = pd.DataFrame({
        'Label': [f'Class_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    return metrics_df, accuracy, conf_matrix

def rf_with_gridsearch(X, y):
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handling class imbalance
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    # Scaling features
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Defining the model and parameters for grid search
    dt_model = RandomForestClassifier(random_state=43)
    param_grid = {
        'n_estimators': [300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }

    # Grid search
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_resampled_scaled, y_resampled)

    # Predicting labels for the test set
    y_pred = grid_search.predict(X_test_scaled)

    # Calculating the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Formatting the confusion matrix for display
    cm_df = pd.DataFrame(conf_matrix, index=[f'Actual_{i}' for i in range(conf_matrix.shape[0])], 
                         columns=[f'Predicted_{i}' for i in range(conf_matrix.shape[1])])
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    metrics = {
        'Label': [f'Class_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df, accuracy, conf_matrix