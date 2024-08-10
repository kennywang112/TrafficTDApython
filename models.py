import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def svc_cm_with_grid_search(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)

    min_class_count = min(sum(y_test == 1), sum(y_test == 2))
    
    rus_test = RandomUnderSampler(sampling_strategy={1: min_class_count, 2: min_class_count}, random_state=42)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    param_grid = {
        'C': [1],
        'multi_class': ['ovr', 'crammer_singer']
    }

    grid_search = GridSearchCV(LinearSVC(penalty='l2', dual=True, fit_intercept=True, random_state=42, max_iter=100000), param_grid, cv=5, scoring='accuracy')
    
    grid_search.fit(X_resampled_train, y_resampled_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_resampled_test)

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

    return metrics_df, accuracy, conf_matrix

def logistic_cm_gridsearch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)
    
    min_class_count = min(sum(y_test == 1), sum(y_test == 2))
    
    rus_test = RandomUnderSampler(sampling_strategy={1: min_class_count, 2: min_class_count}, random_state=42)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)
    
    model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=10000)
    
    parameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_resampled_train, y_resampled_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_resampled_test)

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

    return metrics_df, accuracy, conf_matrix

def rf_with_gridsearch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=3)
    enn = EditedNearestNeighbours(n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn)
    X_resampled_train, y_resampled_train = smote_enn.fit_resample(X_train, y_train)
    
    min_class_count = min(sum(y_test == 1), sum(y_test == 2))
    
    rus_test = RandomUnderSampler(sampling_strategy={1: min_class_count, 2: min_class_count}, random_state=42)
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)
    
    scaler = StandardScaler()
    X_resampled_train_scaled = scaler.fit_transform(X_resampled_train)
    X_resampled_test_scaled = scaler.transform(X_resampled_test)

    model = RandomForestClassifier(random_state=43)
    
    param_grid = {
        'n_estimators': [300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_resampled_train_scaled, y_resampled_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_resampled_test_scaled)
    
    conf_matrix = confusion_matrix(y_resampled_test, y_pred)
    accuracy = accuracy_score(y_resampled_test, y_pred)

    precision = precision_score(y_resampled_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_resampled_test, y_pred, average=None)
    f1 = f1_score(y_resampled_test, y_pred, average=None)

    metrics = {
        'Label': [f'Class_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df, accuracy, conf_matrix
