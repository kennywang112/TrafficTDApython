from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd

def logistic_cm_gridsearch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    
    model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000)
    
    parameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Value': [precision, recall, f1]
    })

    return metrics_df, accuracy, conf_matrix

def random_forest_dummy_classifier(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43, stratify=y)
    
    
    rf = RandomForestClassifier(random_state=43, class_weight='balanced')

    parameters = {
        'n_estimators': [50, 100, 200],  # 樹的數量
        'max_depth': [None, 10, 20],     # 樹的深度
        'min_samples_split': [2, 5, 10], # 節點分裂的最小樣本數
        'min_samples_leaf': [1, 2, 4]    # 葉節點的最小樣本數
    }

    grid_search = GridSearchCV(rf, parameters, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    # 預測與評估
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # 生成指標 DataFrame
    metrics_df = pd.DataFrame(class_report).transpose()
    metrics_df = metrics_df[['precision', 'recall', 'f1-score']]

    # 返回結果
    return metrics_df, accuracy, conf_matrix, grid_search.best_params_