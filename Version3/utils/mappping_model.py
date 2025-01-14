import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE, EditedNearestNeighbours
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

def get_train_test_data(input_data, classify=True):
    
    if classify:
        y = input_data['死亡'].apply(lambda x: 1 if x > 0 else 0)
    else:
        y = input_data['死亡']
    new_input_data = input_data.drop(columns=['受傷', '死亡'], inplace=False)
    X = new_input_data

    return X, y

def ridge_cm_kfold(X, y, k=5, alpha=1.0, random_state=42):
    """
    使用 Ridge Regression 進行 K-Fold 交叉驗證。

    Args:
        X (DataFrame): 特徵矩陣。
        y (Series): 目標變量。
        k (int): K-Fold 的折數，默認為 5。
        alpha (float): Ridge 的正則化強度參數，默認為 1.0。

    Returns:
        np.array: 所有測試集的真實值。
        np.array: 所有測試集的預測值。
        np.array: 所有測試集的原始索引。
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Ridge regression model with specified alpha
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Store metrics and results
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        original_indices.extend(test_index)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return np.array(y_true_all), np.array(y_pred_all), np.array(original_indices)

def lasso_cm_kfold(X, y, k=5, alpha=1.0, random_state=42):
    """
    使用 Lasso Regression 進行 K-Fold 交叉驗證。

    Args:
        X (DataFrame): 特徵矩陣。
        y (Series): 目標變量。
        k (int): K-Fold 的折數，默認為 5。
        alpha (float): Lasso 的正則化強度參數，默認為 1.0。

    Returns:
        np.array: 所有測試集的真實值。
        np.array: 所有測試集的預測值。
        np.array: 所有測試集的原始索引。
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Lasso regression model with specified alpha
        model = Lasso(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Store metrics and results
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        original_indices.extend(test_index)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return np.array(y_true_all), np.array(y_pred_all), np.array(original_indices)

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
            'penalty': ['l2', 'l1'],
            'C': [0.01, 0.1, 1, 10]
        }
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=12)
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