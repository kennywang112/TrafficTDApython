from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
import pandas as pd
import numpy as np

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def get_train_test_data(input_data):
    input_data['y'] = input_data['死亡'].apply(lambda x: 1 if x >= 1 else 0)

    new_input_data = input_data.drop(columns=['死亡'], inplace=False)

    X = new_input_data.drop(columns=['y'])
    y = new_input_data['y']

    return X, y


# ---------------------------------------------------------------------------
# 快速版工具
# ---------------------------------------------------------------------------
def _prepare_and_split(X, y, encoding, random_state, train_under_ratio=None, balance_test=False):
    """依 encoding 準備特徵並做時間切分（shuffle=False：前 70% 訓練、後 30% 測試）。

    encoding='dummy'  : 先對整組資料 get_dummies（與舊版行為一致；數值欄位會原樣通過，
                        因此若傳入的已是編碼後的數值 X 也相容）。
    encoding='onehot' : 保留原始類別欄位，OneHotEncoder 放進 pipeline，只 fit 訓練集，
                        並用 min_frequency 合併罕見類別以降維、handle_unknown 處理未見類別。

    train_under_ratio : 訓練集下採樣目標比例 = 少數類/多數類（RandomUnderSampler 的 sampling_strategy）。
                        例如 10:1（多數:少數）→ 0.1；None 表示不下採樣（用全部訓練資料）。
                        僅對「訓練集、切分之後」做，無資料洩漏；可大幅縮短訓練時間。
    balance_test      : False（預設）→ 測試集保留原始真實分布（建議：評估才誠實）。
                        True → 對測試集下採樣成 50/50（會高估 precision/F1，僅供對照）。
    回傳 (X_train, X_test, y_train, y_test)。
    """
    X_use = pd.get_dummies(X) if encoding == 'dummy' else X
    X_train, X_test, y_train, y_test = train_test_split(X_use, y, test_size=0.3, shuffle=False)

    # 訓練集下採樣（僅訓練集、切分後）——加速用
    if train_under_ratio is not None:
        y_arr = np.asarray(y_train)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        # 只有多數類確實多於 (1/ratio)×少數類時才下採樣，否則跳過以免 RandomUnderSampler 報錯
        if n_pos > 0 and n_neg > n_pos / train_under_ratio:
            rus_train = RandomUnderSampler(sampling_strategy=train_under_ratio, random_state=random_state)
            X_train, y_train = rus_train.fit_resample(X_train, y_train)

    if balance_test:
        min_class_count = min((y_test == 0).sum(), (y_test == 1).sum())
        rus_test = RandomUnderSampler(
            sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=random_state)
        X_test, y_test = rus_test.fit_resample(X_test, y_test)
    return X_train, X_test, y_train, y_test


def _wrap(clf, encoding):
    """onehot 模式在 clf 前加 OneHotEncoder；dummy 模式只有 clf。"""
    if encoding == 'onehot':
        ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=True)
        return SkPipeline([('oh', ohe), ('clf', clf)])
    return SkPipeline([('clf', clf)])


def logistic_cm_gridsearch(X, y, random_state=42, n_jobs=-1, encoding='dummy', n_iter=10, train_under_ratio=None, balance_test=False):
    X_train, X_test_bal, y_train, y_test_bal = _prepare_and_split(X, y, encoding, random_state, train_under_ratio, balance_test)

    # 以 class_weight='balanced' 取代 SMOTE：速度快很多、無重採樣洩漏疑慮
    clf = LogisticRegression(solver='saga', max_iter=5000,
                             class_weight='balanced', random_state=random_state)
    pipe = _wrap(clf, encoding)
    param_dist = {'clf__penalty': ['l1', 'l2'], 'clf__C': [0.01, 0.1, 1, 10, 100]}

    search = RandomizedSearchCV(pipe, param_dist, n_iter=min(n_iter, 10), cv=3,
                                scoring='accuracy', n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    y_proba = best_model.predict_proba(X_test_bal)[:, 1]
    return y_test_bal, y_proba, np.arange(len(y_test_bal))


def linear_svc_cm_gridsearch(X, y, random_state=42, n_jobs=-1, encoding='dummy', n_iter=10, train_under_ratio=None, balance_test=False):
    X_train, X_test_bal, y_train, y_test_bal = _prepare_and_split(X, y, encoding, random_state, train_under_ratio, balance_test)

    clf = LinearSVC(class_weight='balanced', max_iter=100000, random_state=random_state)
    pipe = _wrap(clf, encoding)
    param_dist = {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__loss': ['hinge', 'squared_hinge']}

    search = RandomizedSearchCV(pipe, param_dist, n_iter=min(n_iter, 10), cv=3,
                                scoring='accuracy', n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    decision_scores = best_model.decision_function(X_test_bal)
    return y_test_bal, decision_scores, np.arange(len(y_test_bal))


def xgboost_cm_gridsearch(X, y, random_state=42, n_jobs=-1, encoding='dummy', n_iter=20, train_under_ratio=None, balance_test=False):
    X_train, X_test_bal, y_train, y_test_bal = _prepare_and_split(X, y, encoding, random_state, train_under_ratio, balance_test)

    # scale_pos_weight 取代 SMOTE 處理不平衡；tree_method='hist' 大幅加速
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    spw = neg / pos

    clf = XGBClassifier(tree_method='hist', eval_metric='logloss',
                        scale_pos_weight=spw, n_jobs=1, random_state=random_state)
    pipe = _wrap(clf, encoding)
    param_dist = {
        'clf__n_estimators': [100, 200, 300, 400],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__colsample_bytree': [0.6, 0.8, 1.0],
        'clf__subsample': [0.8, 1.0],
    }

    search = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, cv=3,
                                scoring='accuracy', n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    y_proba = best_model.predict_proba(X_test_bal)[:, 1]
    return y_test_bal, y_proba, np.arange(len(y_test_bal))


# ---------------------------------------------------------------------------
# 5.4 DR baseline：無洩漏、時間切分、與 *_cm_gridsearch 一致的分類器設定
# ---------------------------------------------------------------------------
def _build_search_estimator(algo, y_train, random_state, n_jobs):
    """回傳 (estimator, param_dist, n_iter)，設定與新版 *_cm_gridsearch 完全一致。"""
    if algo == 'logistic':
        clf = LogisticRegression(solver='saga', max_iter=5000,
                                 class_weight='balanced', random_state=random_state)
        param = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        return clf, param, 10
    if algo == 'svc':
        clf = LinearSVC(class_weight='balanced', max_iter=100000, random_state=random_state)
        param = {'C': [0.01, 0.1, 1, 10, 100], 'loss': ['hinge', 'squared_hinge']}
        return clf, param, 10
    if algo == 'xgboost':
        pos = max(int((np.asarray(y_train) == 1).sum()), 1)
        neg = int((np.asarray(y_train) == 0).sum())
        clf = XGBClassifier(tree_method='hist', eval_metric='logloss',
                            scale_pos_weight=neg / pos, n_jobs=1, random_state=random_state)
        param = {'n_estimators': [100, 200, 300, 400], 'max_depth': [3, 5, 7, 9],
                 'learning_rate': [0.01, 0.05, 0.1, 0.2],
                 'colsample_bytree': [0.6, 0.8, 1.0], 'subsample': [0.8, 1.0]}
        return clf, param, 20
    raise ValueError(f"未知 algo: {algo}")


def run_dr_baseline(X_repr, y, dr, algo, n_components=10, random_state=42, n_jobs=-1,
                    train_under_ratio=None, balance_test=False):
    """5.4 用：對「單一全域資料」做降維 + 分類的 baseline，方法論與 Mapper 版一致。

    重點（跟舊版差異）：
      1) 時間切分 shuffle=False（X_repr 需已依時間排序）。
      2) 降維只 fit 在訓練集，再 transform 訓練/測試（無洩漏；舊版是對整份 fit）。
      3) 不平衡用 class_weight/scale_pos_weight（非 SMOTE）；RandomizedSearch + cv=3。
      4) 測試集以 RandomUnderSampler 平衡，與 *_cm_gridsearch 相同。
    train_under_ratio : 訓練集下採樣目標比例＝少數/多數（10:1 → 0.1；None 不下採樣）。
                        與 *_cm_gridsearch 一致；在「降維前」執行，故降維與分類器都在 10:1 訓練集上進行。
    dr: 'pca' / 'umap' 需傳 one-hot 數值矩陣；'mca' 需傳原始類別 DataFrame。
    回傳 (y_test_balanced, decision_scores, indices)。
    """
    from sklearn.preprocessing import MinMaxScaler
    y = np.asarray(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_repr, y, test_size=0.3, shuffle=False)

    # --- 訓練集下採樣（切分後、降維前）---
    if train_under_ratio is not None:
        y_arr = np.asarray(y_train)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        if n_pos > 0 and n_neg > n_pos / train_under_ratio:
            rus_train = RandomUnderSampler(sampling_strategy=train_under_ratio, random_state=random_state)
            X_train, y_train = rus_train.fit_resample(X_train, y_train)

    # --- 降維：只 fit 訓練集 ---
    if dr == 'pca':
        from sklearn.decomposition import PCA
        red = PCA(n_components=n_components, random_state=random_state)
        Z_train = red.fit_transform(np.asarray(X_train, dtype=float))
        Z_test = red.transform(np.asarray(X_test, dtype=float))
    elif dr == 'umap':
        import umap
        red = umap.UMAP(n_components=n_components, random_state=random_state)
        Z_train = red.fit_transform(np.asarray(X_train, dtype=float))
        Z_test = red.transform(np.asarray(X_test, dtype=float))
    elif dr == 'mca':
        import prince
        red = prince.MCA(n_components=n_components).fit(X_train)
        Z_train = red.transform(X_train).to_numpy()
        Z_test = red.transform(X_test).to_numpy()
    else:
        raise ValueError(f"未知 dr: {dr}")

    scaler = MinMaxScaler().fit(Z_train)
    Z_train = scaler.transform(Z_train)
    Z_test = scaler.transform(Z_test)

    # --- 測試集：預設保留真實分布；balance_test=True 才平衡 ---
    if balance_test:
        min_class_count = min((y_test == 0).sum(), (y_test == 1).sum())
        rus_test = RandomUnderSampler(
            sampling_strategy={0: min_class_count, 1: min_class_count}, random_state=random_state)
        Z_test_bal, y_test_bal = rus_test.fit_resample(Z_test, y_test)
    else:
        Z_test_bal, y_test_bal = Z_test, y_test

    # --- 分類器 + RandomizedSearch（cv=3）---
    clf, param, n_iter = _build_search_estimator(algo, y_train, random_state, n_jobs)
    search = RandomizedSearchCV(clf, param, n_iter=n_iter, cv=3, scoring='accuracy',
                                n_jobs=n_jobs, random_state=random_state)
    search.fit(Z_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    if algo == 'svc':
        scores = best_model.decision_function(Z_test_bal)
    else:
        scores = best_model.predict_proba(Z_test_bal)[:, 1]
    return y_test_bal, scores, np.arange(len(y_test_bal))


# ---------------------------------------------------------------------------
# Old：隨機 KFold 版本（非時間切分）。保留供 OriginModelSmaller 的 with-fold 段落使用。
# SMOTE 已放進 pipeline，交叉驗證每折各自重採樣，避免調參洩漏。
# ---------------------------------------------------------------------------
def logistic_cm_kfold(X, y, k=5, random_state=42, n_jobs=12):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    y_true_all = []
    y_proba_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        smote = SMOTE(random_state=random_state, k_neighbors=3)
        enn = EditedNearestNeighbours(n_neighbors=3)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)

        model = LogisticRegression(solver='saga', max_iter=10000, random_state=random_state)
        pipeline = ImbPipeline([('resample', smote_enn), ('clf', model)])
        parameters = {'clf__penalty': ['l2', 'l1'], 'clf__C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"Best parameters for this fold: {grid_search.best_params_}")

        y_proba = best_model.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test)
        y_proba_all.extend(y_proba)
        original_indices.extend(test_index)

    return np.array(y_true_all), np.array(y_proba_all), np.array(original_indices)


def linear_svc_kfold(X, y, k=5, random_state=42, n_jobs=12):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    y_true_all = []
    decision_scores_all = []
    original_indices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        smote = SMOTE(random_state=random_state, k_neighbors=3)
        enn = EditedNearestNeighbours(n_neighbors=3)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)

        model = LinearSVC(random_state=random_state, max_iter=500000)
        pipeline = ImbPipeline([('resample', smote_enn), ('clf', model)])
        parameters = {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__loss': ['hinge', 'squared_hinge']}
        grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"Best parameters for this fold: {grid_search.best_params_}")

        decision_scores = best_model.decision_function(X_test)

        y_true_all.extend(y_test)
        decision_scores_all.extend(decision_scores)
        original_indices.extend(test_index)

    return np.array(y_true_all), np.array(decision_scores_all), np.array(original_indices)
