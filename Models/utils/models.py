from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
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


def _prepare_train_validation_test(
        X, y, encoding, random_state, train_under_ratio=None,
        train_frac=0.7, validation_frac=0.1, balance_test=False):
    """依時間切成 train/validation/test，預設 70%/10%/20%。

    validation 保留真實類別比例，專門用來選 threshold；test 完全不參與
    preprocessing、模型 fitting、下採樣或 threshold selection。
    """
    if not (0 < train_frac < 1 and 0 < validation_frac < 1):
        raise ValueError('train_frac 與 validation_frac 必須介於 0 和 1')
    if train_frac + validation_frac >= 1:
        raise ValueError('train_frac + validation_frac 必須小於 1')

    n = len(X)
    n_train = int(np.floor(n * train_frac))
    n_validation = int(np.floor(n * validation_frac))
    validation_end = n_train + n_validation
    if n_train == 0 or n_validation == 0 or validation_end >= n:
        raise ValueError('資料量不足以建立 train/validation/test')

    X_train = X.iloc[:n_train]
    X_validation = X.iloc[n_train:validation_end]
    X_test = X.iloc[validation_end:]
    y_train = y.iloc[:n_train]
    y_validation = y.iloc[n_train:validation_end]
    y_test = y.iloc[validation_end:]

    # dummy vocabulary 也只由 train 決定，validation/test 僅對齊 train 欄位。
    if encoding == 'dummy':
        X_train = pd.get_dummies(X_train)
        X_validation = pd.get_dummies(X_validation).reindex(
            columns=X_train.columns, fill_value=0)
        X_test = pd.get_dummies(X_test).reindex(
            columns=X_train.columns, fill_value=0)

    # 只對真正用來 fit 模型的 train 下採樣；validation/test 保留真實死亡率。
    if train_under_ratio is not None:
        y_arr = np.asarray(y_train)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        if n_pos > 0 and n_neg > n_pos / train_under_ratio:
            rus_train = RandomUnderSampler(
                sampling_strategy=train_under_ratio, random_state=random_state)
            X_train, y_train = rus_train.fit_resample(X_train, y_train)

    if balance_test:
        min_class_count = min((y_test == 0).sum(), (y_test == 1).sum())
        rus_test = RandomUnderSampler(
            sampling_strategy={0: min_class_count, 1: min_class_count},
            random_state=random_state)
        X_test, y_test = rus_test.fit_resample(X_test, y_test)

    return (
        X_train, X_validation, X_test,
        y_train, y_validation, y_test,
    )


def select_threshold_at_target_recall(y_validation, validation_scores, target_recall=0.8):
    """在 validation 上：recall 至少達標時，選 precision 最高的 threshold。

    若 precision 並列，選 recall 最接近 target 的 threshold，避免不必要的警報。
    回傳的 threshold 只可套到 untouched test，不可再用 test label 調整。
    """
    yv = np.asarray(y_validation).astype(int)
    sv = np.asarray(validation_scores, dtype=float)
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall 必須介於 0（不含）和 1（含）')
    if np.unique(yv).size < 2:
        raise ValueError('validation 必須同時包含死亡與非死亡案例')

    precision, recall, thresholds = precision_recall_curve(yv, sv)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if eligible.size == 0:
        raise ValueError(f'validation 上無法達到 target_recall={target_recall}')

    eligible_precision = precision[:-1][eligible]
    best_precision = np.nanmax(eligible_precision)
    tied = eligible[np.isclose(
        eligible_precision, best_precision, rtol=1e-12, atol=1e-15)]
    best_idx = tied[np.argmin(np.abs(recall[:-1][tied] - target_recall))]
    threshold = float(thresholds[best_idx])
    pred = (sv >= threshold).astype(int)

    return {
        'selected_threshold': threshold,
        'target_recall': float(target_recall),
        'validation_recall': float(recall_score(yv, pred, zero_division=0)),
        'validation_precision': float(precision_score(yv, pred, zero_division=0)),
        'validation_f1': float(f1_score(yv, pred, zero_division=0)),
        'n_validation': int(len(yv)),
        'n_validation_positive': int(yv.sum()),
        'n_validation_predicted_positive': int(pred.sum()),
    }


def select_threshold_max_f1(y_validation, validation_scores):
    """在 validation 上選擇 F1-score 最大的 threshold。

    每個模型只使用自己的 validation 分數選門檻；untouched test 不參與。
    若多個門檻的 F1 相同，先選 recall 較高者，再選其中較高的 threshold。
    """
    yv = np.asarray(y_validation).astype(int)
    sv = np.asarray(validation_scores, dtype=float)
    if np.unique(yv).size < 2:
        raise ValueError('validation 必須同時包含死亡與非死亡案例')
    if not np.isfinite(sv).all():
        raise ValueError('validation_scores 必須全部為有限數值')

    precision, recall, thresholds = precision_recall_curve(yv, sv)
    if thresholds.size == 0:
        raise ValueError('validation 無法產生可用的 threshold')

    p = precision[:-1]
    r = recall[:-1]
    denominator = p + r
    f1_values = np.divide(
        2 * p * r,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    best_f1 = np.nanmax(f1_values)
    tied = np.flatnonzero(np.isclose(
        f1_values, best_f1, rtol=1e-12, atol=1e-15))
    best_recall = np.nanmax(r[tied])
    tied = tied[np.isclose(r[tied], best_recall, rtol=1e-12, atol=1e-15)]
    best_idx = tied[-1]

    threshold = float(thresholds[best_idx])
    pred = (sv >= threshold).astype(int)
    return {
        'selected_threshold': threshold,
        'threshold_strategy': 'max_validation_f1',
        'validation_recall': float(recall_score(yv, pred, zero_division=0)),
        'validation_precision': float(precision_score(yv, pred, zero_division=0)),
        'validation_f1': float(f1_score(yv, pred, zero_division=0)),
        'n_validation': int(len(yv)),
        'n_validation_positive': int(yv.sum()),
        'n_validation_predicted_positive': int(pred.sum()),
    }


def select_threshold_youden(y_validation, validation_scores):
    """在 validation 上選擇最大化 Youden's J 的 threshold。

    J = sensitivity + specificity - 1 = TPR - FPR。若多個有限門檻的
    J 相同，先選 sensitivity 較高者；test label 完全不參與門檻選擇。
    """
    yv = np.asarray(y_validation).astype(int)
    sv = np.asarray(validation_scores, dtype=float)
    if np.unique(yv).size < 2:
        raise ValueError('validation 必須同時包含死亡與非死亡案例')
    if not np.isfinite(sv).all():
        raise ValueError('validation_scores 必須全部為有限數值')

    fpr, tpr, thresholds = roc_curve(
        yv, sv, drop_intermediate=False)
    finite = np.flatnonzero(np.isfinite(thresholds))
    if finite.size == 0:
        raise ValueError('validation 無法產生有限的 threshold')

    youden_values = tpr - fpr
    best_j = np.nanmax(youden_values[finite])
    tied = finite[np.isclose(
        youden_values[finite], best_j, rtol=1e-12, atol=1e-15)]
    best_sensitivity = np.nanmax(tpr[tied])
    tied = tied[np.isclose(
        tpr[tied], best_sensitivity, rtol=1e-12, atol=1e-15)]
    best_idx = tied[np.argmax(thresholds[tied])]

    threshold = float(thresholds[best_idx])
    pred = (sv >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        yv, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        'selected_threshold': threshold,
        'threshold_strategy': 'max_validation_youden_j',
        'validation_youden_j': float(sensitivity + specificity - 1),
        'validation_recall': float(sensitivity),
        'validation_specificity': float(specificity),
        'validation_precision': float(precision_score(
            yv, pred, zero_division=0)),
        'validation_f1': float(f1_score(yv, pred, zero_division=0)),
        'n_validation': int(len(yv)),
        'n_validation_positive': int(yv.sum()),
        'n_validation_predicted_positive': int(pred.sum()),
    }


def _select_validation_threshold(
        y_validation, validation_scores, threshold_strategy=None,
        target_recall=None):
    """選擇 validation threshold，同時保留舊 target_recall 呼叫的相容性。"""
    if threshold_strategy is not None and target_recall is not None:
        raise ValueError('threshold_strategy 與 target_recall 不可同時指定')
    if threshold_strategy == 'max_f1':
        return select_threshold_max_f1(y_validation, validation_scores)
    if threshold_strategy == 'youden':
        return select_threshold_youden(y_validation, validation_scores)
    if threshold_strategy is not None:
        raise ValueError(
            f"不支援 threshold_strategy={threshold_strategy!r}；"
            "目前可用 'max_f1' 或 'youden'")
    if target_recall is not None:
        return select_threshold_at_target_recall(
            y_validation, validation_scores, target_recall)
    raise ValueError('必須指定 threshold_strategy 或 target_recall')


def _wrap(clf, encoding):
    """onehot 模式：用 ColumnTransformer 分別處理「類別欄」與「數值欄」；dummy 模式只有 clf。

    - 類別欄（object/category）→ OneHotEncoder(min_frequency=10, handle_unknown='infrequent_if_exist')
      （保留罕見類別合併與未見類別處理）。
    - 數值欄（如拓樸的 degree/size/num_nodes/outlier）→ StandardScaler，不會被 one-hot 亂編。
    這樣同一個 onehot 模式可同時吃「純類別」(OriginModel) 與「類別+數值」(Full 加拓樸) 兩種輸入。
    """
    if encoding == 'onehot':
        cat_sel = make_column_selector(dtype_include=['object', 'category'])
        num_sel = make_column_selector(dtype_include=np.number)
        ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=True)
        pre = ColumnTransformer(
            [('oh', ohe, cat_sel),
             ('num', StandardScaler(with_mean=False), num_sel)],
            remainder='drop', sparse_threshold=0.3)
        return SkPipeline([('pre', pre), ('clf', clf)])
    return SkPipeline([('clf', clf)])


def logistic_cm_gridsearch(
        X, y, random_state=42, n_jobs=-1, encoding='dummy', n_iter=10,
        train_under_ratio=None, balance_test=False, target_recall=None,
        train_frac=0.7, validation_frac=0.1, threshold_strategy=None):
    use_validation = target_recall is not None or threshold_strategy is not None
    if not use_validation:
        X_train, X_test, y_train, y_test = _prepare_and_split(
            X, y, encoding, random_state, train_under_ratio, balance_test)
        X_validation = y_validation = None
    else:
        (X_train, X_validation, X_test,
         y_train, y_validation, y_test) = _prepare_train_validation_test(
            X, y, encoding, random_state, train_under_ratio,
            train_frac, validation_frac, balance_test)

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

    test_scores = best_model.predict_proba(X_test)[:, 1]
    if not use_validation:
        return y_test, test_scores, np.arange(len(y_test))

    validation_scores = best_model.predict_proba(X_validation)[:, 1]
    threshold_info = _select_validation_threshold(
        y_validation, validation_scores, threshold_strategy, target_recall)
    return y_test, test_scores, np.arange(len(y_test)), threshold_info


def linear_svc_cm_gridsearch(
        X, y, random_state=42, n_jobs=-1, encoding='dummy', n_iter=10,
        train_under_ratio=None, balance_test=False, target_recall=None,
        train_frac=0.7, validation_frac=0.1, threshold_strategy=None):
    use_validation = target_recall is not None or threshold_strategy is not None
    if not use_validation:
        X_train, X_test, y_train, y_test = _prepare_and_split(
            X, y, encoding, random_state, train_under_ratio, balance_test)
        X_validation = y_validation = None
    else:
        (X_train, X_validation, X_test,
         y_train, y_validation, y_test) = _prepare_train_validation_test(
            X, y, encoding, random_state, train_under_ratio,
            train_frac, validation_frac, balance_test)

    clf = LinearSVC(class_weight='balanced', max_iter=100000, random_state=random_state)
    pipe = _wrap(clf, encoding)
    param_dist = {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__loss': ['hinge', 'squared_hinge']}

    search = RandomizedSearchCV(pipe, param_dist, n_iter=min(n_iter, 10), cv=3,
                                scoring='accuracy', n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    test_scores = best_model.decision_function(X_test)
    if not use_validation:
        return y_test, test_scores, np.arange(len(y_test))

    validation_scores = best_model.decision_function(X_validation)
    threshold_info = _select_validation_threshold(
        y_validation, validation_scores, threshold_strategy, target_recall)
    return y_test, test_scores, np.arange(len(y_test)), threshold_info


def xgboost_cm_gridsearch(
        X, y, random_state=42, n_jobs=-1, encoding='dummy', n_iter=20,
        train_under_ratio=None, balance_test=False, target_recall=None,
        train_frac=0.7, validation_frac=0.1, threshold_strategy=None):
    use_validation = target_recall is not None or threshold_strategy is not None
    if not use_validation:
        X_train, X_test, y_train, y_test = _prepare_and_split(
            X, y, encoding, random_state, train_under_ratio, balance_test)
        X_validation = y_validation = None
    else:
        (X_train, X_validation, X_test,
         y_train, y_validation, y_test) = _prepare_train_validation_test(
            X, y, encoding, random_state, train_under_ratio,
            train_frac, validation_frac, balance_test)

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

    test_scores = best_model.predict_proba(X_test)[:, 1]
    if not use_validation:
        return y_test, test_scores, np.arange(len(y_test))

    validation_scores = best_model.predict_proba(X_validation)[:, 1]
    threshold_info = _select_validation_threshold(
        y_validation, validation_scores, threshold_strategy, target_recall)
    return y_test, test_scores, np.arange(len(y_test)), threshold_info


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
