"""5.4 統一評估（路線 A）：單一時間切點 + 同一批測試列，讓所有方法 apples-to-apples。

流程：
1. 讀所有子群 CSV → concat 成 canonical 全資料（每列帶 subgroup / 日期 / 死亡），依時間排序。
2. 單一時間切點（預設後 30%）：train = 前段、test = 後段。
3. 測試集只平衡一次（RandomUnderSampler），固定成同一批 test 列 → 所有方法都預測這批。
4. 全域法（onehot / pca / umap / mca）：在 train 上 fit 表徵 + 分類器，預測 test。
   Mapper：各子群在「子群∩train」訓練、預測「子群∩test」，再拼回同一批 test 列。
5. 每個方法回傳對「同一批 test 列」的分數，指標由 evaluate.metrics_from_arrays 計算。

需求：umap-learn、prince（只有用到 umap/mca 時才需要，延遲載入）。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
try:
    from utils.models import _build_search_estimator
    from utils.evaluate import metrics_from_arrays, build_table
except Exception:
    # 當 cwd 不是 Models（例如 Comparison.ipynb chdir 到 Version3）時，改用同資料夾的兄弟模組
    from models import _build_search_estimator
    from evaluate import metrics_from_arrays, build_table

# OriginModel 使用的特徵欄位（含 死亡）
SELECT_LST = [
    '天候名稱', '光線名稱', '道路類別-第1當事者-名稱', '速限-第1當事者',
    '路面狀況-路面鋪裝名稱', '路面狀況-路面狀態名稱', '路面狀況-路面缺陷名稱',
    '道路障礙-障礙物名稱', '道路障礙-視距品質名稱', '道路障礙-視距名稱',
    '號誌-號誌種類名稱', '號誌-號誌動作名稱',
    '車道劃分設施-分道設施-快車道或一般車道間名稱', '車道劃分設施-分道設施-快慢車道間名稱', '車道劃分設施-分道設施-路面邊線名稱',
    '當事者屬-性-別名稱', '當事者事故發生時年齡', '保護裝備名稱', '行動電話或電腦或其他相類功能裝置名稱',
    '肇事逃逸類別名稱-是否肇逃',
    '道路型態大類別名稱', '事故位置大類別名稱', '車道劃分設施-分向設施大類別名稱',
    '事故類型及型態大類別名稱', '當事者區分-類別-大類別名稱-車種', '車輛撞擊部位大類別名稱-其他',
    '肇因研判大類別名稱-主要',
    '道路型態子類別名稱', '事故位置子類別名稱', '事故類型及型態子類別名稱', '肇因研判子類別名稱-主要',
    '當事者區分-類別-子類別名稱-車種', '當事者行動狀態子類別名稱', '車輛撞擊部位子類別名稱-最初',
    '車輛撞擊部位子類別名稱-其他', '肇因研判子類別名稱-個別',
    '死亡',
]
FEATURES = [c for c in SELECT_LST if c != '死亡']
TIME_COLS = ['發生日期', '發生時間']


def _load(root, rel):
    return pd.read_csv(os.path.join(root, rel), encoding='utf-8')


def load_subgroups(data_root):
    """回傳 {subgroup_name: DataFrame}，與 OriginModel 的分群一致。"""
    def oo(a, b):
        a = a.copy(); b = b.copy(); a['type'] = 'out'; b['type'] = 'overlap'
        return pd.concat([a, b], ignore_index=True)

    subs = {
        'car_0': _load(data_root, "CarData/full_0.csv"),
        'car_1': _load(data_root, "CarData/full_1.csv"),
        'car_2': _load(data_root, "CarData/full_2.csv"),
        'car_out_overlap': oo(_load(data_root, "CarData/full_out.csv"), _load(data_root, "CarData/overlap_data.csv")),
        'motor_0': _load(data_root, "MotorData/full_0.csv"),
        'motor_1': _load(data_root, "MotorData/full_1.csv"),
        'motor_out_overlap': oo(_load(data_root, "MotorData/full_out.csv"), _load(data_root, "MotorData/overlap_data.csv")),
        'pass_0': _load(data_root, "PassData/full_0.csv"),
        'pass_1': _load(data_root, "PassData/full_1.csv"),
        'pass_out_overlap': oo(_load(data_root, "PassData/full_out.csv"), _load(data_root, "PassData/overlap_data.csv")),
    }
    # pass_* 的類別覆寫（與 OriginModel 一致）
    for k in ['pass_0', 'pass_1', 'pass_out_overlap']:
        subs[k]['行動電話或電腦或其他相類功能裝置名稱'] = '非駕駛人'
        subs[k]['當事者區分-類別-大類別名稱-車種'] = '人'
    return subs


def build_canonical(data_root="../Version3/Data"):
    """concat 所有子群 → canonical 全資料（features + y + subgroup + 時間），依時間排序。

    不同子群欄位若不一致（例如 motor 缺子類別），缺的欄位以 '未紀錄' 補齊，
    確保全域法可統一編碼；Mapper 端則各子群只用自己實際有的欄位。
    """
    subs = load_subgroups(data_root)
    frames = []
    for name, df in subs.items():
        cols = [c for c in SELECT_LST if c in df.columns] + [c for c in TIME_COLS if c in df.columns]
        sub = df[cols].copy()
        sub['subgroup'] = name
        frames.append(sub)
    full = pd.concat(frames, ignore_index=True)
    # 補齊缺欄（例如 motor 缺子類別）
    for c in FEATURES:
        if c not in full.columns:
            full[c] = '未紀錄'
    full[FEATURES] = full[FEATURES].fillna('未紀錄')
    full = full.sort_values(TIME_COLS).reset_index(drop=True)
    full['y'] = (full['死亡'] >= 1).astype(int)
    return full


def time_cutoff_split(full, test_frac=0.3):
    n = len(full)
    n_train = n - int(np.ceil(n * test_frac))
    cutoff_date = full.iloc[n_train][TIME_COLS[0]]
    return full.index[:n_train], full.index[n_train:], cutoff_date


def balanced_test_index(full, test_idx, random_state=42):
    """對 test 段做一次平衡下採樣，回傳固定的一批 test 列 index（所有方法共用）。"""
    yte = full.loc[test_idx, 'y'].values
    idx_col = np.asarray(test_idx).reshape(-1, 1)
    idx_res, _ = RandomUnderSampler(random_state=random_state).fit_resample(idx_col, yte)
    return pd.Index(np.sort(idx_res.ravel()))


def _under(Xtr, ytr, ratio, random_state):
    if ratio is None:
        return Xtr, ytr
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    if n_pos > 0 and n_neg > n_pos / ratio:
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
        Xtr, ytr = rus.fit_resample(Xtr, ytr)
    return Xtr, ytr


def _fit_predict(Ztr, ytr, Zte, algo, random_state, n_jobs=-1):
    ytr = np.asarray(ytr)
    mc = int(min((ytr == 0).sum(), (ytr == 1).sum()))
    clf, param, n_iter = _build_search_estimator(algo, ytr, random_state, n_jobs)
    if mc < 3:
        # 少數類太少，無法 cv=3；直接用預設參數擬合
        clf.fit(Ztr, ytr)
        best = clf
    else:
        search = RandomizedSearchCV(clf, param, n_iter=n_iter, cv=min(3, mc),
                                    scoring='accuracy', n_jobs=n_jobs, random_state=random_state)
        search.fit(Ztr, ytr)
        best = search.best_estimator_
    if algo == 'svc':
        return best.decision_function(Zte)
    return best.predict_proba(Zte)[:, 1]


def run_global(full, train_idx, test_eval_idx, repr='onehot', algo='xgboost',
               n_components=10, random_state=42, train_under_ratio=0.1, n_jobs=-1):
    """全域法：onehot / pca / umap / mca；預測固定的 test_eval_idx。"""
    Xtr = full.loc[train_idx, FEATURES].astype(str)
    ytr = full.loc[train_idx, 'y'].values
    Xte = full.loc[test_eval_idx, FEATURES].astype(str)
    yte = full.loc[test_eval_idx, 'y'].values

    Xtr, ytr = _under(Xtr, ytr, train_under_ratio, random_state)

    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False)
    if repr == 'onehot':
        Ztr = ohe.fit_transform(Xtr); Zte = ohe.transform(Xte)
    elif repr == 'pca':
        from sklearn.decomposition import PCA
        Ftr = ohe.fit_transform(Xtr); Fte = ohe.transform(Xte)
        red = PCA(n_components=n_components, random_state=random_state)
        Ztr = red.fit_transform(Ftr); Zte = red.transform(Fte)
        sc = MinMaxScaler().fit(Ztr); Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
    elif repr == 'umap':
        import umap
        Ftr = ohe.fit_transform(Xtr); Fte = ohe.transform(Xte)
        red = umap.UMAP(n_components=n_components, random_state=random_state)
        Ztr = red.fit_transform(Ftr); Zte = red.transform(Fte)
        sc = MinMaxScaler().fit(Ztr); Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
    elif repr == 'mca':
        import prince
        red = prince.MCA(n_components=n_components).fit(Xtr)
        Ztr = red.transform(Xtr).to_numpy(); Zte = red.transform(Xte).to_numpy()
        sc = MinMaxScaler().fit(Ztr); Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
    else:
        raise ValueError(repr)

    scores = _fit_predict(Ztr, ytr, Zte, algo, random_state, n_jobs)
    return yte, scores


def _feature_select_train(train_df, threshold=0.01, random_state=42):
    """在子群訓練集上做 RF 特徵重要性篩選（回傳保留的原始欄位名）。"""
    cols = [c for c in FEATURES if c in train_df.columns]
    X = pd.get_dummies(train_df[cols].astype(str))
    y = train_df['y'].values
    if len(np.unique(y)) < 2:
        return cols
    m = RandomForestClassifier(random_state=random_state)
    m.fit(X, y)
    fi = pd.Series(m.feature_importances_, index=X.columns)
    sel = fi[fi > threshold].index.tolist()
    keep = [c for c in cols if any(d.startswith(f"{c}_") for d in sel) or c in sel]
    return keep or cols


def run_mapper(full, train_idx, test_eval_idx, algo='xgboost',
               random_state=42, train_under_ratio=0.1, n_jobs=-1):
    """Mapper：各子群在「子群∩train」訓練、預測「子群∩test_eval」，拼回同一批 test 列。"""
    train_set = set(train_idx)
    scores = pd.Series(index=test_eval_idx, dtype=float)
    test_sub = full.loc[test_eval_idx, 'subgroup']

    for name in full['subgroup'].unique():
        s_test = test_eval_idx[test_sub.values == name]
        if len(s_test) == 0:
            continue
        s_train = full.index[(full['subgroup'] == name) & full.index.isin(train_set)]
        if len(s_train) == 0:
            scores.loc[s_test] = 0.0
            continue

        feat = _feature_select_train(full.loc[s_train], random_state=random_state)
        Xtr = full.loc[s_train, feat].astype(str)
        ytr = full.loc[s_train, 'y'].values
        Xte = full.loc[s_test, feat].astype(str)

        Xtr, ytr = _under(Xtr, ytr, train_under_ratio, random_state)
        if len(np.unique(ytr)) < 2:
            scores.loc[s_test] = float(ytr[0]) if len(ytr) else 0.0
            continue

        ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False)
        Ztr = ohe.fit_transform(Xtr); Zte = ohe.transform(Xte)
        sc = _fit_predict(Ztr, ytr, Zte, algo, random_state, n_jobs)
        scores.loc[s_test] = sc

    yte = full.loc[test_eval_idx, 'y'].values
    return yte, scores.values


def table_5_4_unified(algo='xgboost', data_root="../Version3/Data", test_frac=0.3,
                      random_state=42, train_under_ratio=0.1, threshold='youden',
                      methods=('mapper', 'onehot', 'pca', 'umap', 'mca'), balance_test=False):
    """一次跑完統一評估並回傳對照表（同一批 test 列，n_test 一致）。

    balance_test=False（預設）：測試集保留真實分布（建議；precision/PR-AUC 才真實）。
    balance_test=True：測試集下採樣成 50/50（僅供對照，會高估 precision/F1）。
    """
    full = build_canonical(data_root)
    train_idx, test_idx, cutoff = time_cutoff_split(full, test_frac)
    test_eval_idx = balanced_test_index(full, test_idx, random_state) if balance_test else test_idx
    print(f"時間切點 ~ {cutoff} | train={len(train_idx)} test={len(test_eval_idx)} "
          f"(balance_test={balance_test}, 正類佔比={full.loc[test_eval_idx,'y'].mean():.4f})")

    rows = []
    for m in methods:
        print(f"  running {m} ...")
        if m == 'mapper':
            y, s = run_mapper(full, train_idx, test_eval_idx, algo, random_state, train_under_ratio)
            label = 'Mapper (TDA)'
        else:
            y, s = run_global(full, train_idx, test_eval_idx, repr=m, algo=algo,
                              random_state=random_state, train_under_ratio=train_under_ratio)
            label = {'onehot': 'One-hot', 'pca': 'PCA', 'umap': 'UMAP', 'mca': 'MCA'}[m]
        rows.append((label, metrics_from_arrays(y, s, threshold=threshold)))
    return build_table(rows)
