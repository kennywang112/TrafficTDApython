"""5.4 / 5.5 評估工具：從既有 pkl 讀出預測結果、計算指標、彙總子群、建對照表。

每個 pkl 的格式皆為 dict：{'y', 'decision_scores', 'indices', 'elapsed_time'}。
- xgboost / logistic 的 decision_scores 是機率（[0,1]）
- svc 的 decision_scores 是 decision_function（可正可負）
指標中 roc_auc / pr_auc 為 threshold-free；labeled 指標依分數型態自動選門檻
（機率用 0.5、decision_function 用 0.0）。
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, f1_score, balanced_accuracy_score, accuracy_score,
                             roc_curve)

# 10 個子群（不含 full_data）——TDA 分群那一側
SUBGROUPS = ["pass_0", "pass_1", "pass_out_overlap",
             "car_0", "car_1", "car_2", "car_out_overlap",
             "motor_0", "motor_1", "motor_out_overlap"]

# 欄位順序：真實不平衡測試集下，優先看與門檻/類別比例較穩健的指標
# （PR-AUC、ROC-AUC 與門檻無關；balanced_acc、recall 次之；precision/f1/accuracy 依真實分布，放後面）
METRIC_KEYS = ['pr_auc', 'roc_auc', 'balanced_acc', 'recall', 'precision', 'f1', 'accuracy']


def _default_threshold(scores):
    s = np.asarray(scores, dtype=float)
    return 0.5 if (np.nanmin(s) >= 0.0 and np.nanmax(s) <= 1.0) else 0.0


def youden_threshold(y, scores):
    """回傳最大化 (TPR - FPR)（Youden's J）的門檻。"""
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), np.asarray(scores, dtype=float))
    return float(thr[np.argmax(tpr - fpr)])


def metrics_from_arrays(y, scores, threshold='fixed'):
    """threshold: 'fixed'(機率0.5/分數0.0) | 'youden' | 直接給數值。

    roc_auc / pr_auc 與門檻無關（最公平的比較指標）；recall/precision/f1/acc 依門檻計算。
    註：'youden' 是在「傳入的這批資料」上找門檻；若傳的是測試集，等於在測試集上調門檻，
    會略為樂觀。嚴格做法是用訓練/驗證分數找門檻再套到測試集（見 notebook 說明）。
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    if threshold == 'fixed':
        thr = _default_threshold(s)
    elif threshold == 'youden':
        thr = youden_threshold(y, s)
    else:
        thr = float(threshold)
    pred = (s >= thr).astype(int)
    out = {'threshold': round(float(thr), 4)}
    try:
        out['roc_auc'] = roc_auc_score(y, s)
    except Exception:
        out['roc_auc'] = np.nan
    try:
        out['pr_auc'] = average_precision_score(y, s)
    except Exception:
        out['pr_auc'] = np.nan
    out['recall'] = recall_score(y, pred, zero_division=0)
    out['precision'] = precision_score(y, pred, zero_division=0)
    out['f1'] = f1_score(y, pred, zero_division=0)
    out['balanced_acc'] = balanced_accuracy_score(y, pred)
    out['accuracy'] = accuracy_score(y, pred)
    out['n_test'] = int(len(y))
    return out


def load_pkl(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        # 例如舊 pkl 與現行 pandas 版本不相容；跳過該檔而非中斷整張表
        print(f"[warn] 無法讀取 {path}: {e}")
        return None


def metrics_from_pkl(path, threshold='fixed'):
    """單一 pkl → 指標；檔案不存在回傳 None。"""
    d = load_pkl(path)
    if d is None:
        return None
    return metrics_from_arrays(d['y'], d['decision_scores'], threshold=threshold)


def aggregate_tda(perf_dir, algo, subgroups=SUBGROUPS, pooled=True, threshold='fixed'):
    """把 10 個子群彙總成「TDA 系統整體」的一個指標。

    perf_dir 已含 encoding，例如 "ModelPerformanceSeed/onehot" 或舊版 "ModelPerformanceSeed/42"。
    pooled=True：把所有子群測試樣本的 y / 分數 concat 後算一個整體指標（同一 algo 分數型態一致，
                 這代表「每筆樣本由其所屬子群模型預測」的系統級表現，建議用這個）。
    pooled=False：以各子群測試集大小加權平均每群指標。
    缺檔的子群會被略過，並在 attrs 記錄。
    """
    ys, ss, per, missing = [], [], [], []
    for g in subgroups:
        p = f"{perf_dir}/{algo}/{g}.pkl"
        d = load_pkl(p)
        if d is None:
            missing.append(g)
            continue
        ys.append(np.asarray(d['y']))
        ss.append(np.asarray(d['decision_scores'], dtype=float))
        per.append(metrics_from_arrays(d['y'], d['decision_scores'], threshold=threshold))
    if not per:
        return None
    if pooled:
        agg = metrics_from_arrays(np.concatenate(ys), np.concatenate(ss), threshold=threshold)
    else:
        w = np.array([m['n_test'] for m in per], dtype=float)
        w = w / w.sum()
        agg = {k: float(np.nansum([m[k] * wi for m, wi in zip(per, w)])) for k in METRIC_KEYS}
        agg['n_test'] = int(sum(m['n_test'] for m in per))
    agg['n_groups'] = len(per)
    agg['missing'] = missing
    return agg


def _row(name, m):
    extra = ['threshold', 'n_test']
    if m is None:
        return {'method': name, **{k: np.nan for k in METRIC_KEYS + extra}}
    return {'method': name, **{k: m.get(k, np.nan) for k in METRIC_KEYS + extra}}


def build_table(rows):
    """rows: list of (name, metrics_dict|None) → 排版好的 DataFrame。"""
    df = pd.DataFrame([_row(n, m) for n, m in rows])
    return df.set_index('method')[METRIC_KEYS + ['threshold', 'n_test']].round(4)


# ---------------------------------------------------------------------------
# 5.4 Representation / DR comparison
# ---------------------------------------------------------------------------
def table_5_4(algo, perf_base="ModelPerformanceSeed", encoding="onehot",
              compare_base="../CompareOther", threshold='fixed'):
    """單一分類器下的表徵比較：Mapper(TDA) / one-hot / PCA / UMAP / MCA。

    threshold='youden' 會用 Youden's J 選最佳門檻（改善被固定 0.5 壓垮的 recall/f1）。
    """
    rows = [
        (f"Mapper (TDA, {encoding})", aggregate_tda(f"{perf_base}/{encoding}", algo, pooled=True, threshold=threshold)),
        ("One-hot (full_data)", metrics_from_pkl(f"{perf_base}/{encoding}/{algo}/full_data.pkl", threshold=threshold)),
        ("PCA (full_data)", metrics_from_pkl(f"{compare_base}/{algo}/pca_{algo}.pkl", threshold=threshold)),
        ("UMAP (full_data)", metrics_from_pkl(f"{compare_base}/{algo}/umap_{algo}.pkl", threshold=threshold)),
        ("MCA (full_data)", metrics_from_pkl(f"{compare_base}/{algo}/mca_only_{algo}.pkl", threshold=threshold)),
    ]
    return build_table(rows)


# ---------------------------------------------------------------------------
# 5.5 Ablation（可由既有 pkl 產生的列）
# ---------------------------------------------------------------------------
def table_5_5(algo, perf_base="ModelPerformanceSeed",
              time_enc="onehot", old_random_dir=None, threshold='fixed'):
    """Ablation 對照表。

    可由既有 pkl 直接算的列：
      A0 Full(reference) = TDA 彙總（time split, onehot）
      A1 −TDA grouping   = full_data 單一全域模型
      A4 encoding        = TDA 彙總（dummy）對照 A0
      A5 split           = 舊的隨機切分 TDA 彙總（若提供 old_random_dir，例如 "ModelPerformanceSeed/42"）
    需另外重跑、無現成 pkl 的列（A2 −feature selection、A3 −imbalance handling）
    以 NaN 呈現，跑完後再填。
    """
    rows = [
        ("A0 Full (TDA, time, onehot)", aggregate_tda(f"{perf_base}/{time_enc}", algo, pooled=True, threshold=threshold)),
        ("A1 -TDA (full_data, onehot)", metrics_from_pkl(f"{perf_base}/{time_enc}/{algo}/full_data.pkl", threshold=threshold)),
        ("A4 encoding=dummy (TDA)", aggregate_tda(f"{perf_base}/dummy", algo, pooled=True, threshold=threshold)),
    ]
    if old_random_dir is not None:
        rows.append(("A5 split=random (TDA, old)", aggregate_tda(old_random_dir, algo, pooled=True, threshold=threshold)))
    else:
        rows.append(("A5 split=random (TDA, old)", None))
    # 需另外跑
    rows.append(("A2 -feature selection", None))
    rows.append(("A3 -imbalance handling", None))
    return build_table(rows)
