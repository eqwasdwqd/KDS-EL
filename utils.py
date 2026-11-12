"""Data loading and utilities used by KDS-EL."""
import os
import re

import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat


def srbct_prefix(col):
    """SRBCT 列名前缀提取：'BL-C1' → 'BL', 'RMS.4' → 'RMS', 'EWS-T11' → 'EWS'"""
    s = re.sub(r'\.\d+$', '', col.strip())
    return s.split('-')[0]


def load_txt(name, data_dir):
    """加载 .txt 格式基因表达数据集。"""
    from config import SPECIAL_COL, SRBCT_KEEP
    fp = os.path.join(data_dir, f"{name}.txt")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Not found: {fp}")

    ls = ds = SPECIAL_COL.get(name, 1)
    lines = open(fp, encoding="utf-8", errors="replace").read().splitlines()
    header = lines[0].strip().split("\t")
    sample_names = [h for h in header[ls:] if h.strip()]

    if name == "SRBCT":
        raw_labels = [srbct_prefix(h) for h in sample_names]
        keep_mask  = [lb in SRBCT_KEEP for lb in raw_labels]
        labels_raw = [lb for lb, k in zip(raw_labels, keep_mask) if k]
    else:
        keep_mask  = [True] * len(sample_names)
        labels_raw = sample_names

    classes = sorted(set(labels_raw))
    cmap    = {c: i for i, c in enumerate(classes)}
    y       = np.array([cmap[s] for s in labels_raw], dtype=int)
    n       = len(y)
    keep_cols = [i for i, k in enumerate(keep_mask) if k]

    rows = []
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        vals  = [p for p in parts[ds:] if p.strip()]
        if len(vals) != len(sample_names):
            continue
        try:
            rows.append([float(vals[i]) for i in keep_cols])
        except ValueError:
            continue

    X = np.array(rows, dtype=float).T
    assert X.shape[0] == n, f"{name}: X.shape={X.shape} but n={n}"
    return X, y


def load_mat(name, data_dirs):
    """加载 .mat 格式数据集（ASU 标准格式）。"""
    for folder in data_dirs:
        fp = os.path.join(folder, f"{name}.mat")
        if os.path.exists(fp):
            break
    else:
        raise FileNotFoundError(f"Not found: {name}.mat")

    data = loadmat(fp)
    X = next((data[k].astype(float)
              for k in ["X", "data", "fea", "features"]
              if k in data and isinstance(data[k], np.ndarray)), None)
    Y = next((data[k].ravel().astype(int)
              for k in ["Y", "label", "gnd", "labels", "targets"]
              if k in data and isinstance(data[k], np.ndarray)), None)

    if X is None or Y is None:
        raise ValueError(f"{name}.mat: cannot find X/Y. Keys={list(data.keys())}")

    from scipy.sparse import issparse
    if issparse(X):
        X = X.toarray()

    return X, Y - Y.min()


def load_dataset(name, data_dir_txt, data_dir_mat):
    """统一数据加载入口：自动判断 txt/mat 格式，标签重编码 + 健康检查。"""
    from config import MAT_DATASETS
    if name in MAT_DATASETS:
        X, y = load_mat(name, [data_dir_mat, data_dir_txt])
    else:
        txt_fp  = os.path.join(data_dir_txt, f"{name}.txt")
        mat_fp1 = os.path.join(data_dir_txt, f"{name}.mat")
        mat_fp2 = os.path.join(data_dir_mat, f"{name}.mat")
        if os.path.exists(txt_fp):
            X, y = load_txt(name, data_dir_txt)
        elif os.path.exists(mat_fp1) or os.path.exists(mat_fp2):
            X, y = load_mat(name, [data_dir_mat, data_dir_txt])
        else:
            raise FileNotFoundError(f"Cannot find {name}.txt or {name}.mat")

    y = LabelEncoder().fit_transform(y)

    n_classes = len(np.unique(y))
    if n_classes == len(y):
        raise ValueError(
            f"{name}: n_classes({n_classes}) == n_samples — 标签未正确解析！")
    if n_classes < 2:
        raise ValueError(f"{name}: 只有 {n_classes} 个类，无法分类")

    return X, y


def compute_mi_scores(X, y):
    """计算每个特征与标签的互信息，归一化到 [0, 1]。

    用于种群初始化引导：高 MI 特征更可能被初始化为选中状态，
    解决知识权重冷启动问题。
    """
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    if mi.max() > mi.min():
        mi = (mi - mi.min()) / (mi.max() - mi.min())
    return mi


def ker_flip(weight, elite_bin, flip_count, rand_p):
    """
    KER 精英翻转：10% 概率随机翻转，90% 概率基于权重得分选择翻转特征。
    """
    mutant = elite_bin.copy()
    dim = len(mutant)
    if np.random.rand() < rand_p:
        fi = np.random.choice(dim, flip_count, replace=False)
    else:
        sc = np.where(elite_bin == 0, weight, -weight)
        sc += np.random.normal(0.0, np.std(weight) + 1e-5, dim)
        fi = np.argsort(-sc)[:flip_count]
    mutant[fi] = 1 - mutant[fi]
    return mutant


