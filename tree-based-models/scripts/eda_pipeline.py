"""資料與變數介紹 → 判斷任務類型 → EDA → 遺失值檢查（有遺失才診斷與插補）→ 資料切分 → 樹模型比較（一支跑完）。

流程（與 report.html 章節一致）:
    1 資料與變數介紹（欄位、說明、單位、角色、型態；可搭配 --dict 欄位說明檔）
    2 判斷任務類型（分類 or 迴歸，決定模型、指標、切分方式）
    3 敘述性統計
    4 遺失值檢查：沒有遺失 → 跳過診斷與處理；有遺失 → 遺失機制診斷（MCAR/MAR/MNAR）
    5 資料切分（holdout 測試集 + 訓練集 K 折 CV；分類 stratify）
    6 遺失值處理：插補方法比較（只在有遺失時；只用訓練集 CV）
    7 模型訓練與比較：Decision Tree / Random Forest / Gradient Boosting，各自 GridSearchCV（5 折）調參

用法:
    pip install scikit-learn pandas numpy scipy matplotlib openpyxl
    python eda_pipeline.py classification_wine.csv --target cultivar
    python eda_pipeline.py data.csv --target price_wan --task regression \
        --dict regression_house_price_dictionary.csv --out eda_house

參數:
    --target   目標欄位（必填）
    --task     classification / regression / auto（預設 auto：文字/布林、2 種值、或 ≤10 種整數值 → 分類，其餘 → 迴歸）
    --drop     額外要丟掉的欄位（ID 類欄位會自動偵測並丟掉）
    --dict     欄位說明檔（CSV，欄位：欄位, 說明, 單位[, 角色]）；未指定時自動找同資料夾的 <檔名>_dictionary.csv
    --out      輸出資料夾（預設 eda_<檔名>）
    --fast     跳過較慢的 MICE-ExtraTrees 插補
    --no-grid  不做 GridSearchCV，改用預設參數（較快）

輸出（全部在 --out 內）:
    report.html            圖文報告（可直接打開 / 分享，圖片內嵌）
    descriptive_stats.xlsx 所有表格（敘述統計、遺失診斷、插補比較、模型比較、重要度）
    figures/*.png          所有圖
    report.txt             主控台輸出
    summary.json           關鍵數字（給程式讀）

設計原則:
    * 測試集最後才碰；插補 / 編碼都放在 Pipeline 裡，只在訓練折 fit（避免資料洩漏）
    * 插補法用「同一個下游模型（Gradient Boosting）+ 同一組 CV 折」公平比較，而不是憑感覺挑
    * 遺失機制（MCAR/MAR/MNAR）只能「推測」：MAR 可從資料看出跡象，MNAR 無法從觀測資料證明
"""
import argparse
import base64
import io
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_absolute_percentage_error, r2_score,
                             roc_auc_score, roc_curve, root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 42


# ============================================================================ 0. 設定
ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--target", required=True)
ap.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--dict", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--no-grid", action="store_true", help="不做 GridSearchCV，改用預設參數（較快）")
args = ap.parse_args()

OUT = Path(args.out or f"eda_{Path(args.csv).stem}")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

_log = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _log.append(s)


# 中文字型（找不到就退回預設字型，圖仍可產生但中文會變方塊）
for name in ["Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans TC", "Microsoft JhengHei", "PingFang TC",
             "Heiti TC", "SimHei", "Arial Unicode MS"]:
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9b8b2",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY, INK = "#9a998f", "#0b0b0b"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ec", "#eb6834"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f1ec", "#9cc3ef", "#2a78d6", "#0d3c75"])

FIGS = {}  # key -> (path, caption)


def save(fig, key, caption):
    p = FIG / f"{key}.png"
    fig.savefig(p)
    plt.close(fig)
    FIGS[key] = (p, caption)


def grid(n, ncols=3, w=4.0, h=2.8):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


TABLES = {}  # sheet name -> DataFrame（同時進 Excel 與 HTML）
FINDINGS = {"task": [], "check": [], "split": [], "overview": [], "desc": [], "missing": [], "impute": [], "model": []}

# ============================================================================ 1. 讀檔 + 任務類型判斷 + 資料概覽
t0 = time.time()
df = pd.read_csv(args.csv)
target = args.target
if target not in df.columns:
    sys.exit(f"找不到目標欄位 {target}；可用欄位: {list(df.columns)}")
df = df[df[target].notna()].reset_index(drop=True)  # 目標遺失的列不能用於監督式學習

# ---------------------------------------------------------------- 1.0 第一步：判斷任務類型（分類 or 迴歸）
def detect_task(s):
    """依目標欄位的型態與取值判斷任務；回傳 (task, 子類型, 判斷理由, 提醒, 依據表)。
    規則：文字/布林 → 分類；數值且只有 2 種值 → 二元分類；整數且 ≤10 種值 → 多類別分類（但可能是計數/評分）；
          其餘（連續小數、很多不同值）→ 迴歸。"""
    s = s.dropna()
    n, n_u = len(s), s.nunique()
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    is_int = bool(numeric and np.allclose(s, np.round(s)))
    warn = ""
    if not numeric:
        t, sub, why = "classification", ("二元分類" if n_u == 2 else f"多類別分類（{n_u} 類）"), "目標是文字／布林值，代表「類別」而非數量"
    elif n_u == 2:
        t, sub, why = "classification", "二元分類", f"目標只有 2 種值（{sorted(s.unique().tolist())}），是「是/否」型標籤"
    elif is_int and n_u <= 10:
        t, sub, why = "classification", f"多類別分類（{n_u} 類）", f"目標是整數且只有 {n_u} 種值"
        warn = "若這些整數其實是「計數」或「評分」（例如 1–5 星、來電次數），有大小順序與距離意義，也可以當迴歸或有序模型處理，請和業務方確認。"
    else:
        t = "regression"
        sub = "連續數值預測"
        why = (f"目標是數值且有 {n_u:,} 種不同值（佔 {n_u / n:.1%}）" + ("，含小數" if not is_int else "，雖為整數但取值很多")
               + "，大小與差距都有意義")
        if is_int and n_u <= 30:
            warn = "目標是取值不多的整數，可能是計數資料；樹模型仍可當迴歸處理，但要確認業務上關心的是數量還是類別。"
    ev = pd.DataFrame({
        "檢查項目": ["目標欄位", "dtype", "是否為數值", "是否全為整數", "不同值個數", "不同值佔比", "範例值", "判斷"],
        "結果": [s.name, str(s.dtype), "是" if numeric else "否", "是" if is_int else "否", f"{n_u:,}", f"{n_u / n:.2%}",
               ", ".join(map(str, s.unique()[:6])), f"{'分類' if t == 'classification' else '迴歸'}：{sub}"],
    })
    return t, sub, why, warn, ev


auto_task, task_sub, task_why, task_warn, task_ev = detect_task(df[target])
task = auto_task if args.task == "auto" else args.task
if args.task != "auto" and args.task != auto_task:
    task_warn = (task_warn + " " if task_warn else "") + f"使用者以 --task 指定為 {args.task}（自動判斷為 {auto_task}），以使用者指定為準。"
    task_sub = "使用者指定"
IS_CLF = task == "classification"
TABLES["任務類型判斷"] = task_ev

# 自動偵測 ID 類欄位：名稱像 id，或每列都不同的文字欄位
id_cols = [c for c in df.columns if c != target and (
    c.lower() in ("id", "index") or c.lower().endswith("_id") or
    (not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == len(df)))]
drop_cols = sorted(set(id_cols) | set(args.drop))
X_all = df.drop(columns=drop_cols + [target])
y_all = df[target]
CLS = {}  # 類別代碼 -> 顯示名稱
if IS_CLF and not pd.api.types.is_numeric_dtype(y_all):
    classes = sorted(y_all.unique())
    y_all = y_all.map({c: i for i, c in enumerate(classes)})
    CLS = dict(enumerate(classes))
elif IS_CLF:
    CLS = {k: str(k) for k in sorted(y_all.unique())}
cname = lambda k: CLS.get(k, str(k))
BINARY = IS_CLF and y_all.nunique() == 2
num_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
cat_cols = [c for c in X_all.columns if c not in num_cols]
# 離散數值（整數且 ≤10 種值，如房間數、車位數）：分佈用長條圖，而非直方圖
disc_cols = [c for c in num_cols if X_all[c].dropna().nunique() <= 10
             and np.allclose(X_all[c].dropna(), np.round(X_all[c].dropna()))]

n_dup = int(df.duplicated(subset=[c for c in df.columns if c not in id_cols]).sum())
const_cols = [c for c in X_all.columns if X_all[c].nunique(dropna=True) <= 1]
# ---------------------------------------------------------------- 1.1 資料與變數介紹（可搭配欄位說明檔）
dict_path = Path(args.dict) if args.dict else Path(args.csv).with_name(f"{Path(args.csv).stem}_dictionary.csv")
ddict = {}
if dict_path.exists():
    _d = pd.read_csv(dict_path)
    ddict = {r["欄位"]: r for _, r in _d.iterrows()}
DATASET_DESC = str(ddict.pop("__dataset__")["說明"]) if "__dataset__" in ddict else ""


def role_of(c):
    if c == target:
        return "目標"
    return "ID（排除）" if c in id_cols else ("排除" if c in drop_cols else "特徵")


def kind_of(c):
    if c == target:
        return "類別（標籤）" if IS_CLF else "數值（連續）"
    if c in drop_cols:
        return "識別碼"
    return "數值" if c in num_cols else "類別"


def desc_of(c, with_unit=True):
    if c not in ddict:
        return ""
    d, u = str(ddict[c]["說明"]), ddict[c].get("單位")
    return d + (f"，{u}" if with_unit and pd.notna(u) and str(u) not in ("—", "-", "") else "")


var_tbl = pd.DataFrame({
    "欄位": df.columns,
    "說明": [str(ddict[c]["說明"]) if c in ddict else "—" for c in df.columns],
    "單位": [str(ddict[c]["單位"]) if c in ddict and pd.notna(ddict[c].get("單位")) else "—" for c in df.columns],
    "角色": [role_of(c) for c in df.columns],
    "變數型態": [kind_of(c) for c in df.columns],
    "dtype": [str(df[c].dtype) for c in df.columns],
    "遺失數": [int(df[c].isna().sum()) for c in df.columns],
})
overview = pd.DataFrame({  # 特徵層級（給後續遺失值檢查用）
    "欄位": X_all.columns, "型態": ["數值" if c in num_cols else "類別" for c in X_all.columns],
    "遺失數": X_all.isna().sum().values, "遺失%": (X_all.isna().mean() * 100).round(2).values,
})
TABLES["變數與欄位介紹"] = var_tbl
log(f"=== 1. 資料與變數介紹: {args.csv} ===")
log(f"{df.shape[0]} 列 × {df.shape[1]} 欄；欄位說明檔: {dict_path.name if ddict else '（無）'}")
log(var_tbl.to_string(index=False))
log(f"\n=== 2. 任務類型判斷 ===")
log(task_ev.to_string(index=False))
log(f"→ {'分類' if IS_CLF else '迴歸'}（{task_sub}）：{task_why}" + (f"\n  提醒：{task_warn}" if task_warn else ""))
FINDINGS["task"] += [
    f"<b>判斷結果：{'分類' if IS_CLF else '迴歸'}（{task_sub}）</b>。理由：{task_why}。",
    ("分類 = 預測「屬於哪一類」，輸出類別或機率；評估用 AUC、PR-AUC、F1、混淆矩陣；切分要 stratify 保持類別比例。"
     if BINARY else
     "多類別分類 = 預測「屬於哪一類」（超過兩類）；評估用 Accuracy、Macro-F1、混淆矩陣；切分要 stratify 保持各類別比例。"
     if IS_CLF else
     "迴歸 = 預測「連續數值是多少」；評估用 RMSE、MAE、R²、MAPE；切分用一般隨機 KFold。"),
    "判斷規則：文字／布林 → 分類；只有 2 種值 → 二元分類；整數且 ≤10 種值 → 多類別分類（需確認是否為計數/評分）；其餘數值 → 迴歸。"
    "任務類型決定了之後的模型（Classifier vs Regressor）、評估指標與切分方式，所以是第一步。",
] + ([f"⚠ {task_warn}"] if task_warn else [])
_tgt_desc = desc_of(target)
if IS_CLF:
    _tgt_detail = f"，共 {y_all.nunique()} 類：" + "、".join(f"{cname(k)}（{v} 筆）" for k, v in y_all.value_counts().sort_index().items())
else:
    _tgt_detail = f"；平均 {y_all.mean():,.1f}、範圍 {y_all.min():,.0f}–{y_all.max():,.0f}"
FINDINGS["overview"] += ([f"<b>資料說明</b>：{DATASET_DESC}"] if DATASET_DESC else []) + [
    f"<b>分析目的（要預測什麼）</b>：預測 <code>{target}</code>" + (f"——{_tgt_desc}" if _tgt_desc else "") + f"{_tgt_detail}。"
    f"這是一個{'分類' if IS_CLF else '迴歸'}問題（判斷依據見第 2 節）。",
    f"<b>用了哪些變數</b>：{X_all.shape[1]} 個特徵（數值 {len(num_cols)}、類別 {len(cat_cols)}）——" + "；".join(
        f"<code>{c}</code>" + (f"（{desc_of(c)}）" if desc_of(c) else "") for c in X_all.columns) + "。",
]
FINDINGS["overview"] += [
    f"資料共 <b>{df.shape[0]:,} 筆、{df.shape[1]} 個欄位</b>：目標 1 個（<code>{target}</code>）、"
    f"特徵 {X_all.shape[1]} 個（數值 {len(num_cols)}、類別 {len(cat_cols)}）、ID／排除 {len(drop_cols)} 個。",
    (f"ID 類欄位 {', '.join(drop_cols)} 只用來識別資料列，沒有預測意義，建模前排除（留著還可能造成洩漏）。"
     if drop_cols else "沒有 ID 類欄位，所有非目標欄位都當作特徵。"),
    f"完全重複的列 {n_dup} 筆；常數欄位 {const_cols or '無'}。"
    + ("" if ddict else "（未提供欄位說明檔，可用 --dict 指定，讓下表帶出中文說明與單位。）"),
]

# ============================================================================ 3. 敘述性統計
log("\n=== 3. 敘述性統計 ===")


def iqr_bounds(s, k):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


rows = []
for c in num_cols + ([target] if not IS_CLF else []):
    s = (df[c] if c == target else X_all[c]).dropna()
    lo, hi = iqr_bounds(s, 1.5)
    rows.append({
        "欄位": c, "個數": len(s), "遺失%": round(100 * (1 - len(s) / len(df)), 2),
        "平均數": s.mean(), "標準差": s.std(), "最小值": s.min(), "Q1": s.quantile(.25), "中位數": s.median(),
        "Q3": s.quantile(.75), "最大值": s.max(), "IQR": s.quantile(.75) - s.quantile(.25),
        "變異係數CV": s.std() / s.mean() if s.min() >= 0 and s.mean() > 0 else np.nan,  # 只對非負變數有意義
        "偏態": stats.skew(s), "峰態(超額)": stats.kurtosis(s),
        "離群值數(1.5IQR)": int(((s < lo) | (s > hi)).sum()),
    })
num_desc = pd.DataFrame(rows).set_index("欄位").round(3)
TABLES["數值敘述統計"] = num_desc.reset_index()
log(num_desc.to_string())

cat_rows = []
for c in cat_cols + ([target] if IS_CLF else []):
    s = df[c]
    vc = s.value_counts(dropna=True)
    cat_rows.append({
        "欄位": c, "個數": int(s.notna().sum()), "遺失%": round(100 * s.isna().mean(), 2),
        "類別數": int(s.nunique()), "眾數": vc.index[0], "眾數佔比%": round(100 * vc.iloc[0] / vc.sum(), 2),
        "最少類別": vc.index[-1], "最少類別佔比%": round(100 * vc.iloc[-1] / vc.sum(), 2),
        "各類別次數": "; ".join(f"{k}={v}" for k, v in vc.items()),
    })
if cat_rows:
    cat_desc = pd.DataFrame(cat_rows)
    TABLES["類別敘述統計"] = cat_desc
    log(cat_desc.drop(columns="各類別次數").to_string(index=False))

# --- 目標變數
if IS_CLF:
    tgt_vc = y_all.value_counts().sort_index()
    pos_rate = y_all.mean() if y_all.nunique() == 2 else None
    imb = tgt_vc.min() / tgt_vc.max()
    FINDINGS["desc"].append(
        f"目標類別分佈：{', '.join(f'{k}={v}（{v / len(y_all):.1%}）' for k, v in tgt_vc.items())}。"
        + ("類別不平衡，評估請看 AUC / PR-AUC / F1，不要只看 accuracy；切分要 stratify。" if imb < 0.6 else ""))
else:
    sk = stats.skew(y_all)
    FINDINGS["desc"].append(
        f"目標 {target} 平均 {y_all.mean():,.1f}、中位數 {y_all.median():,.1f}、偏態 {sk:.2f}。"
        + ("右偏明顯：報告時中位數比平均數更有代表性；若在意相對誤差，可考慮對 log(y) 建模"
           "（sklearn 的 TransformedTargetRegressor）。" if sk > 1 else ""))

skewed = num_desc.loc[[c for c in num_cols if abs(num_desc.loc[c, "偏態"]) > 1], "偏態"]
if len(skewed):
    FINDINGS["desc"].append("偏態 |skew|>1 的欄位：" + "、".join(f"{k}（{v:.2f}）" for k, v in skewed.items())
                            + "。樹模型不受單調轉換影響，不必為了模型做 log；但線性模型或距離型插補（KNN）會受影響。")
# --- 與目標的關係（單變量）
rel = []
for c in num_cols:
    s = X_all[c]
    m = s.notna()
    if IS_CLF and y_all.nunique() == 2:
        a, b = s[m & (y_all == 1)], s[m & (y_all == 0)]
        p = stats.mannwhitneyu(a, b).pvalue
        smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        rel.append({"特徵": c, "類型": "數值", f"{target}=1 平均": a.mean(), f"{target}=0 平均": b.mean(),
                    "效果量": smd, "效果量指標": "SMD", "p值": p, "檢定": "Mann-Whitney U"})
    elif IS_CLF:
        ks_ = sorted(y_all.unique())
        groups = [s[m & (y_all == k)] for k in ks_]
        h, p = stats.kruskal(*groups)
        eps2 = (h - len(groups) + 1) / (m.sum() - len(groups))
        row = {"特徵": c, "類型": "數值"}
        row.update({f"{cname(k)} 平均": g.mean() for k, g in zip(ks_, groups)})
        row.update({"效果量": eps2, "效果量指標": "η²(H)", "p值": p, "檢定": "Kruskal-Wallis"})
        rel.append(row)
    else:
        r, p = stats.spearmanr(s[m], y_all[m])
        rel.append({"特徵": c, "類型": "數值", "效果量": r, "效果量指標": "Spearman ρ", "p值": p, "檢定": "Spearman"})
for c in cat_cols:
    s = X_all[c].fillna("（遺失）")
    if IS_CLF:
        ct = pd.crosstab(s, y_all)
        chi2, p, _, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        rel.append({"特徵": c, "類型": "類別", "效果量": v, "效果量指標": "Cramér's V", "p值": p, "檢定": "卡方"})
    else:
        groups = [y_all[s == k] for k in s.unique()]
        h, p = stats.kruskal(*groups)
        eps2 = (h - len(groups) + 1) / (len(y_all) - len(groups))
        rel.append({"特徵": c, "類型": "類別", "效果量": eps2, "效果量指標": "η²(H)", "p值": p, "檢定": "Kruskal-Wallis"})
rel_df = pd.DataFrame(rel)
rel_df["|效果量|"] = rel_df["效果量"].abs()
rel_df = rel_df.sort_values("|效果量|", ascending=False).drop(columns="|效果量|").round(4)
top_rel = rel_df.head(4)
FINDINGS["desc"].append("單變量與目標關係最強的特徵：" + "、".join(
    f"{r['特徵']}（{r['效果量指標']}={r['效果量']:.2f}）" for _, r in top_rel.iterrows())
    + "。這只是單變量篩選，交互作用與非線性要靠後面的模型。")

if BINARY and cat_cols:  # 類別特徵在各水準的目標比率表
    TABLES["類別水準目標率"] = pd.concat([
        df.assign(**{c: df[c].fillna("（遺失）")}).groupby(c)[target].agg(["count", "mean"])
        .rename(columns={"count": "筆數", "mean": f"{target} 比率"}).reset_index()
        .rename(columns={c: "水準"}).assign(特徵=c) for c in cat_cols], ignore_index=True
    )[["特徵", "水準", "筆數", f"{target} 比率"]].round(4)

# --- 相關性與共線性
corr_cols = num_cols + [target] if (not IS_CLF or BINARY) else num_cols
corr = pd.concat([X_all[num_cols], y_all.rename(target)], axis=1)[corr_cols].corr(method="spearman")
TABLES["Spearman相關矩陣"] = corr.round(3).reset_index().rename(columns={"index": "欄位"})
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_cols) for b in num_cols[i + 1:]
         if abs(corr.loc[a, b]) > 0.8]
if pairs:
    pairs = sorted(pairs, key=lambda t: -abs(t[2]))
    FINDINGS["desc"].append(f"高度相關（|ρ|>0.8）的特徵對共 {len(pairs)} 對"
                            + ("，最高的 8 對" if len(pairs) > 8 else "") + "："
                            + "、".join(f"{a}–{b}（{r:.2f}）" for a, b, r in pairs[:8])
                            + "。樹模型預測不受影響，但重要度會被兩者瓜分，解讀時要合併看；它們也是 MICE/KNN 插補的好幫手。")

# --- 圖：分佈
fig, axs = grid(len(num_cols))
for ax, c in zip(axs, num_cols):
    s = X_all[c].dropna()
    if c in disc_cols:  # 離散值：長條圖
        vc = s.value_counts().sort_index()
        ax.bar([f"{v:g}" for v in vc.index], vc.values, color=C[0], width=0.65)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=7, color="#52514e")
        ax.set_ylim(0, vc.max() * 1.35)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{c}（離散，長條圖）")
    else:
        ax.hist(s, bins=int(np.clip(np.sqrt(len(s)) * 1.5, 10, 40)), color=C[0], edgecolor="white", linewidth=0.4)
        ax.axvline(s.mean(), color=INK, lw=1, ls="--")
        ax.set_title(f"{c}" + (f"（遺失 {X_all[c].isna().mean():.1%}）" if X_all[c].isna().any() else ""))
    ax.text(0.98, 0.95, f"平均 {s.mean():.4g}\n偏態 {stats.skew(s):.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#52514e",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
fig.suptitle("數值特徵分佈（連續變數：直方圖，虛線 = 平均數；離散變數：長條圖）", fontweight="bold", y=1.0)
fig.tight_layout()
save(fig, "02_numeric_hist", "連續變數用直方圖（虛線為平均數），取值少的離散變數（如房間數、車位數）用長條圖；右上角為平均數與偏態。")

fig, axs = grid(len(num_cols), w=4.0, h=1.6)
for ax, c in zip(axs, num_cols):
    s = X_all[c].dropna()
    ax.boxplot(s, vert=False, widths=0.55, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor="#d6e6f8", edgecolor=C[0]), medianprops=dict(color=INK),
               whiskerprops=dict(color=C[0]), capprops=dict(color=C[0]),
               flierprops=dict(marker="o", markersize=3, markerfacecolor=C[1], markeredgecolor="none", alpha=.6))
    ax.set_yticks([])
    ax.set_title(f"{c}（離群值 {num_desc.loc[c, '離群值數(1.5IQR)']} 筆）")
fig.suptitle("箱形圖（橘點 = 超過 1.5×IQR）", fontweight="bold", y=1.0)
fig.tight_layout()
save(fig, "03_numeric_box", "箱形圖：橘點為 1.5×IQR 外的離群值；樹模型對離群值穩健，但明顯的輸入錯誤仍應修正。")

if cat_cols:
    fig, axs = grid(len(cat_cols), w=4.2, h=2.6)
    for ax, c in zip(axs, cat_cols):
        vc = X_all[c].value_counts()
        nmiss = X_all[c].isna().sum()
        labels, vals = list(vc.index.astype(str)), list(vc.values)
        colors = [C[0]] * len(vals)
        if nmiss:
            labels.append("（遺失）"); vals.append(nmiss); colors.append(GRAY)
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.65)
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v}", va="center", fontsize=7, color="#52514e")
        ax.set_title(c)
        ax.grid(axis="y", visible=False)
    fig.suptitle("類別特徵次數（灰 = 遺失）", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "04_categorical_bar", "類別特徵各水準次數；稀有水準（<1%）可考慮合併為「其他」。")

# 目標分佈
fig, ax = plt.subplots(figsize=(5.5, 3))
if IS_CLF:
    ax.bar([cname(k) for k in tgt_vc.index], tgt_vc.values, color=[C[0], C[1]] + C[2:len(tgt_vc)], width=0.6)
    for i, v in enumerate(tgt_vc.values):
        ax.text(i, v, f"{v}\n{v / len(y_all):.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, tgt_vc.max() * 1.2)
    ax.grid(axis="x", visible=False)
else:
    ax.hist(y_all, bins=50, color=C[0], edgecolor="white", linewidth=0.4)
    ax.axvline(y_all.median(), color=INK, ls="--", lw=1, label=f"中位數 {y_all.median():,.0f}")
    ax.axvline(y_all.mean(), color=C[1], ls=":", lw=1.5, label=f"平均數 {y_all.mean():,.0f}")
    ax.legend()
ax.set_title(f"目標變數 {target} 分佈")
save(fig, "01_target", "目標變數分佈：分類看各類別比例（是否不平衡）；迴歸看分佈形狀與偏態。")

# 特徵 vs 目標
feat_plot = num_cols + cat_cols
fig, axs = grid(len(feat_plot), w=4.0, h=2.8)
for ax, c in zip(axs, feat_plot):
    if c in num_cols and not IS_CLF and c in disc_cols:  # 迴歸 × 離散特徵：各取值的目標箱形圖
        m = X_all[c].notna()
        lv = sorted(X_all.loc[m, c].unique())
        ax.boxplot([y_all[m & (X_all[c] == v)] for v in lv], tick_labels=[f"{v:g}" for v in lv], widths=0.55,
                   showfliers=False, patch_artist=True, boxprops=dict(facecolor="#d6e6f8", edgecolor=C[0]),
                   medianprops=dict(color=INK))
        ax.set_ylabel(target)
    elif c in num_cols:
        m = X_all[c].notna()
        if IS_CLF:
            ks = sorted(y_all.unique())
            bp = ax.boxplot([X_all.loc[m & (y_all == k), c] for k in ks], tick_labels=[cname(k) for k in ks],
                            widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
            for patch, col in zip(bp["boxes"], C):
                patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(col)
            ax.set_xlabel(target)
        else:
            ax.scatter(X_all.loc[m, c], y_all[m], s=5, alpha=0.25, color=C[0], edgecolors="none")
            q = pd.qcut(X_all.loc[m, c].rank(method="first"), 12, labels=False)
            g = pd.DataFrame({"x": X_all.loc[m, c], "y": y_all[m], "q": q}).groupby("q").median()
            ax.plot(g["x"], g["y"], color=C[1], lw=2)
            ax.set_ylabel(target)
    else:
        s = X_all[c].fillna("（遺失）")
        if IS_CLF and not BINARY:
            ct = pd.crosstab(s, y_all, normalize="index")
            left = np.zeros(len(ct))
            for j, k in enumerate(ct.columns):
                ax.barh(ct.index.astype(str), ct[k].values, left=left, color=C[j % 8], height=0.6, label=cname(k))
                left += ct[k].values
            ax.set_xlabel("各類別佔比")
            ax.legend(fontsize=7)
        elif IS_CLF:
            r = y_all.groupby(s).mean().sort_values()
            ax.barh(r.index.astype(str), r.values, color=[GRAY if i == "（遺失）" else C[0] for i in r.index],
                    height=0.6)
            ax.axvline(y_all.mean(), color=INK, ls="--", lw=1)
            ax.set_xlabel(f"{target} 比率（虛線 = 整體）")
            ax.grid(axis="y", visible=False)
        else:
            order = y_all.groupby(s).median().sort_values().index
            ax.boxplot([y_all[s == k] for k in order], tick_labels=[str(k) for k in order], vert=False,
                       widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="#d6e6f8", edgecolor=C[0]), medianprops=dict(color=INK))
            ax.set_xlabel(target)
    ax.set_title(c)
fig.suptitle(f"各特徵與目標 {target} 的關係" + ("（迴歸：橘線 = 分箱中位數）" if not IS_CLF else ""),
             fontweight="bold", y=1.0)
fig.tight_layout()
save(fig, "06_feature_vs_target", "雙變量分析：分類看各類別的分佈差異或各水準的目標比率；迴歸看趨勢與非線性。")

fig, ax = plt.subplots(figsize=(0.55 * len(corr) + 2, 0.5 * len(corr) + 1.5))
im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.values[i, j]) > 0.6 else INK)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Spearman 相關係數（含目標）")
save(fig, "07_corr", "Spearman 相關（對偏態與單調非線性較穩健）：找共線性與和目標相關的特徵。")

# ============================================================================ 4. 遺失值診斷
log("\n=== 4. 遺失值檢查 ===")
miss_cols = [c for c in X_all.columns if X_all[c].isna().any()]
HAS_MISS = len(miss_cols) > 0
n_cells, n_miss_cells = X_all.size, int(X_all.isna().sum().sum())
TABLES["遺失值檢查"] = pd.DataFrame({
    "項目": ["特徵欄位數", "資料筆數", "總格數", "遺失格數", "有遺失的欄位數", "完整列比例", "判斷"],
    "結果": [X_all.shape[1], len(X_all), f"{n_cells:,}", f"{n_miss_cells:,}（{n_miss_cells / n_cells:.2%}）",
           f"{len(miss_cols)}" + (f"：{', '.join(miss_cols)}" if miss_cols else ""),
           f"{X_all.notna().all(axis=1).mean():.1%}",
           "有遺失 → 進行遺失機制診斷與插補比較" if HAS_MISS else "無遺失 → 跳過遺失值診斷與處理"],
})
FINDINGS["check"] = [
    (f"<b>有遺失值</b>：{len(miss_cols)} 個特徵欄位有遺失，共 {n_miss_cells:,} 格（佔 {n_miss_cells / n_cells:.2%}）。"
     "接下來進行遺失機制診斷（本節），並在切分資料後比較插補方法（第 6 節）。")
    if HAS_MISS else
    (f"<b>沒有遺失值</b>：{X_all.shape[1]} 個特徵 × {len(X_all):,} 筆資料都完整，"
     "<b>不需要做遺失值診斷與插補</b>，直接進入資料切分與建模。（遺失值的檢查仍必做——上線資料可能出現遺失，"
     "屆時再依本 skill 的遺失值流程處理。）")
]
log(TABLES["遺失值檢查"].to_string(index=False))
miss_tbl = overview.loc[overview["遺失數"] > 0, ["欄位", "型態", "遺失數", "遺失%"]].sort_values(
    "遺失%", ascending=False)
miss_cols = list(miss_tbl["欄位"])  # 依遺失比例由高到低（所有圖表統一此順序）
# 行（欄位）遺失比例 與 列（資料筆）遺失比例
col_ratio = overview[["欄位", "遺失數", "遺失%"]].sort_values("遺失%", ascending=False, kind="stable").rename(
    columns={"遺失%": "行遺失比例%"})
row_k = X_all.isna().sum(axis=1)
row_ratio = (row_k.value_counts().sort_index().rename_axis("該列遺失的欄位數").reset_index(name="資料筆數"))
row_ratio["佔全部資料%"] = (100 * row_ratio["資料筆數"] / len(X_all)).round(2)
row_ratio["該列遺失比例%"] = (100 * row_ratio["該列遺失的欄位數"] / X_all.shape[1]).round(1)
n_complete = int(X_all.notna().all(axis=1).sum())
if HAS_MISS:
    TABLES["行（欄位）遺失比例"] = col_ratio
    TABLES["列（資料筆）遺失比例"] = row_ratio
    log(col_ratio.to_string(index=False))
    log(row_ratio.to_string(index=False))
log(f"完整列（所有特徵都沒有遺失）: {n_complete}/{len(X_all)}（{n_complete / len(X_all):.1%}）")


def little_mcar_test(data, max_iter=200, tol=1e-6):
    """Little (1988) MCAR 檢定。H0：資料為 MCAR。p<0.05 → 拒絕 MCAR（至少有 MAR/MNAR）。
    以 EM 估計多變量常態的平均與共變異，再比較各遺失型態的平均是否偏離整體。"""
    Z = data.to_numpy(float)
    Z = (Z - np.nanmean(Z, 0)) / np.nanstd(Z, 0)
    n, p = Z.shape
    M = np.isnan(Z)
    keep = ~M.all(1)
    Z, M = Z[keep], M[keep]
    pats, inv = np.unique(M, axis=0, return_inverse=True)
    mu, S = np.nanmean(Z, 0), np.diag(np.nanvar(Z, 0))
    for _ in range(max_iter):
        sx, sxx = np.zeros(p), np.zeros((p, p))
        for k, pat in enumerate(pats):
            rows = Z[inv == k]
            o, m = ~pat, pat
            Xh = rows.copy()
            Cmm = np.zeros((p, p))
            if m.any():
                B = S[np.ix_(m, o)] @ np.linalg.pinv(S[np.ix_(o, o)])
                Xh[:, m] = mu[m] + (rows[:, o] - mu[o]) @ B.T
                Cmm[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
            sx += Xh.sum(0)
            sxx += Xh.T @ Xh + len(rows) * Cmm
        mu_new = sx / len(Z)
        S_new = sxx / len(Z) - np.outer(mu_new, mu_new)
        done = np.abs(mu_new - mu).max() < tol and np.abs(S_new - S).max() < tol
        mu, S = mu_new, S_new
        if done:
            break
    d2, dof = 0.0, 0
    for k, pat in enumerate(pats):
        o = ~pat
        rows = Z[inv == k][:, o]
        diff = rows.mean(0) - mu[o]
        d2 += len(rows) * diff @ np.linalg.pinv(S[np.ix_(o, o)]) @ diff
        dof += o.sum()
    dof -= p
    return d2, dof, stats.chi2.sf(d2, dof), len(pats)


little = None
num_miss = [c for c in num_cols if c in miss_cols]
if num_miss:
    d2, dof, p_little, n_pat = little_mcar_test(X_all[num_cols])
    little = {"chi2": round(d2, 2), "df": int(dof), "p": float(p_little), "patterns": int(n_pat)}
    log(f"Little MCAR 檢定（數值欄）: χ²={d2:.1f}, df={dof}, p={p_little:.3g}, 遺失型態數={n_pat}")
    FINDINGS["missing"].append(
        f"Little MCAR 整體檢定：χ²={d2:.1f}、df={dof}、p={p_little:.2g} → "
        + ("<b>拒絕 MCAR</b>：至少有部分欄位的遺失和其他變數有關，直接刪除遺失列會造成樣本偏誤。"
           if p_little < 0.05 else "無法拒絕 MCAR（但檢定力有限，仍需逐欄檢查）。"))


def assoc(feature, R):
    """遺失指標 R（1=遺失）與另一個變數的關聯：回傳 (效果量, 指標名, p)。"""
    if pd.api.types.is_numeric_dtype(feature):
        m = feature.notna()
        a, b = feature[m & (R == 1)], feature[m & (R == 0)]
        if len(a) < 5 or len(b) < 5:
            return np.nan, "SMD", np.nan
        sd = np.sqrt((a.var() + b.var()) / 2)
        return (a.mean() - b.mean()) / sd if sd > 0 else 0.0, "SMD", stats.mannwhitneyu(a, b).pvalue
    f = feature.fillna("（遺失）")
    ct = pd.crosstab(f, R)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, "V", np.nan
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))), "V", p


def _fmt(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


diag_rows, detail_rows = [], []
n_tests = max(1, (X_all.shape[1] - 1) * max(1, len(miss_cols)))
alpha = 0.05 / n_tests  # Bonferroni
for c in miss_cols:
    R = X_all[c].isna().astype(int)
    links = []
    for o in X_all.columns:
        if o == c:
            continue
        eff, kind, p = assoc(X_all[o], R)
        detail_rows.append({"遺失欄位": c, "比較變數": o, "效果量": eff, "指標": kind, "p值": p})
        big = abs(eff) >= (0.2 if kind == "SMD" else 0.1)
        if p < alpha and big:
            links.append((o, eff, kind))
    links.sort(key=lambda t: -abs(t[1]))
    # 遺失與目標的關係
    if IS_CLF:
        rate_m, rate_o = y_all[R == 1].mean(), y_all[R == 0].mean()
        _, p_t, _, _ = stats.chi2_contingency(pd.crosstab(R, y_all))
        tgt_eff = rate_m - rate_o
        tgt_txt = f"遺失時 {target} 率 {rate_m:.1%} vs 未遺失 {rate_o:.1%}"
        tgt_big = abs(tgt_eff) >= 0.05
    else:
        a, b = y_all[R == 1], y_all[R == 0]
        p_t = stats.mannwhitneyu(a, b).pvalue
        tgt_eff = (a.mean() - b.mean()) / y_all.std()
        tgt_txt = f"遺失時 {target} 平均 {a.mean():,.1f} vs 未遺失 {b.mean():,.1f}"
        tgt_big = abs(tgt_eff) >= 0.2
    tgt_link = p_t < 0.05 and tgt_big
    pct = X_all[c].isna().mean() * 100

    if not links and not tgt_link:
        mech = "符合 MCAR"
        why = "遺失與其他變數、目標都無明顯關聯"
        rec = ("中位數/眾數插補即可" if pct < 5 else "中位數/眾數或 KNN；刪除列會浪費資料") + \
              ("（類別：另立「遺失」類別或眾數）" if c in cat_cols else "")
    elif links:
        mech = "MAR（或 MNAR）"
        why = "遺失與 " + "、".join(f"{o}（{k}={e:.2f}）" for o, e, k in links[:3]) + " 有關"
        rec = "用其他欄位插補（KNN / MICE）；不要整列刪除"
    else:
        mech = "疑似 MNAR / 資訊性遺失"
        why = "遺失與其他特徵無關，卻與目標有關"
        rec = "加遺失指標欄位；單純插補會抹掉訊號"
    if tgt_link:
        rec += "；遺失本身與目標有關 → 務必保留遺失指標"
    if pct > 60:
        rec = "遺失 >60%：考慮刪除欄位，只保留遺失指標"
    diag_rows.append({"欄位": c, "型態": "數值" if c in num_cols else "類別", "遺失%": round(pct, 2),
                      "推測機制": mech, "依據": why, "與目標": tgt_txt, "目標關聯p值": p_t, "建議處理": rec})
    log(f"- {c}: {mech}｜{why}｜{tgt_txt} (p={p_t:.2g})")

diag = pd.DataFrame(diag_rows)
TABLES["遺失摘要"] = miss_tbl
TABLES["遺失機制診斷"] = diag
TABLES["遺失關聯明細"] = pd.DataFrame(detail_rows).round(4)
if miss_cols:
    FINDINGS["missing"][0:0] = [
        "<b>行（欄位）遺失比例</b>＝某欄位遺失的筆數 ÷ 總筆數：" + "、".join(
            f"{r['欄位']} {r['行遺失比例%']:.1f}%" for _, r in col_ratio[col_ratio["遺失數"] > 0].iterrows())
        + f"；其餘 {int((col_ratio['遺失數'] == 0).sum())} 個欄位為 0%。",
        f"<b>列（資料筆）遺失比例</b>＝某筆資料遺失的欄位數 ÷ 特徵數（{X_all.shape[1]}）：完整（0 個遺失）的資料 "
        f"{n_complete:,} 筆（{n_complete / len(X_all):.1%}）；至少遺失 1 個欄位 {len(X_all) - n_complete:,} 筆"
        f"（{1 - n_complete / len(X_all):.1%}）；單筆最多遺失 {int(row_k.max())} 個欄位（該列遺失比例 "
        f"{row_k.max() / X_all.shape[1]:.0%}）。若直接刪除含遺失的列，會丟掉 {1 - n_complete / len(X_all):.1%} 的資料。",
    ]
    FINDINGS["missing"].append("提醒：MAR 可以從資料看到跡象（遺失率隨其他欄位變化）；MNAR（遺失取決於值本身）"
                               "無法從觀測資料證明，只能靠領域知識判斷。標成「MAR（或 MNAR）」的欄位，兩種都有可能。")
else:
    for k in ["遺失摘要", "遺失機制診斷", "遺失關聯明細"]:
        TABLES.pop(k, None)

if miss_cols:
    # 遺失矩陣
    order = X_all[miss_cols].isna().sum(1).sort_values(kind="stable").index
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a1.imshow(X_all.loc[order, miss_cols].isna().values.astype(int), aspect="auto", interpolation="nearest",
              cmap=LinearSegmentedColormap.from_list("m", ["#eef3fa", C[0]]))
    a1.set_xticks(range(len(miss_cols)), miss_cols, rotation=30, ha="right")
    a1.set_ylabel("列（依遺失數排序）")
    a1.grid(False)
    a1.set_title("遺失矩陣（藍 = 遺失）")
    a2.bar(range(len(miss_tbl)), miss_tbl["遺失%"], color=C[0], width=0.6)   # 與左圖同一順序（由高到低）
    for i, v in enumerate(miss_tbl["遺失%"]):
        a2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(range(len(miss_tbl)), miss_tbl["欄位"], rotation=30, ha="right")
    a2.set_ylim(0, miss_tbl["遺失%"].max() * 1.2)
    a2.set_ylabel("遺失 %")
    a2.grid(axis="x", visible=False)
    a2.set_title("行（欄位）遺失比例")
    a3.bar(row_ratio["該列遺失的欄位數"].astype(str), row_ratio["佔全部資料%"], color=C[1], width=0.6)
    for i, (v, n_) in enumerate(zip(row_ratio["佔全部資料%"], row_ratio["資料筆數"])):
        a3.text(i, v, f"{v:.1f}%\n({n_:,})", ha="center", va="bottom", fontsize=7.5)
    a3.set_ylim(0, row_ratio["佔全部資料%"].max() * 1.25)
    a3.set_xlabel("該筆資料遺失的欄位數")
    a3.set_ylabel("佔全部資料 %")
    a3.grid(axis="x", visible=False)
    a3.set_title("列（資料筆）遺失比例")
    fig.tight_layout()
    save(fig, "08_missing_matrix", "左：遺失矩陣（欄位依遺失比例由高到低排列，與中圖同順序）；中：行（欄位）遺失比例；"
         "右：列（資料筆）遺失比例——每筆資料遺失幾個欄位、各佔多少。")


# ============================================================================ 5. 資料切分（測試集保留到最後）
log("\n=== 5. 資料切分 ===")
TEST_SIZE, K = 0.25, 5
strat = y_all if IS_CLF else None
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=strat, random_state=SEED)
CV = (StratifiedKFold if IS_CLF else KFold)(K, shuffle=True, random_state=SEED)
# 二元分類用 AUC；多類別用 Accuracy（AUC-ovr 在易分資料常接近 1，難以比較）；迴歸用 RMSE
SCORING = "roc_auc" if BINARY else ("accuracy" if IS_CLF else "neg_root_mean_squared_error")
METRIC = "AUC" if BINARY else ("Accuracy" if IS_CLF else "RMSE")
sign = 1 if IS_CLF else -1  # sklearn 的 neg_RMSE 轉回正值
fm = (lambda v: f"{v:.4f}") if IS_CLF else (lambda v: f"{v:,.1f}")


def tgt_summary(name, yy, n_total):
    d = {"集合": name, "筆數": len(yy), "佔全部%": round(100 * len(yy) / n_total, 1)}
    if IS_CLF:
        for k, v in yy.value_counts(normalize=True).sort_index().items():
            d[f"類別 {k} 佔比%"] = round(100 * v, 2)
    else:
        d.update({"平均數": yy.mean(), "中位數": yy.median(), "標準差": yy.std(), "最小值": yy.min(), "最大值": yy.max()})
    return d


split_tbl = pd.DataFrame([tgt_summary("全部資料", y_all, len(y_all)), tgt_summary("訓練集", y_tr, len(y_all)),
                          tgt_summary("測試集", y_te, len(y_all))]).round(2)
if IS_CLF:  # 各類別在訓練 / 測試各有幾筆
    rows_ = []
    for k in sorted(y_all.unique()):
        n_all, n_tr, n_te = int((y_all == k).sum()), int((y_tr == k).sum()), int((y_te == k).sum())
        rows_.append({"類別": cname(k), "全部筆數": n_all, "訓練筆數": n_tr, "測試筆數": n_te,
                      "訓練佔該類%": round(100 * n_tr / n_all, 1), "測試佔該類%": round(100 * n_te / n_all, 1),
                      "訓練集內佔比%": round(100 * n_tr / len(y_tr), 1), "測試集內佔比%": round(100 * n_te / len(y_te), 1)})
    rows_.append({"類別": "合計", "全部筆數": len(y_all), "訓練筆數": len(y_tr), "測試筆數": len(y_te),
                  "訓練佔該類%": round(100 * len(y_tr) / len(y_all), 1), "測試佔該類%": round(100 * len(y_te) / len(y_all), 1),
                  "訓練集內佔比%": 100.0, "測試集內佔比%": 100.0})
    split_tbl = pd.DataFrame(rows_)
fold_rows = []
for k, (a, b_) in enumerate(CV.split(X_tr, y_tr), 1):
    r = {"折": k, "訓練筆數": len(a), "驗證筆數": len(b_)}
    if IS_CLF:
        for kk in sorted(y_all.unique()):
            r[f"驗證折 {cname(kk)} 筆數"] = int((y_tr.iloc[b_] == kk).sum())
    else:
        r["訓練 目標平均"] = round(y_tr.iloc[a].mean(), 1)
        r["驗證 目標平均"] = round(y_tr.iloc[b_].mean(), 1)
    fold_rows.append(r)
fold_tbl = pd.DataFrame(fold_rows)
TABLES["資料切分（各類別筆數）" if IS_CLF else "資料切分"] = split_tbl
TABLES["交叉驗證各折"] = fold_tbl
log(split_tbl.to_string(index=False))
log(fold_tbl.to_string(index=False))
if IS_CLF:
    same = ("切分後各類別筆數（訓練 / 測試）：" + "、".join(
        f"{cname(k)} {int((y_tr == k).sum())} / {int((y_te == k).sum())}" for k in sorted(y_all.unique()))
        + f"；合計 {len(y_tr)} / {len(y_te)}。各類別在訓練、測試中的比例都和全體一致（stratify 的效果）。")
else:
    ks_p = stats.ks_2samp(y_tr, y_te).pvalue
    same = (f"訓練集 / 測試集的目標平均為 {y_tr.mean():,.1f} / {y_te.mean():,.1f}，中位數 {y_tr.median():,.1f} / {y_te.median():,.1f}；"
            f"KS 檢定 p={ks_p:.2f}（>0.05 代表兩者分佈沒有明顯差異）。")
FINDINGS["split"] += [
    "<b>為什麼要切分</b>：模型在「看過的資料」上的分數一定偏樂觀（尤其是會記住資料的深樹）。必須留一份完全沒碰過的資料，"
    "才能估計上線後的真實表現。",
    f"<b>本次做法</b>：先切出 {int(TEST_SIZE * 100)}% 測試集（{len(X_te):,} 筆，鎖起來直到最後），剩下 {len(X_tr):,} 筆訓練集再做 "
    f"{K} 折交叉驗證（{'StratifiedKFold' if IS_CLF else 'KFold'}）：每次用 4 折訓練、1 折驗證，輪流 5 次取平均。"
    + ("插補法與模型都用 CV 分數挑選" if HAS_MISS else "模型用 CV 分數挑選") + "，測試集只在最後報告一次。",
    ("<b>分類要 stratify</b>：讓每份資料的類別比例都和全體相同，否則少數類別可能在某一份裡特別少，分數忽高忽低。"
     if IS_CLF else
     "<b>迴歸用一般 KFold</b>：目標是連續值，沒有類別比例可維持；若目標很偏或樣本少，可以先把目標分箱再 stratify。"),
    same,
    "<b>避免洩漏</b>：插補、編碼、特徵選擇都放進 Pipeline，在每一折只用該折的訓練資料 fit；"
    "<b>不要</b>在切分前對全部資料算平均數補值或標準化。",
    "<b>其他切分情境</b>：有時間順序 → TimeSeriesSplit（只能用過去預測未來）；同一客戶/病人有多筆 → GroupKFold（同一群不能同時出現在訓練與驗證）；"
    "Boosting 的 early stopping → 再從訓練集切出一小份驗證集；資料很少 → 用重複 CV（RepeatedKFold）讓估計更穩。",
]

# 圖：切分示意 + 目標分佈比較
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
tr_w = 1 - TEST_SIZE
rows_y = [K + 2.4, K + 1.25] + list(range(K, 0, -1))
a1.barh(rows_y[0], 1, left=0, height=0.6, color="#d9d8d2")
a1.text(0.5, rows_y[0], f"全部資料 {len(y_all):,} 筆", ha="center", va="center", fontsize=9)
a1.barh(rows_y[1], tr_w, left=0, height=0.6, color=C[0])
a1.barh(rows_y[1], TEST_SIZE, left=tr_w, height=0.6, color=C[1])
a1.text(tr_w / 2, rows_y[1], f"訓練集 {len(y_tr):,}（{int(tr_w * 100)}%）", ha="center", va="center", fontsize=9, color="white")
a1.text(tr_w + TEST_SIZE / 2, rows_y[1], f"測試集 {len(y_te):,}", ha="center", va="center", fontsize=9, color="white")
fw = tr_w / K
for k in range(K):
    yk = rows_y[2 + k]
    for j in range(K):
        a1.barh(yk, fw * 0.96, left=j * fw, height=0.55, color=C[3] if j == k else "#9cc3ef")
    a1.barh(yk, TEST_SIZE, left=tr_w, height=0.55, color="#f2f1ec", hatch="///", edgecolor="#d9d8d2")
    a1.text(-0.01, yk, f"第 {k + 1} 折", ha="right", va="center", fontsize=8)
a1.text(tr_w + TEST_SIZE / 2, (rows_y[2] + rows_y[-1]) / 2, "CV 期間\n完全不碰", ha="center", va="center", fontsize=8, color="#52514e")
a1.set_xlim(-0.12, 1.01)
a1.set_ylim(0.4, K + 2.9)
a1.axis("off")
a1.set_title(f"切分示意：{int(TEST_SIZE * 100)}% 測試集 + 訓練集 {K} 折交叉驗證（黃 = 該折的驗證資料）", loc="left")
if IS_CLF:
    ks_ = sorted(y_all.unique())
    w = 0.38
    for i, (nm, yy) in enumerate([("訓練", y_tr), ("測試", y_te)]):
        vals = [int((yy == k).sum()) for k in ks_]
        xs = np.arange(len(ks_)) + (i - 0.5) * w
        a2.bar(xs, vals, width=w * 0.92, color=[C[0], C[1]][i], label=nm)
        for x, v in zip(xs, vals):
            a2.text(x, v, f"{v}", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(range(len(ks_)), [cname(k) for k in ks_])
    a2.set_ylabel("筆數")
    a2.grid(axis="x", visible=False)
    a2.set_title("各類別在訓練 / 測試的筆數（stratify）")
    a2.set_ylim(0, max(int((y_tr == k).sum()) for k in ks_) * 1.25)
else:
    bins = np.histogram_bin_edges(y_all, 40)
    a2.hist(y_tr, bins=bins, density=True, color=C[0], alpha=0.55, label="訓練")
    a2.hist(y_te, bins=bins, density=True, histtype="step", lw=1.8, color=C[1], label="測試")
    a2.set_xlabel(target)
    a2.set_yticks([])
    a2.set_title("訓練 vs 測試的目標分佈")
a2.legend(loc="upper right")
fig.tight_layout()
save(fig, "11_split", "左：切分流程——測試集先鎖起來，訓練集內輪流用其中 1 折當驗證；右：檢查切分後目標分佈是否一致。")

# ============================================================================ 6. 插補方法比較
log("\n=== 6. 遺失值處理：插補方法比較 ===" if HAS_MISS else "\n=== 6. 遺失值處理：資料無遺失，略過 ===")


def encoder():
    """類別：遺失另立「（遺失）」類別再 one-hot；數值：原樣通過（交給後面的插補器）。"""
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="（遺失）")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01))])
    return ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat, cat_cols)],
                             verbose_feature_names_out=False)


IMPUTERS = {  # 順序 = 由簡到繁（分數在 1 個標準誤內時，選排前面的）
    "中位數": lambda: [("imp", SimpleImputer(strategy="median"))],
    "平均數": lambda: [("imp", SimpleImputer(strategy="mean"))],
    "中位數+遺失指標": lambda: [("imp", SimpleImputer(strategy="median", add_indicator=True))],
    "KNN(k=5)": lambda: [("scale", StandardScaler()), ("imp", KNNImputer(n_neighbors=5))],
    "MICE(貝氏迴歸)": lambda: [("imp", IterativeImputer(BayesianRidge(), max_iter=15, random_state=SEED))],
    "MICE(ExtraTrees)": lambda: [("imp", IterativeImputer(
        ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        max_iter=5, random_state=SEED))],
}
if args.fast:
    IMPUTERS.pop("MICE(ExtraTrees)")
SIMPLE_ORDER = list(IMPUTERS)  # 越前面越簡單；分數接近時選簡單的


def prep(method):
    steps = [("enc", encoder())]
    steps += IMPUTERS[method]()
    return Pipeline(steps)


def impute_numeric(method, X_fit, X_apply=None):
    """回傳數值欄插補後（原始尺度）的 DataFrame。"""
    p = prep(method).fit(X_fit)
    Z = p.transform(X_fit if X_apply is None else X_apply)
    Z = Z[:, :len(num_cols)]
    if "scale" in p.named_steps:
        sc = p.named_steps["scale"]
        Z = Z * sc.scale_[:len(num_cols)] + sc.mean_[:len(num_cols)]
    idx = (X_fit if X_apply is None else X_apply).index
    return pd.DataFrame(Z, columns=num_cols, index=idx)


def base_model():
    # 下游比較模型固定用 Gradient Boosting（sklearn 版不吃 NaN，正好需要插補）
    return (GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED) if IS_CLF
            else GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED))


imp_rows = []
if miss_cols:
    # (A) 下游 CV：同一個模型、同一組折，只換插補法
    for mname in ["整列刪除(listwise)"] + list(IMPUTERS):
        t = time.time()
        if mname.startswith("整列刪除"):
            # 公平比較：每折只用「完整列」訓練，但在完整的驗證折上評分（預測時遇遺失只能再用中位數補）
            from sklearn.metrics import get_scorer
            scorer, scs = get_scorer(SCORING), []
            for tr_i, va_i in CV.split(X_tr, y_tr):
                Xa, ya = X_tr.iloc[tr_i], y_tr.iloc[tr_i]
                keep = Xa.notna().all(1)
                pipe = Pipeline([("p", prep("中位數")), ("m", base_model())]).fit(Xa[keep], ya[keep])
                scs.append(scorer(pipe, X_tr.iloc[va_i], y_tr.iloc[va_i]))
            sc = np.array(scs) * sign
            keep_all = X_tr.notna().all(1)
            note = f"訓練只用完整列（約 {keep_all.mean():.0%}）；驗證折含遺失列時仍得插補才能預測"
        else:
            sc = cross_val_score(Pipeline([("p", prep(mname)), ("m", base_model())]), X_tr, y_tr, cv=CV,
                                 scoring=SCORING) * sign
            note = ""
        imp_rows.append({"插補法": mname, f"CV {METRIC} 平均": sc.mean(), f"CV {METRIC} 標準差": sc.std(),
                         "耗時(秒)": round(time.time() - t, 1), "備註": note})
        log(f"  {mname:<20} CV {METRIC} = {sc.mean():.4f} ± {sc.std():.4f}")
    imp_df = pd.DataFrame(imp_rows).round(4)
    TABLES["插補比較_下游CV"] = imp_df

    cand = imp_df[imp_df["插補法"].isin(IMPUTERS)].copy()
    col_m = f"CV {METRIC} 平均"
    best_val = cand[col_m].max() if IS_CLF else cand[col_m].min()
    se = cand[f"CV {METRIC} 標準差"].median() / np.sqrt(5)
    ok = cand[(cand[col_m] >= best_val - se) if IS_CLF else (cand[col_m] <= best_val + se)]
    BEST_IMP = sorted(ok["插補法"], key=SIMPLE_ORDER.index)[0]  # 一個標準誤內選最簡單的
    raw_best = cand.loc[cand[col_m].idxmax() if IS_CLF else cand[col_m].idxmin(), "插補法"]
    log(f"  → 選用插補法: {BEST_IMP}（分數最佳 {raw_best}；在 1 個標準誤內選最簡單者）")

    # (B) 遮罩還原：在訓練集已觀測的數值格子再隨機遮 10%，看各方法還原得多準（模擬 MCAR）
    rng = np.random.default_rng(SEED)
    Xm = X_tr.copy()
    mask = pd.DataFrame(False, index=X_tr.index, columns=num_cols)
    for c in num_cols:
        obs = X_tr.index[X_tr[c].notna()]
        pick = rng.choice(obs, int(0.10 * len(obs)), replace=False)
        mask.loc[pick, c] = True
        Xm.loc[pick, c] = np.nan
    nr = {}
    for mname in IMPUTERS:
        Zi = impute_numeric(mname, Xm)
        nr[mname] = {c: np.sqrt(((Zi.loc[mask[c], c] - X_tr.loc[mask[c], c]) ** 2).mean()) / X_tr[c].std()
                     for c in num_cols}
    nrmse = pd.DataFrame(nr).T[num_miss + [c for c in num_cols if c not in num_miss]]
    nrmse["平均"] = nrmse.mean(1)
    TABLES["插補比較_遮罩還原NRMSE"] = nrmse.round(4).reset_index().rename(columns={"index": "插補法"})

    # (C) 變異縮水：插補後欄位標準差 / 原觀測標準差
    var_rows = {}
    filled = {m: impute_numeric(m, X_tr) for m in IMPUTERS}
    for mname, Zi in filled.items():
        var_rows[mname] = {c: Zi[c].std() / X_tr[c].std() for c in num_miss}
    TABLES["插補比較_標準差比"] = pd.DataFrame(var_rows).T.round(4).reset_index().rename(columns={"index": "插補法"})

    # 圖：下游 CV
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp_df) + 1))
    d = imp_df.iloc[::-1]
    cols = [C[1] if m == BEST_IMP else (GRAY if m.startswith("整列") else C[0]) for m in d["插補法"]]
    ax.barh(d["插補法"], d[col_m], xerr=d[f"CV {METRIC} 標準差"], color=cols, height=0.6,
            error_kw=dict(ecolor="#52514e", lw=1, capsize=2))
    lo = (d[col_m] - 3 * d[f"CV {METRIC} 標準差"]).min()
    hi = (d[col_m] + 2 * d[f"CV {METRIC} 標準差"]).max()
    ax.set_xlim(lo, hi)
    for i, v in enumerate(d[col_m]):
        ax.text(v, i + 0.3, fm(v), fontsize=7, color="#52514e")
    ax.set_xlabel(f"5-fold CV {METRIC}（{'越高越好' if IS_CLF else '越低越好'}；誤差棒 = 標準差）")
    ax.set_title(f"插補法對下游模型的影響（橘 = 選用：{BEST_IMP}）")
    ax.grid(axis="y", visible=False)
    save(fig, "12_impute_cv", "同一個 Gradient Boosting、同一組 CV 折，只換插補法；灰色為整列刪除（只用完整列訓練、在完整驗證折評分）。")

    # 圖：遮罩還原 NRMSE 熱圖
    fig, ax = plt.subplots(figsize=(0.95 * nrmse.shape[1] + 2.5, 0.45 * len(nrmse) + 1.4))
    im = ax.imshow(nrmse.values, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(nrmse.shape[1]), nrmse.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(nrmse)), nrmse.index)
    for i in range(nrmse.shape[0]):
        for j in range(nrmse.shape[1]):
            v = nrmse.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanpercentile(nrmse.values, 70) else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="NRMSE（越低越好）")
    ax.set_title("遮罩還原：隨機遮掉 10% 已知值，各方法還原誤差")
    save(fig, "13_impute_nrmse", "NRMSE = RMSE / 標準差；1.0 約等於用平均數猜。有相關欄位可借力的變數，KNN/MICE 會明顯較好。")

    # 圖：插補值分佈（遺失最多的 2 個數值欄）
    show_cols = sorted(num_miss, key=lambda c: -X_tr[c].isna().mean())[:2]
    meths = ["平均數", "KNN(k=5)", "MICE(貝氏迴歸)"] + (["MICE(ExtraTrees)"] if "MICE(ExtraTrees)" in IMPUTERS else [])
    if show_cols:
        fig, axs = plt.subplots(len(show_cols), len(meths), figsize=(3.3 * len(meths), 2.6 * len(show_cols)),
                                squeeze=False)
        for r, c in enumerate(show_cols):
            m = X_tr[c].isna()
            bins = np.histogram_bin_edges(X_tr[c].dropna(), 30)
            for k, mname in enumerate(meths):
                ax = axs[r, k]
                ax.hist(X_tr[c].dropna(), bins=bins, density=True, color="#d9d8d2", label="觀測值")
                vals = filled[mname].loc[m, c]
                if vals.nunique() <= 1:
                    ax.axvline(vals.iloc[0], color=C[1], lw=2, label="插補值（單一值）")
                else:
                    ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.8, color=C[1], label="插補值")
                ax.set_title(f"{c}｜{mname}", fontsize=9)
                if r == 0 and k == 0:
                    ax.legend(loc="upper right")
        fig.tight_layout()
        save(fig, "14_impute_dist", "插補值分佈 vs 觀測值：平均數插補把所有遺失擠在同一點，會壓縮變異、扭曲分佈。")

    # 文字結論
    d0 = imp_df.set_index("插補法")
    if "指標" not in BEST_IMP and diag["建議處理"].str.contains("遺失指標").any():
        FINDINGS["impute"].append(
            f"第 4 節的診斷建議部分欄位保留遺失指標；實測「中位數+遺失指標」CV {METRIC} {fm(d0.loc['中位數+遺失指標', col_m])}，"
            f"與選用的 {BEST_IMP} {fm(d0.loc[BEST_IMP, col_m])} 差距在雜訊內，代表其他特徵已能補足這個訊號，所以選較簡單的做法。")
    FINDINGS["impute"] += [
        f"下游模型（Gradient Boosting，5-fold CV {METRIC}）比較："
        + "、".join(f"{m} {fm(d0.loc[m, col_m])}" for m in d0.index) + "。",
        f"選用 <b>{BEST_IMP}</b>：分數最佳為 {raw_best}，在 1 個標準誤（≈{fm(se)}）內挑最簡單的方法，避免為了雜訊等級的差距增加複雜度。",
        f"遮罩還原（模擬 MCAR）平均 NRMSE 最低：{nrmse['平均'].idxmin()}（{nrmse['平均'].min():.3f}），"
        f"平均數插補 {nrmse.loc['平均數', '平均']:.3f}。數值還原準不等於預測好——樹模型更在乎「遺失本身」這個訊號。",
    ]
else:
    BEST_IMP = "無（資料無遺失）"

# ============================================================================ 7. 監督式學習：模型比較
IMP_M = BEST_IMP if HAS_MISS else "中位數"  # 無遺失時插補器不會動到任何值
log(f"\n=== 7. 模型比較（插補 = {BEST_IMP}） ===")
# 每個模型：(插補, 估計器, 固定參數說明, GridSearch 參數網格)
if IS_CLF:
    models = {
        "Decision Tree": (IMP_M, DecisionTreeClassifier(random_state=SEED), "criterion=gini",
                          {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 10, 20]}),
        "Random Forest": (IMP_M, RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                          "n_estimators=300、bootstrap=True",
                          {"max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 3, 5]}),
        "Gradient Boosting": (IMP_M, GradientBoostingClassifier(subsample=0.8, random_state=SEED), "subsample=0.8",
                              {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]}),
    }
else:
    models = {
        "Decision Tree": (IMP_M, DecisionTreeRegressor(random_state=SEED), "criterion=squared_error",
                          {"max_depth": [4, 6, 8, 10, 12, None], "min_samples_leaf": [1, 5, 10, 20]}),
        "Random Forest": (IMP_M, RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
                          "n_estimators=300、bootstrap=True",
                          {"max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 3, 5]}),
        "Gradient Boosting": (IMP_M, GradientBoostingRegressor(subsample=0.8, random_state=SEED), "subsample=0.8",
                              {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [3, 4, 5]}),
    }
USE_GRID = not args.no_grid
if not USE_GRID:  # 不做網格搜尋時，用網格中的「中間值」當預設參數
    DEFAULTS = {"max_depth": 5, "min_samples_leaf": 5, "max_features": "sqrt", "n_estimators": 300,
                "learning_rate": 0.05}

PARAM_DESC = {
    "max_depth": ("最大深度", "樹最多切幾層；越深越能擬合細節，但越容易過擬合（None = 不限制）"),
    "min_samples_leaf": ("葉節點最少樣本數", "每片葉子至少要有幾筆資料；越大越平滑、越不容易過擬合"),
    "max_features": ("每次分裂考慮的特徵比例", "RF 每個節點只從隨機抽出的特徵中找切點；越小樹之間越不相關（sqrt = √特徵數）"),
    "n_estimators": ("樹的數量", "集成中的樹數；RF 越多越穩定，Boosting 太多會過擬合（需搭配學習率）"),
    "learning_rate": ("學習率 η", "Boosting 每棵樹的貢獻縮放；越小越穩但需要更多樹"),
    "subsample": ("列抽樣比例", "每棵樹只用部分資料訓練（隨機梯度提升），可降低過擬合"),
}

res, fitted, param_rows = [], {}, []
for name, (imp, est, fixed_txt, grid_) in models.items():
    pipe = Pipeline([("p", prep(imp)), ("m", est)])
    n_comb = int(np.prod([len(v) for v in grid_.values()])) if grid_ else 0
    if grid_ and USE_GRID:
        gs = GridSearchCV(pipe, {f"m__{k}": v for k, v in grid_.items()}, cv=CV, scoring=SCORING, n_jobs=1)
        gs.fit(X_tr, y_tr)
        pipe = gs.best_estimator_
        cv_mean = gs.best_score_ * sign
        cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
        best_p = {k.replace("m__", ""): v for k, v in gs.best_params_.items()}
        how = f"GridSearchCV（{K} 折 CV，{n_comb} 組 × {K} 折 = {n_comb * K} 次訓練）"
    else:
        if grid_:
            best_p = {k: DEFAULTS[k] for k in grid_}
            pipe.set_params(**{f"m__{k}": v for k, v in best_p.items()})
            how = "未調參（--no-grid，使用預設值）"
        else:
            best_p, how = {}, "不需調參"
        cv_sc = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring=SCORING) * sign
        cv_mean, cv_std = cv_sc.mean(), cv_sc.std()
        pipe.fit(X_tr, y_tr)
    param_rows.append({
        "模型": name, "調參方式": how,
        "搜尋範圍": "；".join(f"{k} ∈ {v}" for k, v in grid_.items()) or "—",
        "最佳參數": "、".join(f"{k}={v}" for k, v in best_p.items()) or "—",
        "固定參數": fixed_txt + "、random_state=42",
        "前處理": ("類別 one-hot" if cat_cols else "數值直接輸入") + (f"＋遺失值插補：{imp}" if HAS_MISS else ""),
    })
    row = {"模型": name, "插補": imp if HAS_MISS else "無", f"CV {METRIC}": cv_mean, f"CV {METRIC} 標準差": cv_std}
    if IS_CLF:
        ptr, pte = pipe.predict_proba(X_tr), pipe.predict_proba(X_te)
        pred_tr, pred = ptr.argmax(1), pte.argmax(1)
        if BINARY:
            row.update({"訓練 AUC": roc_auc_score(y_tr, ptr[:, 1]), "測試 AUC": roc_auc_score(y_te, pte[:, 1]),
                        "測試 PR-AUC": average_precision_score(y_te, pte[:, 1]),
                        "測試 F1": f1_score(y_te, pred), "測試 Accuracy": accuracy_score(y_te, pred)})
        else:
            row.update({"訓練 Accuracy": accuracy_score(y_tr, pred_tr), "測試 Accuracy": accuracy_score(y_te, pred),
                        "測試 Macro-F1": f1_score(y_te, pred, average="macro"),
                        "測試 AUC(ovr)": roc_auc_score(y_te, pte, multi_class="ovr")})
    else:
        ptr, pte = pipe.predict(X_tr), pipe.predict(X_te)
        row.update({"訓練 RMSE": root_mean_squared_error(y_tr, ptr), "測試 RMSE": root_mean_squared_error(y_te, pte),
                    "測試 MAE": mean_absolute_error(y_te, pte), "測試 R²": r2_score(y_te, pte),
                    "測試 MAPE%": 100 * mean_absolute_percentage_error(y_te, pte)})
    res.append(row)
    fitted[name] = (pipe, pte)
    log(f"  {name:<24} CV {METRIC} {cv_mean:.4f}±{cv_std:.4f}｜{how}｜最佳參數 {param_rows[-1]['最佳參數']}")
res_df = pd.DataFrame(res).round(4)
param_df = pd.DataFrame(param_rows)
# 最佳值落在搜尋範圍邊界 → 提醒擴大範圍
edge = []
if USE_GRID:
    for (name, (_, _, _, grid_)), pr in zip(models.items(), param_rows):
        for k, v in grid_.items():
            nums = [x for x in v if isinstance(x, (int, float)) and x is not None]
            bp = dict(kv.split("=", 1) for kv in pr["最佳參數"].split("、")) if pr["最佳參數"] != "—" else {}
            at_edge = k in bp and bp[k] in (str(nums[0]), str(nums[-1]))
            natural_limit = (k == "min_samples_leaf" and bp.get(k) == "1")  # 1 已是可能的最小值，不必再擴大
            if len(nums) >= 3 and at_edge and not natural_limit and None not in v:
                edge.append(f"{name} 的 {k}={bp[k]}")
used = sorted({k for _, _, _, g in models.values() for k in g} | {"subsample"}, key=list(PARAM_DESC).index)
param_desc_df = pd.DataFrame([{"參數": k, "中文": PARAM_DESC[k][0], "意義與影響": PARAM_DESC[k][1]} for k in used])
TABLES["模型參數與調參"] = param_df
TABLES["參數說明"] = param_desc_df
TABLES["模型比較"] = res_df
log(param_df[["模型", "最佳參數", "固定參數"]].to_string(index=False))
log(res_df.drop(columns=["插補"]).to_string(index=False))

best_name = res_df.loc[res_df[f"CV {METRIC}"].idxmax() if IS_CLF else res_df[f"CV {METRIC}"].idxmin(), "模型"]
best_pipe, best_pred = fitted[best_name]
b = res_df.set_index("模型").loc[best_name]
FINDINGS["model"].append(
    "比較三個樹模型：<b>Decision Tree</b>（單棵樹，可解釋）、<b>Random Forest</b>（多棵深樹平均，降方差）、"
    "<b>Gradient Boosting</b>（多棵淺樹接力修正殘差，降偏差）。三者使用相同的前處理 Pipeline 與相同的 CV 折，比較才公平。")
FINDINGS["model"].append(
    ("<b>調參方式</b>：每個樹模型都用 <b>GridSearchCV</b> 在訓練集上做 "
     f"{K} 折交叉驗證，窮舉「模型參數與調參」表中的參數組合，選 CV {METRIC} 最好的一組，再用整個訓練集重新訓練；"
     "最後只在測試集評估一次。註：調參後的 CV 分數因為是「挑過的最佳值」而略偏樂觀，所以最終以測試集確認。")
    if USE_GRID else
    "<b>調參方式</b>：本次以 --no-grid 執行，未做 GridSearchCV，各模型使用表中的預設參數。")
if edge:
    FINDINGS["model"].append("⚠ 最佳參數落在搜尋範圍的邊界：" + "、".join(edge)
                             + "。代表更好的值可能在範圍外，實務上應擴大該參數的搜尋範圍再跑一次。")
FINDINGS["model"].append(
    (f"指標：AUC（排序能力，0.5=亂猜、1=完美）、PR-AUC 與 F1（以 <b>{cname(1)}</b> 為正類計算；隨機猜時 PR-AUC 等於正類比例 "
     f"{(y_all == 1).mean():.1%}）、Accuracy（閾值 0.5 下的整體正確率；類別不平衡時會誤導）。"
     f"文字標籤依字母排序編碼：{cname(0)} = 0、{cname(1)} = 1。")
    if BINARY else
    "指標（多類別）：Accuracy（整體猜對比例，各類別筆數相同時最直觀，用來調參與選模型）、Macro-F1（各類別 F1 的平均，不受類別多寡影響）、AUC(ovr)（每一類對其他類的排序能力平均）。"
    if IS_CLF else
    "指標：RMSE（誤差的平方平均再開根號，對大誤差敏感，單位同目標）、MAE（平均絕對誤差，較穩健）、R²（解釋了多少變異，1=完美、0=等同猜平均）、MAPE（平均百分比誤差）。")
if BINARY:
    gap = b["訓練 AUC"] - b["測試 AUC"]
    FINDINGS["model"] += [
        f"依 CV 選出最佳模型 <b>{best_name}</b>：CV AUC {b['CV AUC']:.3f}±{b['CV AUC 標準差']:.3f}，"
        f"測試集 AUC {b['測試 AUC']:.3f}、PR-AUC {b['測試 PR-AUC']:.3f}、F1 {b['測試 F1']:.3f}。",
        f"訓練/測試 AUC 差距 {gap:.3f}" + ("：有過擬合跡象，可加大 min_samples_leaf、降學習率或用 early stopping。" if gap > 0.08 else "：過擬合可接受。"),
        f"測試集 {len(y_te)} 筆，AUC 標準誤約 ±0.02；模型之間 <0.01 的差距不應過度解讀。",
    ]
elif IS_CLF:
    gap = b["訓練 Accuracy"] - b["測試 Accuracy"]
    n_wrong = int(round((1 - b["測試 Accuracy"]) * len(y_te)))
    FINDINGS["model"] += [
        f"依 CV 選出最佳模型 <b>{best_name}</b>：CV Accuracy {b['CV Accuracy']:.3f}±{b['CV Accuracy 標準差']:.3f}，"
        f"測試集 Accuracy {b['測試 Accuracy']:.3f}（{len(y_te)} 筆猜錯 {n_wrong} 筆）、Macro-F1 {b['測試 Macro-F1']:.3f}。",
        f"訓練/測試 Accuracy 差距 {gap:.3f}" + ("：有過擬合跡象。" if gap > 0.08 else "：過擬合可接受。"),
        f"測試集只有 {len(y_te)} 筆，猜錯 1 筆就差 {1 / len(y_te):.3f}；多個模型分數接近時應視為平手，不要過度解讀。",
    ]
else:
    gap = b["測試 RMSE"] / b["訓練 RMSE"]
    FINDINGS["model"] += [
        f"依 CV 選出最佳模型 <b>{best_name}</b>：CV RMSE {b['CV RMSE']:,.1f}±{b['CV RMSE 標準差']:,.1f}，"
        f"測試集 RMSE {b['測試 RMSE']:,.1f}、MAE {b['測試 MAE']:,.1f}、R² {b['測試 R²']:.3f}、MAPE {b['測試 MAPE%']:.1f}%"
        f"（目標標準差 {y_te.std():,.1f}；RMSE 越接近 0 越好）。",
        f"測試/訓練 RMSE 比 {gap:.2f}" + ("：訓練誤差遠小於測試誤差，有過擬合跡象。" if gap > 1.6 else "：過擬合可接受。"),
    ]

# 置換重要度（在測試集、以原始欄位為單位——包含該欄位的遺失訊號）
pi = permutation_importance(best_pipe, X_te, y_te, scoring=SCORING, n_repeats=10, random_state=SEED, n_jobs=1)
imp_s = pd.DataFrame({"特徵": X_te.columns, "重要度平均": pi.importances_mean * (1 if IS_CLF else 1),
                      "重要度標準差": pi.importances_std}).sort_values("重要度平均", ascending=False).round(5)
TABLES["置換重要度"] = imp_s
FINDINGS["model"].append("置換重要度（測試集）前五名：" + "、".join(
    f"{r['特徵']}（{r['重要度平均']:.4f}）" for _, r in imp_s.head(5).iterrows())
    + f"。數值 = 打亂該欄後 {METRIC} 變差多少；接近 0 或為負的欄位對模型幾乎沒貢獻。重要度不是因果。")

# 圖：模型比較
fig, ax = plt.subplots(figsize=(8, 0.5 * len(res_df) + 1.2))
yk = np.arange(len(res_df))
test_col = f"測試 {METRIC}"
ax.barh(yk + 0.2, res_df[f"CV {METRIC}"], height=0.38, color=C[0], label=f"CV {METRIC}（訓練集 5 折）",
        xerr=res_df[f"CV {METRIC} 標準差"], error_kw=dict(ecolor="#52514e", lw=1, capsize=2))
ax.barh(yk - 0.2, res_df[test_col], height=0.38, color=C[1], label=f"測試集 {METRIC}")
for i, (a, sd, t_) in enumerate(zip(res_df[f"CV {METRIC}"], res_df[f"CV {METRIC} 標準差"], res_df[test_col])):
    ax.text(a + sd, i + 0.2, f" {a:.3f}" if IS_CLF else f" {a:,.0f}", va="center", fontsize=7)
    ax.text(t_, i - 0.2, f" {t_:.3f}" if IS_CLF else f" {t_:,.0f}", va="center", fontsize=7)
ax.set_yticks(yk, res_df["模型"])
ax.invert_yaxis()
if IS_CLF:
    ax.set_xlim(0.45, 1.0)
ax.set_xlabel(f"{METRIC}（{'越高越好' if IS_CLF else '越低越好'}）")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
ax.grid(axis="y", visible=False)
ax.set_title("模型比較")
save(fig, "16_model_compare", "以 CV 選模型、測試集只做最終確認；誤差棒為 CV 標準差。")

if BINARY:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for k, (name, (_, pte)) in enumerate(fitted.items()):
        fpr, tpr, _ = roc_curve(y_te, pte[:, 1])
        a1.plot(fpr, tpr, lw=2 if name == best_name else 1.2, color=C[k % 8],
                label=f"{name}（{roc_auc_score(y_te, pte[:, 1]):.3f}）")
    a1.plot([0, 1], [0, 1], color="#d9d8d2", lw=1, ls="--")
    a1.set_xlabel("偽陽性率 FPR"); a1.set_ylabel("真陽性率 TPR")
    a1.legend(loc="lower right", fontsize=7)
    a1.set_title("測試集 ROC 曲線")
    cm = confusion_matrix(y_te, best_pred.argmax(1))
    a2.imshow(cm, cmap=SEQ)
    for i in range(2):
        for j in range(2):
            a2.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK)
    a2.set_xticks([0, 1], [f"預測 {cname(0)}", f"預測 {cname(1)}"]); a2.set_yticks([0, 1], [f"實際 {cname(0)}", f"實際 {cname(1)}"])
    a2.grid(False)
    a2.set_title(f"混淆矩陣：{best_name}（閾值 0.5）")
    fig.tight_layout()
    save(fig, "17_roc_cm", "ROC 看整體排序能力；混淆矩陣在閾值 0.5 下計算，實務上應依誤判成本調整閾值。")
elif IS_CLF:
    ks_ = sorted(y_all.unique())
    cm = confusion_matrix(y_te, best_pred.argmax(1), labels=ks_)
    fig, ax = plt.subplots(figsize=(1.3 * len(ks_) + 2.5, 1.1 * len(ks_) + 1.6))
    ax.imshow(cm, cmap=SEQ)
    for i in range(len(ks_)):
        for j in range(len(ks_)):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / max(cm[i].sum(), 1):.0%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontsize=9)
    ax.set_xticks(range(len(ks_)), [f"預測 {cname(k)}" for k in ks_], rotation=20)
    ax.set_yticks(range(len(ks_)), [f"實際 {cname(k)}" for k in ks_])
    ax.grid(False)
    ax.set_title(f"測試集混淆矩陣：{best_name}")
    fig.tight_layout()
    save(fig, "17_roc_cm", "對角線 = 猜對；括號為該實際類別中被預測成各類的比例（每列加總 100%），可看出哪兩類容易混淆。")
else:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.scatter(y_te, best_pred, s=8, alpha=0.4, color=C[0], edgecolors="none")
    lim = [min(y_te.min(), best_pred.min()), max(y_te.max(), best_pred.max())]
    a1.plot(lim, lim, color=C[1], lw=1.2, ls="--")
    a1.set_xlabel(f"實際 {target}"); a1.set_ylabel(f"預測 {target}")
    a1.set_title(f"預測 vs 實際：{best_name}")
    resid = y_te - best_pred
    a2.scatter(best_pred, resid, s=8, alpha=0.4, color=C[0], edgecolors="none")
    a2.axhline(0, color=C[1], lw=1.2, ls="--")
    a2.set_xlabel("預測值"); a2.set_ylabel("殘差（實際 − 預測）")
    a2.set_title("殘差圖：喇叭形 = 異質變異（考慮 log(y)）")
    fig.tight_layout()
    save(fig, "17_pred_resid", "點越貼近對角線越準；殘差隨預測值擴大代表誤差和價格成比例，可考慮對 log(y) 建模或看 MAPE。")

fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp_s) + 1))
d = imp_s.iloc[::-1]
ax.barh(d["特徵"], d["重要度平均"], xerr=d["重要度標準差"], color=C[0], height=0.6,
        error_kw=dict(ecolor="#52514e", lw=1, capsize=2))
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel(f"打亂後 {METRIC} 變差多少（測試集，10 次平均）")
ax.grid(axis="y", visible=False)
ax.set_title(f"置換重要度：{best_name}")
save(fig, "18_perm_importance", "以原始欄位為單位計算（one-hot 與遺失指標一起算），比 impurity 重要度更不易被高基數 / 雜訊欄位騙。")

# ============================================================================ 8. 輸出：Excel / HTML / JSON
log(f"\n總耗時 {time.time() - t0:.1f} 秒")
xlsx = OUT / "descriptive_stats.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    for sheet, tbl in TABLES.items():
        tbl.to_excel(xw, sheet_name=sheet[:31], index=False)
        ws = xw.sheets[sheet[:31]]
        ws.freeze_panes = "B2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.3 + 2), 60)
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)


def fig_html(key):
    if key not in FIGS:
        return ""
    p, cap = FIGS[key]
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


def tbl_html(name, max_rows=60, fmt=None):
    if name not in TABLES:
        return ""
    t = TABLES[name].head(max_rows)
    isnum = [pd.api.types.is_numeric_dtype(t[c]) for c in t.columns]

    def cell(v, n):
        if pd.isna(v):
            return '<td class="n">—</td>'
        if n:
            return f'<td class="n">{v:,.4g}</td>' if isinstance(v, (float, np.floating)) else f'<td class="n">{v:,}</td>'
        return f'<td class="t">{v}</td>'
    head = "".join(f'<th class="{"n" if n else "t"}">{c}</th>' for c, n in zip(t.columns, isnum))
    body = "".join("<tr>" + "".join(cell(v, n) for v, n in zip(r, isnum)) + "</tr>" for r in t.itertuples(index=False))
    return f'<h3>{name}</h3><div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def bullets(k):
    return "<ul>" + "".join(f"<li>{s}</li>" for s in FINDINGS[k]) + "</ul>"


title = f"EDA 與建模報告：{Path(args.csv).name}"
_secs = [
    ("s1", "資料與變數介紹", bullets("overview") + tbl_html("變數與欄位介紹")),
    ("s0", "判斷任務類型（分類 or 迴歸）", bullets("task") + tbl_html("任務類型判斷") + fig_html("01_target")),
    ("s2", "敘述性統計", bullets("desc") + tbl_html("數值敘述統計") + tbl_html("類別敘述統計") + fig_html("02_numeric_hist")
     + fig_html("03_numeric_box") + fig_html("04_categorical_bar")
     + tbl_html("類別水準目標率") + fig_html("06_feature_vs_target") + fig_html("07_corr")),
    ("s3", "遺失值檢查" + ("與診斷" if HAS_MISS else ""), bullets("check") + tbl_html("遺失值檢查")
     + ((bullets("missing") + fig_html("08_missing_matrix") + tbl_html("行（欄位）遺失比例")
         + tbl_html("列（資料筆）遺失比例") + tbl_html("遺失機制診斷")) if HAS_MISS else "")),
    ("s35", "資料切分", bullets("split") + fig_html("11_split") + tbl_html("資料切分（各類別筆數）") + tbl_html("資料切分")
     + tbl_html("交叉驗證各折")),
]
if HAS_MISS:
    _secs.append(("s4", "遺失值處理：插補方法比較", bullets("impute") + tbl_html("插補比較_下游CV") + fig_html("12_impute_cv")
                  + tbl_html("插補比較_遮罩還原NRMSE") + fig_html("13_impute_nrmse") + fig_html("14_impute_dist")
                  + tbl_html("插補比較_標準差比")))
_secs.append(("s5", f"模型訓練與比較（{'分類' if IS_CLF else '迴歸'}）", bullets("model") + tbl_html("模型參數與調參")
              + tbl_html("參數說明") + tbl_html("模型比較")
              + fig_html("16_model_compare") + fig_html("17_roc_cm") + fig_html("17_pred_resid") + fig_html("18_perm_importance")))
NAV = "<nav>" + "".join(f'<a href="#{k}">{i} {t}</a>' for i, (k, t, _) in enumerate(_secs, 1)) + "</nav>"
SECTIONS_HTML = NAV + "".join(f'<h2 id="{k}">{i}. {t}</h2>{body}' for i, (k, t, body) in enumerate(_secs, 1))
html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e8e7e2;--accent:#2a78d6;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5;--card:#22221f}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5;--card:#22221f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Noto Sans TC","PingFang TC","Microsoft JhengHei",system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:20px;margin:44px 0 10px;padding-top:12px;border-top:2px solid var(--ink)}}
h3{{font-size:15px;margin:22px 0 6px;color:var(--ink2)}}
.meta{{color:var(--ink2);font-size:13px}}
nav{{display:flex;flex-wrap:wrap;gap:6px 14px;margin:18px 0;font-size:13px}} nav a{{color:var(--accent);text-decoration:none}}
ul{{padding-left:20px}} li{{margin:4px 0}}
figure{{margin:18px 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px}}
figure img{{max-width:100%;height:auto;display:block;margin:0 auto}}
figcaption{{font-size:12.5px;color:#52514e;margin-top:6px}}
.tw{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 9px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{background:var(--card);color:var(--ink2);font-weight:600;white-space:nowrap}}
.n{{text-align:right;white-space:nowrap}} .t{{text-align:left;min-width:4em;max-width:340px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.kpi{{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--card)}}
.kpi b{{display:block;font-size:22px}} .kpi span{{font-size:12px;color:var(--ink2)}}
code{{background:var(--line);padding:1px 5px;border-radius:4px}}
</style></head><body><main>
<h1>{title}</h1>
<div class="meta">任務：{'分類' if IS_CLF else '迴歸'}（{task_sub}）｜目標：<code>{target}</code>｜產生：{time.strftime('%Y-%m-%d %H:%M')}｜遺失值：{"有，插補選用 " + BEST_IMP if HAS_MISS else "無"}｜最佳模型：{best_name}</div>
<div class="kpis">
<div class="kpi"><b>{len(df):,}</b><span>列數</span></div>
<div class="kpi"><b>{X_all.shape[1]}</b><span>特徵數（數值 {len(num_cols)} / 類別 {len(cat_cols)}）</span></div>
<div class="kpi"><b>{len(miss_cols) if HAS_MISS else "無"}</b><span>{"有遺失的欄位數" if HAS_MISS else "遺失值"}</span></div>
<div class="kpi"><b>{best_name}</b><span>最佳模型（依 CV）</span></div>
<div class="kpi"><b>{(f"{b['測試 ' + METRIC]:.3f}" if IS_CLF else f"{b['測試 R²']:.3f}")}</b><span>最佳模型測試 {METRIC if IS_CLF else 'R²'}</span></div>
</div>
{SECTIONS_HTML}
<p class="meta">所有表格另存於 descriptive_stats.xlsx；圖檔在 figures/。</p>
</main></body></html>"""
(OUT / "report.html").write_text(html, "utf-8")
(OUT / "report.txt").write_text("\n".join(_log), "utf-8")
summary = {
    "csv": args.csv, "task": task, "target": target, "rows": len(df), "num_cols": num_cols, "cat_cols": cat_cols,
    "dropped": drop_cols, "missing_cols": miss_cols, "complete_row_ratio": n_complete / len(X_all),
    "little_mcar": little, "mechanisms": dict(zip(diag["欄位"], diag["推測機制"])) if len(diag) else {},
    "has_missing": HAS_MISS, "best_imputer": BEST_IMP, "best_model": best_name,
    "best_model_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in b.items()},
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), "utf-8")
print(f"\n輸出 → {OUT}/report.html、descriptive_stats.xlsx、figures/（{len(FIGS)} 張圖）")
