"""Tree-based models 行為 demo（分類）：Decision Tree / Random Forest / Gradient Boosting。

用法:
    pip install scikit-learn pandas numpy
    python demo_tree_models.py classification_wine.csv --target cultivar

示範：模型比較 → 淺樹規則 → 深度掃描（過擬合）→ RF OOB → 特徵重要度（MDI vs permutation，
     會額外加入一個純雜訊欄位 _noise 檢驗重要度是否被騙）→ Gradient Boosting early stopping。
二元分類用 AUC，多類別用 Accuracy / Macro-F1。
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="classification_wine.csv")
ap.add_argument("--target", default="cultivar")
args = ap.parse_args()

# ------------------------------------------------------------------ 1. 資料
df = pd.read_csv(args.csv)
y_raw = df[args.target]
classes = sorted(y_raw.unique())
y = y_raw.map({c: i for i, c in enumerate(classes)})
X = pd.get_dummies(df.drop(columns=[args.target]), dtype=float)
rng = np.random.default_rng(SEED)
X["_noise"] = rng.normal(0, 1, len(X))          # 純雜訊欄位：用來檢驗重要度
BINARY = len(classes) == 2
SCORING = "roc_auc" if BINARY else "accuracy"
MET = "AUC" if BINARY else "Accuracy"

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"資料: {len(X)} 筆, {X.shape[1]} 個特徵（含純雜訊欄位 _noise）, 類別: {classes}")
print(f"訓練 {len(X_tr)} / 測試 {len(X_te)}（stratified）；評估指標: {MET}\n")


def score(model, Xa, ya):
    if BINARY:
        return roc_auc_score(ya, model.predict_proba(Xa)[:, 1])
    return accuracy_score(ya, model.predict(Xa))


# ------------------------------------------------------------------ 2. 模型比較
models = {
    "Decision Tree (不限深度)": DecisionTreeClassifier(random_state=SEED),
    "Decision Tree (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2,
                                                    random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = {}
for name, m in models.items():
    cv_s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring=SCORING).mean()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    rows[name] = {f"train_{MET}": score(m, X_tr, y_tr), f"cv_{MET}": cv_s, f"test_{MET}": score(m, X_te, y_te),
                  "test_macroF1": f1_score(y_te, pred, average="macro")}
res = pd.DataFrame(rows).T.round(3)
print(f"=== 模型比較（train 與 test 差距越大 = 越過擬合） ===\n{res.to_string()}\n")

# ------------------------------------------------------------------ 3. 淺樹規則
print("=== 淺層決策樹的可讀規則 (depth=3) ===")
print(export_text(models["Decision Tree (depth=3)"], feature_names=list(X.columns)))

# ------------------------------------------------------------------ 4. 深度掃描
sweep = []
for d in [1, 2, 3, 4, 5, 6, 8, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
    sweep.append({"max_depth": "None" if d is None else d, f"train_{MET}": score(t, X_tr, y_tr),
                  f"cv_{MET}": cross_val_score(t, X_tr, y_tr, cv=cv, scoring=SCORING).mean(),
                  f"test_{MET}": score(t, X_te, y_te), "n_leaves": t.get_n_leaves()})
print("=== 決策樹深度 vs 過擬合（bias-variance） ===")
print(pd.DataFrame(sweep).round(3).to_string(index=False), "\n")

# ------------------------------------------------------------------ 5. RF OOB
rf_oob = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
print(f"=== Random Forest OOB accuracy = {rf_oob.oob_score_:.3f}（測試 accuracy "
      f"{accuracy_score(y_te, rf_oob.predict(X_te)):.3f}；OOB 免額外驗證集）\n")

# ------------------------------------------------------------------ 6. 特徵重要度
rf = models["Random Forest"]
mdi = pd.Series(rf.feature_importances_, index=X.columns)
perm = permutation_importance(rf, X_te, y_te, scoring=SCORING, n_repeats=20, random_state=SEED)
perm = pd.Series(perm.importances_mean, index=X.columns)
tbl = pd.DataFrame({"RF_MDI": mdi, f"RF_permutation_{MET}_drop": perm}).sort_values("RF_MDI", ascending=False).round(4)
print("=== 特徵重要度：impurity(MDI) vs permutation（在測試集） ===")
print(tbl.to_string())
print(f"\n_noise -> MDI {mdi['_noise']:.4f}, permutation {perm['_noise']:.4f}（純雜訊欄位；permutation 應接近 0）\n")

# ------------------------------------------------------------------ 7. Early stopping
# sklearn GradientBoosting：validation_fraction 從訓練集切出驗證集，n_iter_no_change 輪沒進步就停
es = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.03, max_depth=2, validation_fraction=0.2,
                                n_iter_no_change=50, random_state=SEED).fit(X_tr, y_tr)
print("=== Gradient Boosting early stopping（用驗證集自動決定樹的數量） ===")
print(f"最佳樹數 = {es.n_estimators_}（上限 2000），測試 {MET} = {score(es, X_te, y_te):.3f}")
