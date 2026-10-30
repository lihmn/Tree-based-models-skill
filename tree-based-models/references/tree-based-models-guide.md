# 監督式學習：樹模型指南（Decision Tree、Random Forest、Gradient Boosting）

涵蓋：本 skill 使用的三個樹模型——**Decision Tree、Random Forest、Gradient Boosting**（皆為 scikit-learn 實作）。
適用：表格型（tabular）資料的分類與迴歸。
建模前的變數介紹、任務判斷、EDA、遺失值與資料切分見 `eda-missing-data-guide.md`。
本文的程式碼與 demo 數字以 scikit-learn 1.9 執行驗證。

---

## 0. 全景圖：三個模型的關係

```
Decision Tree（單棵樹，可解釋，但高方差）
   │
   ├── 並行、降方差 ──► Random Forest（bootstrap 抽樣 + 每次分裂只看部分特徵，多棵深樹平均）
   │
   └── 序列、降偏差 ──► Gradient Boosting（多棵淺樹依序擬合前面模型的殘差／負梯度）
```

| 模型 | 核心想法 | 主要解決的問題 | 樹的形態 |
|---|---|---|---|
| Decision Tree | 用一連串 if-else 把資料切成同質區塊 | 可解釋、非線性、免特徵縮放 | 1 棵，可深可淺 |
| Random Forest | 對資料做 bootstrap 抽樣（bagging），每次分裂只看隨機特徵子集，訓練多棵樹後平均／投票 | 降**方差**（variance） | 多棵、深、彼此獨立 |
| Gradient Boosting | 後一棵樹擬合前面模型的負梯度（迴歸 MSE 時就是殘差） | 降**偏差**（bias） | 多棵、淺、依序 |

**一句話記憶**：Random Forest 是「很多個過擬合的專家投票」；Gradient Boosting 是「很多個弱專家接力補錯」。

---

## 1. Decision Tree（決策樹）

### 1.1 直覺
每個內部節點問一個關於單一特徵的問題（例如 `color_intensity <= 3.82?`），資料分往左右子節點；葉節點給出預測（分類：各類別比例，取最多者；迴歸：該區塊 y 的平均）。模型本質是**分段常數函數**。

### 1.2 如何選分裂（CART，貪婪式）
對每個節點，遍歷「每個特徵 × 每個候選切點」，選出讓**不純度下降最多**的那一個（Breiman et al., 1984）。

分類不純度（節點中類別比例為 p_k）：
- **Gini**：`G = 1 − Σ p_k²`（sklearn 預設，`criterion="gini"`）
- **Entropy**：`H = − Σ p_k log₂ p_k`（`criterion="entropy"`）

迴歸不純度：**平方誤差**（節點內 y 的變異數，`criterion="squared_error"`，預設），葉值為平均。

分裂增益：
```
Gain = Impurity(parent) − [ n_L/n · Impurity(left) + n_R/n · Impurity(right) ]
```
> 貪婪 = 每步只選當下最好，不保證全域最佳樹。

### 1.3 過擬合與剪枝
不限制的樹會一路長到每片葉子都純，**訓練分數接近完美但測試較差**。葡萄酒 demo：不限深度的樹訓練 Accuracy 1.000、CV 0.917；限深度 3 時 CV 反而升到 0.925。

| 控制方式 | 參數（sklearn） | 說明 |
|---|---|---|
| 預剪枝：限深度 | `max_depth` | 最直接；常從 2–10 試 |
| 預剪枝：葉最小樣本 | `min_samples_leaf` | 防止葉子只含幾筆；越大越平滑 |
| 預剪枝：分裂最小樣本 | `min_samples_split` | 太小的節點不再分 |
| 預剪枝：最小增益 | `min_impurity_decrease` | 增益太小不分 |
| 預剪枝：葉數上限 | `max_leaf_nodes` | 限制複雜度 |
| 後剪枝 | `ccp_alpha` | Cost-Complexity Pruning；先長滿再剪，用 `cost_complexity_pruning_path` 找候選 α，以 CV 挑 |

本 skill 的 `eda_pipeline.py` 用 GridSearchCV 搜 `max_depth` × `min_samples_leaf`。

### 1.4 實務性質
- **不需要特徵標準化**（只看大小順序）。
- **類別特徵**：sklearn 樹需要先編碼（本 skill 用 one-hot）。
- **遺失值**：sklearn 1.3 起 `DecisionTree*` 可直接接受 NaN（1.4 起 `RandomForest*` 也可以），但 `GradientBoosting*` 不行。本 skill 為了三個模型一致，**一律在 Pipeline 內先插補**（見 eda guide §4）。
- **對資料微小變動敏感**：訓練集少幾筆，樹的結構可能大改 → 高方差，這也是要用集成的原因。
- **無法外推**：迴歸樹預測值永遠落在訓練 y 的範圍內。
- **偏好高基數／連續特徵**：impurity 式重要度會偏向取值很多的特徵（見 §2.5）。

### 1.5 優缺點
| 優點 | 缺點 |
|---|---|
| 可視覺化、可轉成規則、易向非技術人員說明 | 單棵樹準確度通常不高 |
| 免縮放、捕捉非線性／交互作用 | 高方差、不穩定 |
| 訓練與預測快 | 軸對齊切分，對斜向邊界不佳 |

### 1.6 程式碼
```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(export_text(tree, feature_names=list(X_train.columns)))   # 文字規則

# 後剪枝：用 CV 選 ccp_alpha
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
# 對 path.ccp_alphas 的每個值做 cross_val_score，選最佳
```

---

## 2. Random Forest（隨機森林）

### 2.1 基礎：Bagging（Bootstrap Aggregating）
Random Forest 建立在 bagging 之上（Breiman, 1996）：
1. 重複 B 次：從訓練集**有放回**抽 n 筆（bootstrap sample），訓練一棵不剪枝的深樹。
2. 預測：分類用多數決（sklearn 實際是平均各樹的類別機率），迴歸用平均。

為什麼有效——**降方差**：B 個預測的平均，若每個預測方差為 σ²、兩兩相關係數為 ρ：
```
Var(平均) = ρ·σ² + (1 − ρ)·σ² / B
```
- B 增加只能消掉第二項；**ρ 越小，降方差效果越好**。
- Bagging 幾乎不改變偏差，適合「低偏差、高方差」的深樹。

### 2.2 Random Forest 的關鍵：特徵抽樣
Breiman (2001) 在 bagging 之外，讓每個**節點分裂時**只從隨機抽出的 `max_features` 個特徵中找最佳切分，迫使樹彼此不同（降低 ρ）。
sklearn 預設：分類 `max_features="sqrt"`；迴歸 `max_features=1.0`（全部特徵，等同 bagging，值得調）。

### 2.3 OOB（Out-Of-Bag）
每棵樹的 bootstrap 約含 **63.2%** 的相異樣本（1 − (1 − 1/n)ⁿ → 1 − e⁻¹），其餘約 **36.8%** 是 OOB。用每筆樣本「沒參與訓練的那些樹」預測它，即可得到泛化誤差估計（`oob_score=True`）。葡萄酒 demo：OOB accuracy 0.977，與 CV 0.985 接近。

### 2.4 重要超參數
| 參數 | 作用 | 建議起點／調法 |
|---|---|---|
| `n_estimators` | 樹的數量 | 200–500；越多越穩，只增加時間，**不會**因此過擬合 |
| `max_features` | 每次分裂可看的特徵比例 | 分類 `"sqrt"`；可試 0.3–1.0 |
| `min_samples_leaf` | 葉最小樣本 | 1（預設）→ 3–10 通常更平滑 |
| `max_depth` | 樹深度上限 | 預設不限；資料雜訊大時限制 |
| `class_weight` | 類別不平衡 | `"balanced"` 或 `"balanced_subsample"` |
| `oob_score` | 啟用 OOB 估計 | 快速評估，不取代測試集 |
| `n_jobs` | 並行 | `-1` |

本 skill 固定 `n_estimators=300`，用 GridSearchCV 搜 `max_features` × `min_samples_leaf`。

### 2.5 特徵重要度的陷阱
1. **Impurity-based（MDI，`feature_importances_`）**：用訓練資料計算，**偏向連續／高基數特徵**，對純雜訊也可能給不低的分數（Strobl et al., 2007）。
   葡萄酒 demo（人為加入純雜訊欄位 `_noise`）：`_noise` 的 MDI = 0.006（仍大於 0），permutation = 0.000；`od280_od315` 的 MDI 排第 5（0.102），permutation 卻是 0——與其他酚類特徵相關，打亂後由其他欄位補上。
2. **Permutation importance**（`sklearn.inspection.permutation_importance`）：在**測試集**上打亂某特徵，看指標掉多少，較可靠；特徵高度相關時重要度會被分攤。本 skill 的報告用這個方法（以原始欄位為單位）。
3. 需要單筆預測的解釋 → 可用 SHAP（`shap.TreeExplainer`）。

### 2.6 優缺點
- 優點：幾乎不需調參就有不錯的表現、對雜訊與異常值穩健、可並行、提供 OOB。
- 缺點：模型大、預測較慢、可解釋性低於單棵樹、通常略遜於調好的 boosting、無法外推。

### 2.7 程式碼
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=1,
                            oob_score=True, n_jobs=-1, random_state=42).fit(X_train, y_train)
print("OOB acc:", rf.oob_score_)
```

---

## 3. Gradient Boosting（梯度提升）

### 3.1 核心想法：函數空間的梯度下降
建立加法模型（Friedman, 2001）：
```
F_m(x) = F_{m−1}(x) + η · h_m(x)
```
h_m 是一棵**淺樹**，η 是學習率（shrinkage）。每一步，h_m 擬合損失函數對當前預測的**負梯度**（pseudo-residual）：
```
r_i = − ∂L(y_i, F(x_i)) / ∂F(x_i)  │ F = F_{m−1}
```

| 任務 | 損失 L | 負梯度 r_i（直覺） |
|---|---|---|
| 迴歸（squared_error） | ½(y − F)² | **殘差** y − F |
| 二元分類（log loss） | −[y log p + (1−y) log(1−p)]，p = σ(F) | **y − p** |
| 多類別分類（K 類） | 多項 log loss | 每一類各一棵樹擬合 y_k − p_k（每輪建 K 棵） |
| 迴歸（absolute_error / huber） | 絕對誤差 / Huber | 殘差的符號 / 截斷殘差（對離群值較穩） |

### 3.2 演算法步驟（二元分類）
1. 初始化：F₀ = 訓練集 log-odds（常數）。
2. 對 m = 1…M：計算 r_i = y_i − p_i → 用淺樹擬合 r_i → 決定各葉輸出值 → F_m = F_{m−1} + η · h_m。
3. 最終預測：p = σ(F_M)。

### 3.3 與 Random Forest 的對比
- 樹**依序**訓練、每棵樹**淺**（max_depth 通常 2–5）→ 單棵偏差高、方差低，靠累加降低偏差。
- **樹太多（M 太大）會過擬合**——與 RF 的 n_estimators 不同。

### 3.4 超參數與取捨
| 參數 | 說明 | 經驗法則 |
|---|---|---|
| `learning_rate` (η) | 每棵樹的貢獻縮放 | 0.01–0.1；越小越需更多樹，通常泛化較好 |
| `n_estimators` (M) | 樹的數量 | 與 η 成反比；可用 early stopping 決定 |
| `max_depth` | 單棵樹複雜度（交互作用階數） | 2–5（sklearn 預設 3） |
| `subsample` | 每棵樹用的樣本比例（< 1 即 Stochastic GB，Friedman, 2002） | 0.5–0.9，降方差、加速 |
| `min_samples_leaf` | 葉最小樣本 | 提高可抗雜訊 |
| `max_features` | 每次分裂的特徵比例 | 加入類似 RF 的隨機性 |

**η × M 的取捨**：固定 η·M 大致維持總學習量；η 小 + M 大 → 更精細、更穩，但更慢。
本 skill 固定 `subsample=0.8`，用 GridSearchCV 搜 `n_estimators` × `learning_rate` × `max_depth`。

### 3.5 Early stopping（sklearn GradientBoosting）
設 `n_iter_no_change=N` 與 `validation_fraction`，sklearn 會從訓練集切出驗證集，驗證分數連續 N 輪沒進步就停；實際樹數在 `n_estimators_`。葡萄酒 demo：上限 2000 棵，289 棵停下。

### 3.6 程式碼
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.03, max_depth=2, subsample=0.8,
                                validation_fraction=0.2, n_iter_no_change=50, random_state=42)
gb.fit(X_train_imputed, y_train)          # sklearn GradientBoosting 不接受 NaN，需先插補
print("實際樹數:", gb.n_estimators_)
```

---

## 4. 三個模型比較

| 面向 | Decision Tree | Random Forest | Gradient Boosting |
|---|---|---|---|
| 主要降低 | — | 方差 | 偏差 |
| 訓練方式 | 單棵 | 並行（多棵獨立） | 序列（依序修正） |
| 單棵樹 | 深或淺 | 深 | 淺 |
| 過擬合風險來源 | 樹太深 | 低（加樹不會更糟） | 樹太多／η 太大／樹太深 |
| 調參敏感度 | 中 | 低 | 高 |
| 可解釋性 | 最高（可轉規則） | 低 | 低 |
| 遺失值（sklearn） | 1.3 起可接受 NaN | 1.4 起可接受 NaN | 需先插補 |
| 典型用途 | 說明規則、建立直覺 | 快速且穩定的第一個模型 | 追求較高準確度 |

### 選用建議
- 要**向人解釋規則** → 淺層 Decision Tree。
- 要**快速、穩定、少調參** → Random Forest。
- 要**較高準確度**且願意調參 → Gradient Boosting（搭配 GridSearchCV 或 early stopping）。
- 資料很少或關係近似線性、需要外推 → 先試線性模型，樹模型不一定較好。
- 影像／文字／序列 → 通常用深度學習；樹模型適合表格資料。

---

## 5. 標準工作流程

1. **資料與變數介紹**：資料來源、要預測什麼、用了哪些變數（eda guide §0）。
2. **判斷任務類型**：目標是類別 → 分類；連續數值 → 迴歸。它決定 `Classifier`/`Regressor`、指標與切分方式（eda guide §1）。
3. **EDA**：敘述統計、分佈、與目標的關係、共線性（eda guide §2–3）。
4. **遺失值**：先檢查有沒有；有才診斷機制與比較插補（eda guide §4）。
5. **切分資料**：先切出測試集鎖起來，訓練集內做 K 折 CV；分類 `stratify=y` / `StratifiedKFold`，迴歸 `KFold`；時間序列用 `TimeSeriesSplit`，同一群多筆用 `GroupKFold`（eda guide §5）。
6. **檢查資料洩漏**：目標衍生欄位、未來資訊、ID 類欄位；插補／編碼只在訓練折 fit（Pipeline）。
7. **訓練與調參**：三個模型各用 GridSearchCV（5 折）搜參數；最佳值落在範圍邊界時擴大範圍。
8. **比較與評估**：看 CV 平均 ± 標準差與訓練／測試落差；測試集只用一次。
9. **門檻與校準**（二元分類）：依成本調整閾值，必要時 `CalibratedClassifierCV`。
10. **解釋**：permutation importance、部分依賴圖（PDP）、SHAP。

---

## 6. 常見陷阱 checklist

- [ ] 只報訓練分數 → 一定要報 CV / 測試。
- [ ] 在全部資料上先插補／標準化／特徵選擇才切分（洩漏）。
- [ ] 用 `feature_importances_`（MDI）下結論 → 改用 permutation 交叉確認。
- [ ] 類別不平衡只看 accuracy。
- [ ] 把 ID、時間戳記當特徵導致記憶化。
- [ ] 時間序列隨機切分。
- [ ] Gradient Boosting 樹太多、η 太大又沒有用 CV 或 early stopping 控制。
- [ ] 用測試集反覆調參（測試集變成驗證集）。
- [ ] 期待迴歸樹外推。
- [ ] 小測試集上把 1–2 筆的差距當成模型優劣。

---

## 7. Demo 實測結果

### 7.1 完整流程：迴歸與分類各一筆（`scripts/eda_pipeline.py`）

完整數字、搜尋範圍與參數說明見 `eda-missing-data-guide.md` §7。

**迴歸 `regression_house_price.csv`**（預測成交總價，萬元；10 個數值特徵；有遺失 → 插補選中位數；隨機 75/25 + KFold(5)；GridSearchCV）

| 模型 | 最佳參數 | CV RMSE | 訓練 RMSE | 測試 RMSE | 測試 R² | 測試 MAPE |
|---|---|---|---|---|---|---|
| Decision Tree | max_depth=12、min_samples_leaf=10 | 276 | 191 | 274 | 0.78 | 18.1% |
| Random Forest | max_features=1.0、min_samples_leaf=1 | 217 | 78 | 217 | 0.86 | 14.4% |
| Gradient Boosting | n_estimators=400、learning_rate=0.05、max_depth=3 | **195** | 124 | **191** | **0.89** | **12.0%** |

**分類 `classification_wine.csv`**（預測葡萄酒品種，3 類；13 個數值特徵；無遺失；stratified 75/25 + StratifiedKFold(5)；GridSearchCV）

| 模型 | 最佳參數 | CV Accuracy | 訓練 Accuracy | 測試 Accuracy | 測試 Macro-F1 |
|---|---|---|---|---|---|
| Decision Tree | max_depth=3、min_samples_leaf=1 | 0.910 | 0.992 | 0.956 | 0.957 |
| Random Forest | max_features=sqrt、min_samples_leaf=1 | **0.985** | 1.000 | **1.000** | **1.000** |
| Gradient Boosting | n_estimators=300、learning_rate=0.1、max_depth=2 | 0.978 | 1.000 | 0.978 | 0.980 |

房價：Random Forest、Gradient Boosting 明顯優於單棵樹，Gradient Boosting 最好。葡萄酒：集成模型（CV 0.98）明顯優於單棵樹（CV 0.91），Random Forest 依 CV 最佳；測試集只有 45 筆（錯 1 筆差 0.022），測試 1.000 不宜過度解讀。

### 7.2 樹模型行為細節（葡萄酒，`scripts/demo_tree_models.py`）

設定：stratified 75/25（訓練 133 / 測試 45）、`random_state=42`；腳本額外加入純雜訊欄位 `_noise`。此處使用固定參數（未調參），所以數字與 7.1 略有不同。完整輸出見 `references/demo_output.txt`。

| 模型 | train Acc | CV Acc | test Acc | test Macro-F1 |
|---|---|---|---|---|
| Decision Tree（不限深度） | 1.000 | 0.917 | 0.911 | 0.916 |
| Decision Tree（depth=3） | 0.992 | 0.925 | 0.956 | 0.956 |
| Random Forest（300 棵） | 1.000 | **0.985** | **1.000** | **1.000** |
| Gradient Boosting（100 棵、η=0.05、depth=2） | 1.000 | 0.970 | 0.956 | 0.956 |

重點：
1. 深度掃描：depth 1 → 2 → 3 時 CV Accuracy 由 0.639 → 0.887 → 0.925；depth ≥ 4 訓練已達 1.000，CV 反而降到 0.917 → 再加深只是在背訓練資料。
2. 淺樹規則：第一刀 `color_intensity ≤ 3.82`（色澤淡的多為 cultivar_2）；色澤深的再依 `flavanoids ≤ 1.58`（類黃酮低 → 多為 cultivar_3）與 `proline > 724.5`（脯胺酸高 → cultivar_1）區分。
3. 特徵重要度：MDI 前三為 `color_intensity`（0.180）、`proline`（0.162）、`flavanoids`（0.135）；permutation 前三為 `proline`（0.057）、`color_intensity`（0.051）、`hue`（0.023）。`_noise` MDI 0.006、permutation 0.000。
4. RF OOB accuracy 0.977，與 CV 0.985 接近。
5. Gradient Boosting early stopping：上限 2000 棵，289 棵停下，測試 Accuracy 1.000。

> 測試集 45 筆，錯 1 筆 Accuracy 就差 0.022，模型間 0.02–0.03 以內的差距不具統計意義；比較模型應看 CV 平均與變異。

---

## 8. 參考資料

### 8.1 內容參考（方法與理論）
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth. —— 決策樹（CART）、Gini、剪枝（§1）。
- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140. —— Bagging、降方差（§2.1）。
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. —— 隨機森林、特徵抽樣、OOB（§2.2–2.3）。
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. —— 梯度提升、負梯度擬合（§3.1–3.2）。
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378. —— `subsample`（§3.4）。
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. —— 偏差／方差、樹與集成方法、交叉驗證總論。
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25. —— MDI 重要度偏誤（§2.5）。
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830；以及 scikit-learn 使用者指南（Decision Trees、Ensembles、Permutation feature importance、Cross-validation、GridSearchCV）—— 參數名稱、預設值、NaN 支援版本（本文以 scikit-learn 1.9 驗證）。

### 8.2 資料來源
- **葡萄酒 `classification_wine.csv`**：Aeberhard, S., & Forina, M. (1992). *Wine* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J。本 skill 取自 scikit-learn 內建的 `sklearn.datasets.load_wine()`（UCI Wine 資料集的複本），由 `scripts/make_demo_data.py` 改成易讀的欄位名稱與品種標籤後存檔。
- **房價 `regression_house_price.csv`**：**模擬資料**，非真實成交紀錄。由 `scripts/make_demo_data.py` 以固定亂數種子（seed = 11）產生：價格由面積、屋齡、距市中心、距捷運、樓層與車位依設定的公式加上隨機誤差生成，再依 MCAR / MAR / MNAR 機制注入遺失值。
