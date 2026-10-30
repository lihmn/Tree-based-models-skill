---
name: tree-based-models
description: 協助表格資料從資料與變數介紹、判斷任務類型（分類或迴歸）、探索式資料分析（EDA）、遺失值檢查與插補、資料切分，到用 Decision Tree、Random Forest、Gradient Boosting 做監督式學習（GridSearchCV 調參、模型比較、特徵重要度）的完整流程，並輸出敘述性統計報表與圖。當使用者提到判斷分類或迴歸、EDA、探索式資料分析、敘述統計、遺失值/缺失值、插補/補值、資料切分/交叉驗證、決策樹、隨機森林、梯度提升、GridSearch 調參、feature importance、或要「用樹模型」建模時使用。Use for EDA, missing-data imputation and tree-based supervised learning (Decision Tree, Random Forest, Gradient Boosting) on tabular data.
---

# Tree-Based Models Skill

協助使用者以樹模型完成表格資料的**分類 / 迴歸**，固定流程：

**① 資料與變數介紹 → ② 判斷任務類型（分類 or 迴歸）→ ③ 敘述性統計 → ④ 遺失值檢查（沒有遺失就跳過診斷與處理）→ ⑤ 資料切分 →（有遺失才）⑥ 插補比較 → ⑦ 模型訓練與比較（Decision Tree、Random Forest、Gradient Boosting，各自 GridSearchCV）**

| 需要什麼 | 去哪裡 |
|---|---|
| 一支跑完 ①–⑦ 並輸出 HTML / Excel / PNG 報表 | `scripts/eda_pipeline.py` |
| 變數介紹、任務判斷、EDA、遺失值、切分的說明與兩筆 demo 完整結果 | `references/eda-missing-data-guide.md` |
| 三個樹模型的原理、公式、參數、陷阱與 demo | `references/tree-based-models-guide.md` |
| 樹模型行為小 demo（過擬合、OOB、重要度、early stopping） | `scripts/demo_tree_models.py` |
| 兩筆 demo 資料與欄位說明 | `assets/` |

## 何時使用 / 不使用

- 使用：表格資料、數值與類別混合特徵、有遺失值、非線性或交互作用、需要可解釋規則或較高準確度。
- 先考慮別的方法：影像／文字／音訊（深度學習）、樣本極少、需要外推的趨勢預測（線性／時間序列模型）。

## 開工前先確認（缺了就問，一次問完）

1. 目標欄位是哪個？（分類／迴歸可由目標自動判斷，見「第二步」；與業務目的衝突時以使用者為準）
2. 有沒有欄位說明？（沒有就向使用者詢問，或依欄位名稱推測並標明是推測）
3. 資料是否有時間順序或同一對象多筆？（有 → 時間切分或 GroupKFold，不可隨機切）
4. 需要「可解釋」還是「準」？（決定是否以淺層決策樹為主）

## 第一步：介紹資料與變數

分析前一定先講清楚：
1. **資料說明**：資料從哪來、幾筆、每筆代表什麼。
2. **分析目的**：要預測什麼（目標欄位與其意義；分類列出各類別與筆數，迴歸列出單位、平均與範圍）。
3. **用了哪些變數**：每個特徵的名稱、中文說明與單位（數值／類別各幾個），哪些是 ID（要排除）。

`eda_pipeline.py` 報告第 1 節會自動寫出這三點，並產生「變數與欄位介紹」表（欄位、說明、單位、角色、變數型態、dtype、遺失數）。說明與單位來自欄位說明檔 `--dict`（CSV：`欄位,說明,單位,角色`；`欄位 = __dataset__` 那一列寫整份資料的描述），未指定時自動讀同資料夾的 `<檔名>_dictionary.csv`。

## 第二步：判斷是分類還是迴歸

任務類型決定模型（`*Classifier` / `*Regressor`）、評估指標與切分方式，所以一定最先確定，並向使用者說明判斷依據。

```
目標是文字 / 布林？            ──► 分類
數值但只有 2 種值（0/1）？      ──► 二元分類
整數且 ≤ 10 種值？             ──► 多類別分類（⚠ 若其實是計數或 1–5 星評分，也可當迴歸 → 問使用者）
其餘（小數、很多不同值）        ──► 迴歸
```

| | 二元分類 | 多類別分類 | 迴歸 |
|---|---|---|---|
| 預測什麼 | 是／否 | 屬於哪一類 | 數值是多少 |
| 調參與選模型的指標 | AUC | Accuracy | RMSE |
| 其他報告指標 | PR-AUC、F1、Accuracy、ROC、混淆矩陣 | Macro-F1、AUC(ovr)、混淆矩陣 | MAE、R²、MAPE、預測 vs 實際、殘差圖 |
| 切分 | `stratify=y`、`StratifiedKFold` | 同左 | 一般 `KFold` |

易錯：郵遞區號、方案代碼等「數字代碼」是類別；來電次數等「整數」是數量；把連續值切門檻（如逾期 > 30 天）就變分類——由業務需求決定。`--task auto`（預設）依上述規則判斷並在報告第 2 節列出依據；需要時用 `--task classification|regression` 覆寫。

## 探索式資料分析、遺失值處理與建模（一鍵執行）

優先用腳本產生報告，再依報告內容解說；**不要憑記憶寫數字**。

```bash
pip install scikit-learn pandas numpy scipy matplotlib openpyxl
python scripts/eda_pipeline.py data.csv --target <目標欄位> [--dict 欄位說明.csv] [--task classification|regression] [--drop 欄位...] [--out 資料夾] [--fast] [--no-grid]
```

- 輸出：`report.html`（圖文報告，圖內嵌，可直接分享）、`descriptive_stats.xlsx`（所有表格，一表一工作表）、`figures/*.png`、`report.txt`（主控台輸出）、`summary.json`。
- 耗時（2 核心）：葡萄酒（178 筆、無遺失）約 1 分鐘；房價（2,500 筆、有遺失、插補比較＋GridSearchCV）約 2.5 分鐘。`--fast` 跳過 MICE-ExtraTrees 插補；`--no-grid` 不做 GridSearchCV（改用預設參數）。

報告章節（依資料自動調整）與要向使用者說明的重點：

1. **資料與變數介紹**：資料說明、分析目的、用了哪些變數；變數表；自動排除 ID 欄位（`*_id`、每列不同的文字欄）；檢查重複列與常數欄。
2. **判斷任務類型**：依規則判斷並列出依據表（dtype、是否整數、不同值個數、範例值）與目標分佈圖。
3. **敘述性統計**：
   - 數值（平均、標準差、四分位、IQR、偏態、峰態、CV、1.5×IQR 離群值數）、類別（類別數、眾數、各水準次數）。
   - 分佈圖：連續變數畫直方圖（虛線與右上角標**平均數**與偏態）；類別變數與取值 ≤ 10 的離散數值（房間數、車位數）畫**長條圖**；箱形圖。
   - 與目標的關係：以文字列出關係最強的特徵（二元分類 SMD、多類別 η²(H)、迴歸 Spearman ρ、類別特徵 Cramér's V），搭配「各特徵與目標的關係」圖；Spearman 相關熱圖與高度共線的特徵對。**看效果量，不只看 p 值。**
4. **遺失值檢查**：先判斷有沒有遺失（遺失格數、有遺失的欄位、完整列比例）。
   - **沒有遺失 → 寫明「無遺失，跳過遺失值診斷與處理」**，直接切分與建模。
   - **有遺失 → 行（欄位）遺失比例與列（資料筆）遺失比例**（遺失矩陣、欄位比例圖、資料筆比例圖三者欄位順序一致，依遺失比例由高到低），**再做遺失機制診斷**：Little's MCAR 檢定、逐欄比較「遺失 vs 未遺失」的其他變數（Bonferroni）→ 推測 MCAR / MAR（或 MNAR）/ 資訊性遺失並給處理建議。
5. **資料切分**：測試集 25% 先鎖起來；訓練集做 5 折 CV（分類 StratifiedKFold、迴歸 KFold）；切分示意圖；分類列出**每個類別在訓練 / 測試各幾筆**與各折驗證集的類別筆數，迴歸比較訓練 / 測試的目標分佈（平均、中位數、KS 檢定）。
6. **遺失值處理：插補比較**（只在有遺失時；只用訓練集）：整列刪除、中位數、平均數、中位數＋遺失指標、KNN、MICE（貝氏迴歸）、MICE（ExtraTrees）。用**同一個 Gradient Boosting＋同一組 CV 折**比較下游分數，**1 個標準誤內選最簡單的**；另附遮罩還原 NRMSE、標準差縮水比、插補值分佈圖。
7. **模型訓練與比較**：Decision Tree、Random Forest、Gradient Boosting 都放在同一個 Pipeline（類別 one-hot ＋ 選出的插補），各自用 **GridSearchCV（5 折）** 調參。報告列出：
   - 「模型參數與調參」表：調參方式、搜尋範圍、最佳參數、固定參數、前處理；「參數說明」表：每個參數的意義與影響。
   - 最佳值落在搜尋範圍邊界時提醒擴大範圍（`min_samples_leaf=1` 已是最小值，不提醒）。
   - 模型比較表與圖（CV 平均 ± 標準差、訓練與測試分數）；二元分類 ROC＋混淆矩陣、多類別混淆矩陣、迴歸預測 vs 實際＋殘差圖；測試集置換重要度（以原始欄位為單位）。

| 模型 | GridSearchCV 搜尋的參數 | 固定參數 |
|---|---|---|
| Decision Tree | `max_depth`、`min_samples_leaf` | criterion（分類 gini、迴歸 squared_error） |
| Random Forest | `max_features`（sqrt / 0.5 / 1.0）、`min_samples_leaf`（1 / 3 / 5） | n_estimators=300 |
| Gradient Boosting | `n_estimators`、`learning_rate`（0.05 / 0.1）、`max_depth` | subsample=0.8 |

遺失值處理決策（只在有遺失時；細節見 eda guide §4）：
```
遺失 > 60%                 ──► 刪欄位，只留遺失指標
類別欄位                   ──► 另立「（遺失）」類別
遺失與目標有關             ──► 加遺失指標（add_indicator=True）
MAR 跡象 & 變數間相關強    ──► KNN / MICE（先 one-hot 類別再插補，才能借用類別資訊）
MCAR、少量                 ──► 中位數 / 眾數
整列刪除                   ──► 幾乎不用：丟資料、非 MCAR 有偏、上線遇遺失仍得插補
最後一律用下游 CV 驗證（差距在 1 個標準誤內就選簡單的）
```

## 向使用者解說時務必講到

- **資料與目標**：資料是什麼、要預測什麼、用了哪些變數。
- **任務類型的判斷依據**，以及因此選用的指標與切分方式。
- **遺失值**：先說有沒有；有的話說行／列遺失比例、推測機制與選用的插補法。MNAR 無法從資料證明，標「MAR（或 MNAR）」的欄位要問業務方「為什麼會遺失」。插補一定在 Pipeline 內、切分後 fit。
- **切分**：測試集多大、分類各類別在訓練 / 測試各幾筆。
- **模型**：每個模型用了哪些參數、哪些是 GridSearchCV 搜出來的（搜尋範圍、最佳值）、哪些是固定的；調參只用訓練集 CV，測試集只評估一次。
- **結果**：永遠報告 CV / 測試分數，不要拿訓練分數宣稱模型好；訓練遠優於測試要明說過擬合並給處置（限深度、加大 `min_samples_leaf`、降學習率、減少樹數）。
- 小測試集的差距要說明不確定性（45 筆測試集錯 1 筆 Accuracy 就差 0.022；750 筆的 AUC 標準誤約 ±0.02）。
- 分數好得不真實 → 先懷疑資料洩漏（ID、目標衍生欄位、未來資訊、切分前做了前處理）。
- 不要把特徵重要度說成因果；`feature_importances_`（MDI）偏好連續／高基數特徵，用置換重要度交叉確認。
- 「把遺失值猜準」≠「把目標預測準」：KNN/MICE 還原數值較準，但樹模型下游差距常在雜訊內。

## 標準流程（照順序做，不要跳步）

0. **變數介紹 → 判斷任務類型 → EDA → 遺失值檢查**：見上面各節（使用者沒要求完整 EDA 時，至少介紹欄位、看目標分佈、檢查有無遺失與 ID／洩漏欄位）。
1. **切分**：分類 `stratify=y`、迴歸不用；時間序列用 `TimeSeriesSplit`；同一對象多筆用 `GroupKFold`。測試集最後才碰。
2. **洩漏檢查**：丟掉 ID 類欄位；確認沒有目標衍生欄位或未來資訊；插補／編碼只在訓練折 fit（`Pipeline`）。
3. **前處理**：樹不需要標準化；類別特徵 one-hot；有遺失時在 Pipeline 內插補（sklearn 的 `GradientBoosting*` 不接受 NaN）。
4. **訓練 + 調參**：三個模型各用 `GridSearchCV`（5 折，分類 `StratifiedKFold`、迴歸 `KFold`）；參數組合太多時改 `RandomizedSearchCV`。
5. **比較**：CV 平均 ± 標準差、訓練與測試的落差。
6. **門檻／校準**（二元分類）：依成本調決策閾值；必要時 `CalibratedClassifierCV`。
7. **解釋**：測試集置換重要度；淺樹可用 `export_text` 轉成規則。
8. **報告**：附 CV 平均 ± 標準差、測試集一次性結果、模型參數、重要特徵、已知限制。

## 選模型決策流程

```
要向人說明規則？            ──► 淺層 Decision Tree（max_depth 2–5）
要快速、穩、少調參？        ──► Random Forest
要較高準確度、願意調參？    ──► Gradient Boosting（GridSearchCV 或 early stopping）
資料很小、模型分數差不多？  ──► 選最簡單、最好解釋的（通常是淺樹）
```

## 三個模型的一頁重點

| 模型 | 一句話 | 關鍵參數 | 最常見的坑 |
|---|---|---|---|
| Decision Tree | 貪婪地找讓不純度（Gini／Entropy／平方誤差）下降最多的切分 | `max_depth`、`min_samples_leaf`、`ccp_alpha` | 不限深度 → 訓練完美、測試較差 |
| Random Forest | bootstrap 抽樣 + 每次分裂只看隨機特徵子集，多棵深樹平均，**降方差** | `n_estimators`（200–500）、`max_features`、`min_samples_leaf` | 用 `feature_importances_` 直接下結論 |
| Gradient Boosting | 每棵淺樹擬合前面模型的負梯度，**降偏差** | `learning_rate`、`n_estimators`、`max_depth`（2–5）、`subsample` | 樹太多、學習率太大又沒用 CV／early stopping 控制 |

核心公式（需要時引用）：
- 分裂增益：`Gain = Imp(parent) − Σ (n_child/n)·Imp(child)`
- Bagging 平均的方差：`Var = ρσ² + (1−ρ)σ²/B`；OOB 比例 ≈ 36.8%
- Gradient Boosting：`F_m = F_{m−1} + η·h_m`，h_m 擬合負梯度（平方誤差 → 殘差；log loss → y − p）

## 程式碼模板（複製即用）

### Decision Tree（可解釋）
```python
from sklearn.tree import DecisionTreeClassifier, export_text
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
print(export_text(tree, feature_names=list(X_tr.columns)))
```

### Random Forest（+ OOB）
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=1,
                            oob_score=True, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
print("OOB:", rf.oob_score_)
```

### Gradient Boosting（+ early stopping；不接受 NaN，需先插補）
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.03, max_depth=2, subsample=0.8,
                                validation_fraction=0.2, n_iter_no_change=50, random_state=42).fit(X_tr_imputed, y_tr)
print("實際樹數:", gb.n_estimators_)
```

### GridSearchCV（插補放在 Pipeline 內）
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=300, random_state=42))])
grid = {"m__max_features": ["sqrt", 0.5, 1.0], "m__min_samples_leaf": [1, 3, 5]}
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy").fit(X_tr, y_tr)
print(gs.best_params_, gs.best_score_)        # 最佳參數與 CV 分數（略偏樂觀）
print(gs.score(X_te, y_te))                   # 測試集只評估一次
```

### 可靠的特徵重要度
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_te, y_te, scoring="accuracy", n_repeats=10, random_state=42)
imp = pd.Series(r.importances_mean, index=X_te.columns).sort_values(ascending=False)
```

## Demo

- 完整流程 demo：**兩筆都跑、分別呈現**（迴歸一份報告、分類一份報告，各自比較模型），把 `report.html` 與 `descriptive_stats.xlsx` 交給使用者，並用報告裡的實際數字解說：
  ```bash
  python scripts/eda_pipeline.py assets/regression_house_price.csv --target price_wan   # 有遺失 → 診斷與插補
  python scripts/eda_pipeline.py assets/classification_wine.csv --target cultivar      # 無遺失 → 跳過遺失處理
  ```
- 樹模型行為 demo（過擬合、OOB、重要度、early stopping）：`python scripts/demo_tree_models.py assets/classification_wine.csv --target cultivar`。
- 使用者給自己的資料時：先確認目標欄位；有欄位說明就用 `--dict` 帶入。

### Demo 資料集（只有兩筆）

| 資料 | 要預測什麼 | 筆數 | 特徵 | 遺失值 | 來源 |
|---|---|---|---|---|---|
| `regression_house_price.csv` | 迴歸：成交總價 `price_wan`（萬元，右偏） | 2,500 | 10 個**數值**：`area_sqm` 面積、`bedrooms`、`bathrooms`、`house_age` 屋齡、`floor` 樓層、`total_floors` 總樓層、`dist_center_km` 距市中心、`dist_mrt_km` 距捷運、`parking_spaces` 車位、`mgmt_fee` 管理費；ID `listing_id` | **有**：`floor` MCAR 4.7%、`house_age` MAR 12.6%（總樓層 ≤ 5 的老公寓常沒資料）、`dist_mrt_km` MAR 5.7%（距市中心 > 12 km 常未標示）、`mgmt_fee` MNAR 15.8%（越貴越不寫） | 模擬資料，`scripts/make_demo_data.py`（seed = 11） |
| `classification_wine.csv` | 多類別分類：葡萄品種 `cultivar`（cultivar_1 / 2 / 3 各 59 / 71 / 48） | 178 | 13 個數值化學成分：`alcohol` 酒精、`malic_acid` 蘋果酸、`ash` 灰分、`alcalinity_of_ash` 灰分鹼度、`magnesium` 鎂、`total_phenols` 總酚、`flavanoids` 類黃酮、`nonflavanoid_phenols` 非類黃酮酚、`proanthocyanins` 原花青素、`color_intensity` 色澤強度、`hue` 色調、`od280_od315` 吸光比、`proline` 脯胺酸 | **無** | UCI Wine（Aeberhard & Forina），取自 scikit-learn `load_wine()` |

每筆資料附欄位說明檔 `*_dictionary.csv`（含 `__dataset__` 資料描述列），`eda_pipeline.py` 會自動讀取。

### 兩筆 demo 的模型比較結果（實際執行 `eda_pipeline.py`；細節見 eda guide §7）

| | 迴歸：房價 | 分類：葡萄酒 |
|---|---|---|
| 任務判斷 | 1,394 種不同數值 → 迴歸 | 文字、3 種值 → 多類別分類 |
| 遺失值 | 有 4 欄 → 診斷；插補比較後選中位數 | 無 → 跳過診斷與處理 |
| 切分 | 隨機 75/25（1,875 / 625）+ KFold(5) | stratified 75/25（133 / 45；三類測試各 15 / 18 / 12）+ StratifiedKFold(5) |
| Decision Tree | max_depth=12、min_samples_leaf=10 → 測試 RMSE 274、R² 0.78 | max_depth=3 → CV 0.910、測試 Accuracy 0.956 |
| Random Forest | max_features=1.0 → RMSE 217、R² 0.86 | max_features=sqrt → **CV 0.985（最佳）、測試 1.000** |
| Gradient Boosting | n=400、lr=0.05、depth=3 → **RMSE 191、R² 0.89、MAPE 12.0%（最佳）** | n=300、lr=0.1、depth=2 → CV 0.978、測試 0.978 |

解說時：房價資料 Random Forest、Gradient Boosting 明顯優於單棵樹，Gradient Boosting 最佳；葡萄酒有 13 個化學特徵，集成模型（CV 0.98）明顯優於單棵樹（CV 0.91），Random Forest 依 CV 最佳；但測試集只有 45 筆（錯 1 筆差 0.022），測試 1.000 不宜過度解讀。迴歸要把 RMSE / MAE 對照目標尺度（房價中位數 1,118 萬）。

注意：CSV 中別用 `None`、`NA` 當類別值——`pd.read_csv` 會把它讀成遺失值（要保留就加 `keep_default_na=False, na_values=[""]`）。

## 檔案索引

- `scripts/eda_pipeline.py`：變數介紹 → 任務判斷 → 敘述統計 → 遺失值檢查（有才診斷與插補）→ 切分 → GridSearchCV 調參與模型比較，輸出 HTML / Excel / PNG 報表。
- `scripts/demo_tree_models.py`：Decision Tree、Random Forest、Gradient Boosting 的行為 demo（深度掃描、OOB、重要度、early stopping）。
- `scripts/make_demo_data.py`：產生兩筆 demo 資料與欄位說明檔（房價為模擬，葡萄酒取自 scikit-learn）。
- `references/eda-missing-data-guide.md`：建模前流程教學（資料與變數介紹、任務判斷、敘述統計、遺失值、資料切分、程式碼模板、兩筆 demo 完整結果、參考資料）。
- `references/tree-based-models-guide.md`：三個樹模型的原理、公式、參數、比較、工作流程、陷阱、demo 結果、參考資料。
- `references/eda_output_regression.txt`、`references/eda_output_classification.txt`：`eda_pipeline.py` 在兩筆 demo 資料上的主控台輸出。
- `references/demo_output.txt`：`demo_tree_models.py` 在葡萄酒資料上的輸出。
- `assets/regression_house_price.csv`、`assets/classification_wine.csv`：demo 資料。
- `assets/*_dictionary.csv`：兩筆資料的欄位說明。
