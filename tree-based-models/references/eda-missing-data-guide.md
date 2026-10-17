# 建模前流程指南：變數介紹 → 任務判斷 → EDA → 遺失值 → 資料切分 → 模型比較

搭配：`scripts/eda_pipeline.py`（一支跑完，章節順序與本指南、與 `report.html` 一致）
示範資料：`assets/regression_house_price.csv`（迴歸、全數值、有遺失）、`assets/classification_iris.csv`（分類、鳶尾花、無遺失）
本文的程式碼與 demo 數字以 scikit-learn 1.9 執行驗證。

```
資料與變數介紹 ─► 判斷任務類型 ─► 敘述性統計 ─► 遺失值檢查 ─┬─ 無遺失 ──────────────────► 資料切分 ─► 模型訓練與比較
                 （分類 or 迴歸）                          └─ 有遺失 ─► 遺失機制診斷 ─► 資料切分 ─► 插補比較 ─► 模型訓練與比較
                                                                                  （先切再補，避免洩漏）
```
模型訓練與比較：Decision Tree、Random Forest、Gradient Boosting，各自以 GridSearchCV（5 折）調參。

---

## 0. 資料與變數介紹

分析前先把「資料有哪些欄位、各代表什麼」講清楚，後面的圖表與模型結果才讀得懂。報告第 1 節依序寫出：

1. **資料說明**：資料從哪來、幾筆、每筆代表什麼（欄位說明檔中 `欄位 = __dataset__` 那一列）。
2. **分析目的（要預測什麼）**：目標欄位與其意義——分類列出各類別與筆數，迴歸列出單位、平均與範圍。
3. **用了哪些變數**：每個特徵的名稱、中文說明與單位，以及數值／類別各幾個；哪些欄位被排除（ID）。
4. **變數表**：

| 欄 | 內容 | 來源 |
|---|---|---|
| 欄位 | 原始欄位名稱 | 資料 |
| 說明、單位 | 欄位的中文意義與單位 | 欄位說明檔（`--dict`，或同資料夾的 `<檔名>_dictionary.csv`） |
| 角色 | 目標 / 特徵 / ID（排除） | 自動判斷（`--target` 指定目標；`*_id`、每列都不同的文字欄 → ID） |
| 變數型態 | 數值 / 類別 / 識別碼 | 自動判斷（dtype） |
| dtype、遺失數 | 資料型態與完整度 | 自動計算 |

欄位說明檔格式（CSV，UTF-8）：
```
欄位,說明,單位,角色
__dataset__,模擬的都會區房屋成交資料：2,500 筆……預測成交總價（萬元）,—,資料集
area_sqm,室內面積,平方公尺,特徵
price_wan,成交總價（預測目標）,萬元,目標
```
要點：
- **ID 欄位一定排除**：只用來識別資料列，沒有預測意義，留著可能讓模型「背答案」。
- **數字代碼要當類別**（郵遞區號、方案代碼）；文字欄位若每列都不同（姓名、備註）通常不能直接當特徵。
- 確認沒有「目標的衍生欄位」或「預測當下拿不到的未來資訊」（資料洩漏）。

---

## 1. 判斷是分類還是迴歸

任務類型決定了後面幾乎所有選擇，所以要最先確定：

| | 分類（Classification） | 迴歸（Regression） |
|---|---|---|
| 要預測的是 | 屬於哪一類（鳶尾花品種、是否流失、A/B/C 級） | 一個數量是多少（房價、銷售額、溫度） |
| 目標欄位長相 | 文字、布林，或少數幾個整數代碼 | 連續數值、很多不同值，大小和差距有意義 |
| sklearn 模型 | `DecisionTreeClassifier`、`RandomForestClassifier`、`GradientBoostingClassifier` | 對應的 `*Regressor` |
| 葉節點輸出 | 類別比例（機率） | 平均值 |
| 分裂準則 | Gini / Entropy | 平方誤差 |
| 評估指標 | 二元：AUC、PR-AUC、F1；多類別：Accuracy、Macro-F1、混淆矩陣 | RMSE、MAE、R²、MAPE |
| 切分方式 | `train_test_split(stratify=y)`、`StratifiedKFold` | `KFold`（目標很偏時可分箱後 stratify） |

**判斷規則**（`eda_pipeline.py` 的 `detect_task()`，`--task auto` 時使用）：
```
目標是文字 / 布林？                 ──► 分類
數值，但只有 2 種值（0/1）？         ──► 二元分類
整數且 ≤ 10 種值？                  ──► 多類別分類（⚠ 若是計數或評分，也可能當迴歸，要確認）
其餘（含小數、很多不同值）           ──► 迴歸
```

容易判斷錯的情況：
- **數字代碼其實是類別**：郵遞區號、門市代碼、1/2/3 代表三種方案 → 分類（或當特徵時要 one-hot）。
- **整數其實是數量**：來電次數、購買件數 → 迴歸。
- **評分（1–5 星）**：有順序，可當多類別分類，也可當迴歸；看業務是要「哪一級」還是「平均幾分」。
- **連續值被要求做決策**：例如「會不會逾期超過 30 天」，把連續目標切成類別就變分類問題——由業務需求決定。
- 自動判斷只是建議，與業務目的衝突時用 `--task` 指定。

Demo 判斷結果：`price_wan`（1,394 種不同的數值）→ **迴歸**；`species`（文字，3 種值）→ **多類別分類**。

---

## 2. 為什麼建模前一定要做 EDA

| EDA 要回答的問題 | 沒做會怎樣 |
|---|---|
| 目標分佈長什麼樣？（不平衡？偏態？） | 用錯指標（不平衡還看 accuracy）、迴歸被少數高值主導 |
| 哪些欄位是 ID、常數、重複、洩漏？ | 模型學到 ID 或未來資訊，上線崩盤 |
| 有沒有輸入錯誤 / 離群值？ | 錯誤資料被當成真訊號 |
| 有沒有遺失值？多少？為什麼會遺失？ | 刪掉大量資料、或插補引入偏誤 |
| 特徵和目標、特徵彼此的關係？ | 不知道從何解釋模型、共線性讓重要度被瓜分 |

---

## 3. 敘述性統計：要算哪些、怎麼解讀

### 3.1 數值欄位

| 統計量 | 用途 | 解讀提醒 |
|---|---|---|
| 個數、遺失% | 資料完整度 | 遺失 > 60% 的欄位通常只保留「遺失指標」 |
| 平均數 / 中位數 | 集中趨勢 | 兩者差很多 → 偏態；偏態時報中位數 |
| 標準差、IQR | 離散程度 | IQR 對離群值穩健 |
| 最小 / 最大值 | 合理範圍 | 負的年齡、10 倍的坪數 → 輸入錯誤 |
| 偏態（skew） | 對稱性 | \|skew\| > 1 屬明顯偏態；樹模型不需轉換，KNN 插補會受影響 |
| 峰態（超額 kurtosis） | 尾巴厚度 | 很大 → 尾巴厚、離群值多 |
| 變異係數 CV = σ/μ | 跨尺度比較離散度 | 只對非負、比例尺度的變數有意義 |
| 離群值數（1.5×IQR） | 異常偵測 | 先判斷「錯誤」還是「真實但罕見」；樹模型對真實離群值穩健 |

### 3.2 類別欄位
類別數、眾數與佔比、最少類別佔比、各水準次數。稀有水準（< 1%）可合併為「其他」（`OneHotEncoder(min_frequency=0.01)`）。

### 3.3 與目標的關係（單變量篩選）
腳本會計算下表的檢定與效果量，並在報告中以文字列出關係最強的幾個特徵，搭配「各特徵與目標的關係」圖（不另列表格）。

| 特徵 \ 目標 | 二元分類 | 多類別分類 | 迴歸 |
|---|---|---|---|
| 數值 | Mann-Whitney U ＋ 標準化平均差 SMD | Kruskal-Wallis ＋ η²(H) | Spearman ρ |
| 類別 | 卡方 ＋ Cramér's V | 卡方 ＋ Cramér's V | Kruskal-Wallis ＋ η²(H) |

- η²(H) = (H − k + 1) / (n − k)，H 為 Kruskal-Wallis 統計量、k 為組數、n 為樣本數（Tomczak & Tomczak, 2014）。
- 效果量經驗值（Cohen, 1988）：SMD 0.2 / 0.5 / 0.8 = 小 / 中 / 大；Cramér's V（2×2）0.1 / 0.3 / 0.5；\|ρ\| 0.1 / 0.3 / 0.5；η² 0.01 / 0.06 / 0.14。
- **樣本大時 p 值幾乎都顯著，要看效果量。** 單變量篩選看不到交互作用，不要用它刪特徵。

### 3.4 報告中的圖

| 圖 | 看什麼 |
|---|---|
| 目標分佈（分類：長條圖；迴歸：直方圖） | 不平衡 / 偏態 |
| 數值特徵分佈：連續變數畫直方圖（虛線 = 平均數，右上角標平均與偏態）；取值 ≤ 10 的離散數值（房間數、車位數）畫長條圖 | 偏態、多峰、異常集中值 |
| 箱形圖 | 離群值 |
| 類別特徵長條圖（遺失另列） | 稀有水準 |
| 特徵 vs 目標（分類：各類別的箱形圖；迴歸：散佈圖＋分箱中位數線，離散特徵用箱形圖） | 趨勢、非線性、門檻效應 |
| Spearman 相關熱圖 | 共線性、與目標的相關 |
| 遺失矩陣＋行（欄位）遺失比例＋列（資料筆）遺失比例（只在有遺失時，三圖欄位順序一致） | 哪裡遺失、遺失多少 |

---

## 4. 遺失值：先檢查有沒有，有才診斷與處理

### 4.0 第一步：判斷有沒有遺失
```python
miss = X.isna().sum()
print(miss[miss > 0], "完整列比例:", X.notna().all(axis=1).mean())
```
- **沒有遺失** → 不需要做遺失機制診斷與插補，直接進入資料切分與建模（報告會寫明「無遺失 → 跳過」）。
- **有遺失** → 先算兩種比例，再依 4.1–4.6 診斷機制、選擇處理方式；插補要在切分之後、放進 Pipeline。
- 注意：讀 CSV 時 `None`、`NA`、`N/A` 等字串會被 pandas 當成遺失；若它們是合法類別值，用 `pd.read_csv(path, keep_default_na=False, na_values=[""])`。

| 比例 | 定義 | 回答的問題 |
|---|---|---|
| **行（欄位）遺失比例** | 某欄位遺失筆數 ÷ 總筆數 | 哪些變數遺失得多？> 60% 考慮刪欄位 |
| **列（資料筆）遺失比例** | 某筆資料遺失欄位數 ÷ 特徵數；統計遺失 0、1、2… 個欄位的資料各幾筆 | 遺失集中在少數資料列還是分散？整列刪除會丟多少資料？ |

### 4.1 三種遺失機制（Rubin, 1976；Little & Rubin, 2019）

| 機制 | 定義 | 例子（房價 demo） | 整列刪除 | 單值插補（平均/中位數） | 多變量插補（KNN/MICE） |
|---|---|---|---|---|---|
| **MCAR** 完全隨機 | 遺不遺失與任何變數無關 | 隨機漏填的樓層 `floor` | 無偏，但浪費資料 | 無偏但壓縮變異 | 較佳 |
| **MAR** 隨機 | 遺失機率取決於**其他已觀測**變數 | 老公寓（`total_floors` ≤ 5）沒有屋齡 `house_age`；郊區（`dist_center_km` > 12）沒標捷運距離 `dist_mrt_km` | **有偏** | **有偏** | 可大致修正 |
| **MNAR** 非隨機 | 遺失機率取決於**該值本身** | 管理費越貴越不寫 `mgmt_fee` | 有偏 | 有偏 | 仍有偏，但較輕 |

### 4.2 如何診斷（腳本自動做）
1. **行／列遺失比例與完整列比例**：完整列比例低 → 整列刪除不可行。
2. **Little's MCAR 檢定**（Little, 1988；整體）：H0 = MCAR；p < 0.05 → 至少部分欄位不是 MCAR。檢定力有限，不顯著也不代表一定是 MCAR。
3. **逐欄比較「遺失 vs 未遺失」兩組的其他變數**（數值：SMD＋Mann-Whitney；類別：Cramér's V＋卡方；Bonferroni 校正）：
   - 都無明顯差異 → 符合 MCAR
   - 與某些已觀測變數有關 → MAR（或 MNAR）
4. **遺失本身 vs 目標**：遺失與目標有關 → 「資訊性遺失」，建議保留遺失指標。

> **重要：MNAR 無法從觀測資料證明**——遺失的值我們看不到，自然無法檢查「遺失是否取決於它」。
> 判斷 MNAR 只能靠領域知識（「誰會不想填？」「什麼情況下系統會漏記？」）。

### 4.3 處理方法一覽

| 方法 | sklearn | 優點 | 缺點 | 適用 |
|---|---|---|---|---|
| 整列刪除 | `dropna()` | 簡單 | 丟資料；非 MCAR 時有偏；**上線時遇到遺失仍得處理** | 幾乎不建議；MCAR 且遺失極少時 |
| 刪除欄位 | `drop(columns=)` | 簡單 | 丟訊號 | 遺失 > 60% 且遺失本身無訊息 |
| 平均數 / 中位數 | `SimpleImputer` | 快、穩 | 壓縮變異、扭曲分佈與相關；非 MCAR 有偏 | MCAR、遺失少量；**樹模型最簡單的起點** |
| 眾數 / 「遺失」類別 | `SimpleImputer(strategy="most_frequent" / "constant")` | 「遺失」類別可保留遺失訊號 | 新增一個水準 | 類別欄位首選「遺失」類別 |
| 遺失指標 | `SimpleImputer(add_indicator=True)` | 讓模型學到「遺失本身」 | 多欄位 | 遺失與目標有關（MAR/MNAR） |
| KNN | `KNNImputer`（Troyanskaya et al., 2001） | 借用相似樣本 | 需標準化；大資料慢；受偏態影響 | 中小資料、MAR |
| MICE（迭代迴歸） | `IterativeImputer(BayesianRidge())`（van Buuren & Groothuis-Oudshoorn, 2011） | 利用變數間關係 | 假設近似線性；較慢 | MAR、變數間相關強 |
| MICE＋樹（missForest 類） | `IterativeImputer(ExtraTreesRegressor())`（概念同 Stekhoven & Bühlmann, 2012） | 處理非線性與交互作用 | 最慢 | MAR、非線性關係 |

### 4.4 怎麼「選」插補法：用證據，不用直覺
腳本用三種角度比較（只在有遺失時執行，只用訓練集）：

1. **下游 CV（最重要）**：同一個模型（Gradient Boosting）、同一組 CV 折，只換插補法。插補放在 `Pipeline` 中，每折只在訓練折 fit。
   - 在 **1 個標準誤內選最簡單的方法**，避免為雜訊等級的差距增加複雜度（簡單 → 複雜：中位數、平均數、中位數＋遺失指標、KNN、MICE、MICE-ExtraTrees）。
   - 整列刪除的公平比法：只用完整列訓練，但在**完整的**驗證折上評分（預測時遇遺失仍得插補）。
2. **遮罩還原 NRMSE**：把已知值再隨機遮 10%，看誰還原得準（NRMSE = RMSE / σ；1.0 ≈ 用平均數猜）。只模擬 MCAR，且「還原得準」≠「預測得好」。
3. **分佈與變異**：插補後標準差 / 原標準差（< 1 表示變異被壓縮）；畫插補值 vs 觀測值的分佈。

### 4.5 實務決策流程
```
欄位遺失 > 60%？                ──► 刪欄位，只留遺失指標
類別欄位？                      ──► 另立「（遺失）」類別
遺失與目標有關？                ──► 加遺失指標（add_indicator=True）
MAR 跡象明顯 & 變數間相關強？   ──► KNN / MICE
其餘（MCAR、少量）              ──► 中位數 / 眾數
最後：用下游 CV 驗證你的選擇（差距在 1 個標準誤內就選簡單的）
```

### 4.6 最常見的錯誤
- **在切分前對全資料插補**（用到測試集的平均數）→ 資料洩漏。永遠放在 `Pipeline` 裡。
- 插補時用到目標變數 `y`（不要把 y 放進插補器）。
- 把 NaN 填 0（0 常常是有意義的值）。
- 整列刪除後回報「模型很準」，忽略了上線時遇到遺失根本無法預測。
- 看到 Little 檢定不顯著就宣稱 MCAR。
- 平均數插補後還拿該欄算相關係數 / 變異數做推論（被壓縮了）。

---

## 5. 資料切分：為什麼要切、怎麼切

### 5.1 為什麼
模型在訓練資料上的分數一定偏樂觀（深樹可以把訓練集背到 100%）。要估計「上線遇到新資料時的表現」，必須留一份**建模過程完全沒碰過**的資料。

### 5.2 三種角色

| 資料 | 用途 | 何時碰 |
|---|---|---|
| 訓練集（Training） | fit 模型、插補器、編碼器 | 隨時 |
| 驗證集 / CV 驗證折（Validation） | 選插補法、GridSearchCV 調參、選模型、early stopping | 建模過程 |
| 測試集（Test / Holdout） | 最終一次性報告 | **最後只碰一次** |

### 5.3 本流程的做法
```
全部資料 ──► 測試集 25%（鎖起來）
         └─► 訓練集 75% ──► 5 折交叉驗證：每次 4 折訓練、1 折驗證，輪流 5 次取平均 ± 標準差
```
- 插補法、GridSearchCV 的參數、最佳模型都用訓練集 CV 挑；測試集只拿來報告最終結果。
- 同時看「訓練分數 vs CV / 測試分數」的差距判斷過擬合。
- 分類報告會列出**每個類別在訓練 / 測試各幾筆**，以及各折驗證集的類別筆數；迴歸報告比較訓練 / 測試的目標平均、中位數與 KS 檢定。

### 5.4 分類 vs 迴歸

| | 分類 | 迴歸 |
|---|---|---|
| holdout | `train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)` | `train_test_split(X, y, test_size=0.25, random_state=42)` |
| CV | `StratifiedKFold(5, shuffle=True, random_state=42)` | `KFold(5, shuffle=True, random_state=42)` |
| 為什麼 | stratify 讓每份資料的類別比例都和全體一樣，少數類別才不會在某一折特別少 | 連續目標沒有類別可維持；切完用 KS 檢定或直方圖確認訓練 / 測試分佈相近 |

### 5.5 其他情境

| 情境 | 用什麼 | 原因 |
|---|---|---|
| 有時間順序（銷售、股價、逐月資料） | `TimeSeriesSplit`；或依日期切 | 只能用過去預測未來，隨機切等於偷看未來 |
| 同一個人 / 店有多筆資料 | `GroupKFold` / `StratifiedGroupKFold` | 同一群不能同時出現在訓練和驗證，否則分數虛高 |
| Gradient Boosting early stopping | 從訓練集再切 10–20% 驗證集（`validation_fraction`） | 不能用測試集決定樹的數量 |
| 樣本很少 | `RepeatedStratifiedKFold` / `RepeatedKFold` | 單次切分變異太大 |
| 類別極不平衡 | stratify + PR-AUC | 確保每折都有足夠的正類 |

### 5.6 切分與洩漏
- **先切分，再做任何會「學東西」的前處理**：插補（平均數、KNN、MICE）、標準化、特徵選擇都放進 `Pipeline`，`cross_val_score` / `GridSearchCV` 會在每一折只用訓練折 fit。
- 純看資料的 EDA（畫分佈、算遺失比例）可以在全資料做；但「根據 EDA 結果挑特徵」若用到目標，最好也只看訓練集。
- 重複列可能一筆落在訓練、一筆落在測試 → 先檢查（鳶尾花有 1 筆完全重複）。

---

## 6. 程式碼模板

### 6.1 插補放進 Pipeline（避免洩漏）
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="（遺失）")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
enc = ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat_pipe, cat_cols)])
model = Pipeline([("enc", enc),
                  ("imp", SimpleImputer(strategy="median", add_indicator=True)),   # 數值遺失 + 遺失指標
                  ("rf", RandomForestRegressor(n_estimators=300, random_state=42))])
```
先編碼類別、再對整個矩陣插補，KNN/MICE 才能同時借用數值與類別欄位的資訊。

### 6.2 KNN / MICE
```python
from sklearn.experimental import enable_iterative_imputer  # noqa: IterativeImputer 仍屬實驗功能，須先 import
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

knn  = [("scale", StandardScaler()), ("imp", KNNImputer(n_neighbors=5))]      # KNN 一定要先標準化
mice = [("imp", IterativeImputer(BayesianRidge(), max_iter=15, random_state=42))]
miss_forest = [("imp", IterativeImputer(ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3),
                                        max_iter=5, random_state=42))]
```

### 6.3 公平比較插補法
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(5, shuffle=True, random_state=42)
for name, steps in {"median": [("imp", SimpleImputer(strategy="median"))], "knn": knn, "mice": mice}.items():
    pipe = Pipeline([("enc", enc), *steps, ("m", GradientBoostingRegressor(random_state=42))])
    s = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
    print(name, s.mean().round(1), "±", s.std().round(1))
```

### 6.4 切分（分類 vs 迴歸）
```python
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42,
                                          stratify=y if is_clf else None)
print(pd.crosstab(y_tr, "訓練"), pd.crosstab(y_te, "測試"))     # 分類：各類別在訓練 / 測試各幾筆
cv = StratifiedKFold(5, shuffle=True, random_state=42) if is_clf else KFold(5, shuffle=True, random_state=42)
```

### 6.5 GridSearchCV 調參（插補放在 Pipeline 內）
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("m", GradientBoostingRegressor(subsample=0.8, random_state=42))])
grid = {"m__n_estimators": [200, 400], "m__learning_rate": [0.05, 0.1], "m__max_depth": [3, 4, 5]}
gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=42),
                  scoring="neg_root_mean_squared_error").fit(X_tr, y_tr)
print(gs.best_params_, -gs.best_score_)                     # 最佳參數與其 CV RMSE（略偏樂觀）
print(root_mean_squared_error(y_te, gs.predict(X_te)))      # 測試集只評估一次
```
- 參數名稱加 `m__` 前綴（Pipeline 步驟名＋雙底線）。
- 最佳值落在網格邊界（例如 `max_depth` 選到最小值）→ 擴大範圍再搜；`min_samples_leaf=1` 已是最小可能值，不必再往下。
- 組合太多時改用 `RandomizedSearchCV`。

### 6.6 診斷遺失機制（單欄）
```python
from scipy import stats
R = X[col].isna()
for other in X.columns.drop(col):
    if X[other].dtype.kind in "if":
        a, b = X.loc[R, other].dropna(), X.loc[~R, other].dropna()
        smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        print(other, "SMD", round(smd, 2), "p", stats.mannwhitneyu(a, b).pvalue)
print("目標平均：遺失", y[R].mean(), "未遺失", y[~R].mean())
```

---

## 7. Demo 實測結果：迴歸（有遺失）與分類（無遺失）各一

兩筆資料用同一支腳本、同一套流程跑出來（完整主控台輸出見 `references/eda_output_regression.txt`、`references/eda_output_classification.txt`）：
```bash
python scripts/eda_pipeline.py assets/regression_house_price.csv --target price_wan
python scripts/eda_pipeline.py assets/classification_iris.csv   --target species
```
（欄位說明檔 `assets/*_dictionary.csv` 會自動讀取。耗時約 2.5 分鐘與 1 分鐘，2 核心。）

### 7.1 迴歸：`regression_house_price.csv`（房價，全數值特徵、有遺失）

**① 資料與變數**：模擬的都會區房屋成交資料，2,500 筆。**分析目的：預測成交總價 `price_wan`（萬元）**，平均 1,217 萬、範圍 117–4,952 萬。**用了 10 個數值特徵**：`area_sqm` 室內面積（㎡）、`bedrooms` 房間數、`bathrooms` 衛浴數、`house_age` 屋齡（年）、`floor` 所在樓層、`total_floors` 總樓層、`dist_center_km` 距市中心（km）、`dist_mrt_km` 距捷運站（km）、`parking_spaces` 車位數、`mgmt_fee` 月管理費（元）；ID `listing_id` 排除。沒有類別變數。

**② 任務判斷**：`price_wan` 為數值、有 1,394 種不同值（佔 55.8%）→ **迴歸**；指標用 RMSE / MAE / R² / MAPE，切分用 KFold。

**③ 敘述性統計**：價格平均 1,217 萬、中位數 1,118 萬、偏態 1.09（右偏）；`bedrooms`、`bathrooms`、`parking_spaces` 為離散變數，用長條圖呈現；與價格關係最強為 `area_sqm`（Spearman ρ 0.66）、`mgmt_fee`（0.56）、`bedrooms`（0.51）、`house_age`（−0.44）、`dist_center_km`（−0.41）；右偏欄位 `dist_mrt_km`（3.29）、`dist_center_km`（1.46）、`floor`（1.25）。

**④ 遺失值檢查 → 有遺失**：4 個欄位、971 格（3.88%）。
- 行（欄位）遺失比例：`mgmt_fee` 15.8%、`house_age` 12.6%、`dist_mrt_km` 5.7%、`floor` 4.7%，其餘 6 欄 0%。
- 列（資料筆）遺失比例：0 個遺失 1,622 筆（64.9%）、遺失 1 欄 789 筆（31.6%）、2 欄 85 筆（3.4%）、3 欄 4 筆（0.2%，該列遺失 30%）→ 整列刪除會丟掉 35.1% 的資料。
- Little 檢定 χ² = 695.4、df = 98、p ≈ 9e-91 → 拒絕 MCAR。

| 欄位 | 遺失% | 真實機制（產生時設定） | 腳本推測 | 依據 |
|---|---|---|---|---|
| mgmt_fee | 15.8 | MNAR（越貴越不寫） | MAR（或 MNAR）△ | 與面積、屋齡、房間數相關（管理費本身和它們相關）；遺失者房價平均 1,556 vs 1,153 萬 |
| house_age | 12.6 | MAR（老公寓沒資料） | MAR（或 MNAR）✔ | 與 `total_floors`（SMD −1.00）強相關：低樓層建物遺失率高 |
| dist_mrt_km | 5.7 | MAR（郊區未標示） | MAR（或 MNAR）✔ | 與 `dist_center_km`（SMD +0.88）相關 |
| floor | 4.7 | MCAR | 符合 MCAR ✔ | 與其他變數無明顯關聯 |

> `mgmt_fee` 被判為「MAR（或 MNAR）」說明：MNAR 從資料上看起來常常像 MAR，無法單靠資料區分。

**⑤ 資料切分**：隨機 75/25 → 訓練 1,875 / 測試 625；目標平均 1,215 / 1,224、中位數 1,114 / 1,133，KS 檢定 p = 0.56；訓練集 5 折 KFold，每折 1,500 訓練 / 375 驗證。

**⑥ 插補比較（Gradient Boosting，5-fold CV RMSE，萬元）**：

| 插補法 | CV RMSE |
|---|---|
| 整列刪除 | 227.1 |
| 中位數（選用） | 200.0 |
| 平均數 | 199.7 |
| 中位數＋遺失指標 | 199.3 |
| KNN(k=5) | 199.7 |
| MICE(貝氏迴歸) | 201.4 |
| MICE(ExtraTrees) | 197.6（最低） |

- 除整列刪除外，各法差距都在 1 個標準誤（約 3.9）內 → 選最簡單的**中位數**。整列刪除比中位數差約 14%。
- 遮罩還原 NRMSE：中位數 1.01、KNN 0.78、MICE 0.69、MICE-ExtraTrees 0.68（`mgmt_fee` 由 0.97 降到 0.37）——KNN/MICE 把遺失值猜得較準，但下游預測幾乎沒差。

**⑦ 模型訓練（GridSearchCV，5 折，插補 = 中位數）**：

| 模型 | 搜尋範圍 | 最佳參數 | 固定參數 |
|---|---|---|---|
| Decision Tree | max_depth ∈ {4, 6, 8, 10, 12, None}；min_samples_leaf ∈ {1, 5, 10, 20}（24 組） | max_depth=12、min_samples_leaf=10 | criterion=squared_error |
| Random Forest | max_features ∈ {sqrt, 0.5, 1.0}；min_samples_leaf ∈ {1, 3, 5}（9 組） | max_features=1.0、min_samples_leaf=1 | n_estimators=300、bootstrap=True |
| Gradient Boosting | n_estimators ∈ {200, 400}；learning_rate ∈ {0.05, 0.1}；max_depth ∈ {3, 4, 5}（12 組） | n_estimators=400、learning_rate=0.05、max_depth=3 | subsample=0.8 |

Gradient Boosting 的 `max_depth=3` 落在搜尋範圍下緣，報告會提醒可往 2 擴大範圍。

**⑧ 模型比較（萬元）**：

| 模型 | CV RMSE | 訓練 RMSE | 測試 RMSE | 測試 MAE | 測試 R² | 測試 MAPE |
|---|---|---|---|---|---|---|
| Decision Tree | 276 | 191 | 274 | 206 | 0.78 | 18.1% |
| Random Forest | 217 | 78 | 217 | 162 | 0.86 | 14.4% |
| Gradient Boosting | **195** | 124 | **191** | **141** | **0.89** | **12.0%** |

置換重要度（Gradient Boosting，測試集，RMSE 增加量）：`area_sqm` 430 ≫ `dist_center_km` 169 > `house_age` 93 > `dist_mrt_km` 41 ≈ `parking_spaces` 37。

### 7.2 分類：`classification_iris.csv`（鳶尾花，數值特徵、無遺失）

**① 資料與變數**：Fisher (1936) 鳶尾花資料，150 朵花。**分析目的：預測鳶尾花品種 `species`**——setosa（山鳶尾）、versicolor（變色鳶尾）、virginica（維吉尼亞鳶尾）各 50 朵。**用了 4 個數值特徵**：`sepal_length_cm` 花萼長度、`sepal_width_cm` 花萼寬度、`petal_length_cm` 花瓣長度、`petal_width_cm` 花瓣寬度（公分）。沒有 ID 欄位。

**② 任務判斷**：`species` 是文字、3 種值 → **多類別分類**；指標用 Accuracy（三類筆數相同，最直觀，用於調參與選模型）、Macro-F1、AUC(ovr)；切分要 stratify。

**③ 敘述性統計**：三類各 50 筆（完全平衡）；平均：花萼長 5.84、花萼寬 3.06、花瓣長 3.76、花瓣寬 1.20 公分；花瓣長、寬呈雙峰（setosa 明顯較小）。與品種關係最強為 `petal_width_cm`（η²(H) 0.88；三類平均 0.25 / 1.33 / 2.03）、`petal_length_cm`（0.87）、`sepal_length_cm`（0.65）、`sepal_width_cm`（0.42）；高度相關：`petal_length_cm`–`petal_width_cm`（ρ 0.94）、`sepal_length_cm`–`petal_length_cm`（0.88）、`sepal_length_cm`–`petal_width_cm`（0.83）；有 1 筆完全重複的資料。

**④ 遺失值檢查 → 無遺失**：4 個特徵 × 150 筆共 600 格，遺失 0 格 → **跳過遺失機制診斷與插補**。

**⑤ 資料切分**：stratified 75/25 → 訓練 112 / 測試 38。

| 類別 | 全部 | 訓練 | 測試 |
|---|---|---|---|
| setosa | 50 | 38 | 12 |
| versicolor | 50 | 37 | 13 |
| virginica | 50 | 37 | 13 |
| 合計 | 150 | 112 | 38 |

訓練集再做 5 折 StratifiedKFold（每折驗證 22–23 筆，三類各 7–8 筆）。

**⑥ 模型訓練（GridSearchCV，5 折，scoring = accuracy）**：

| 模型 | 搜尋範圍 | 最佳參數 | 固定參數 |
|---|---|---|---|
| Decision Tree | max_depth ∈ {2, 3, 4, 5, 6, 8}；min_samples_leaf ∈ {1, 5, 10, 20}（24 組） | max_depth=3、min_samples_leaf=1 | criterion=gini |
| Random Forest | max_features ∈ {sqrt, 0.5, 1.0}；min_samples_leaf ∈ {1, 3, 5}（9 組） | max_features=sqrt、min_samples_leaf=1 | n_estimators=300、bootstrap=True |
| Gradient Boosting | n_estimators ∈ {100, 300}；learning_rate ∈ {0.05, 0.1}；max_depth ∈ {2, 3, 4}（12 組） | n_estimators=100、learning_rate=0.05、max_depth=2 | subsample=0.8 |

Gradient Boosting 的 `max_depth=2` 落在搜尋範圍下緣，報告會提醒可往 1 擴大範圍。

**⑦ 模型比較**：

| 模型 | CV Accuracy | 訓練 Accuracy | 測試 Accuracy | 測試 Macro-F1 | 測試 AUC(ovr) |
|---|---|---|---|---|---|
| Decision Tree | 0.955 | 0.964 | 0.895 | 0.897 | 0.977 |
| Random Forest | **0.964** | 1.000 | 0.895 | 0.897 | 0.989 |
| Gradient Boosting | 0.956 | 1.000 | **0.921** | **0.923** | **0.992** |

依 CV 選 Random Forest；混淆矩陣顯示 setosa 全對，錯誤都在 versicolor 與 virginica 之間（兩者花瓣尺寸重疊）。測試集只有 38 筆，猜錯 1 筆 Accuracy 就差 0.026，三個模型實際上不分軒輊。置換重要度（Random Forest）：`petal_length_cm` 最高（0.221），其餘接近 0——花瓣長、寬高度相關（ρ 0.94），打亂其中一個時另一個能補上，重要度因此被分攤。

### 7.3 兩筆資料模型比較總覽

| | 迴歸：房價 | 分類：鳶尾花 |
|---|---|---|
| 要預測什麼 | 成交總價（萬元） | 品種（3 類） |
| 特徵 | 10 個數值 | 4 個數值 |
| 任務判斷依據 | 1,394 種不同數值 | 目標為文字、3 種值 |
| 遺失值 | 有（4 欄）→ 診斷 + 插補（選中位數） | 無 → 跳過 |
| 切分 | 隨機 75/25（1,875 / 625）+ KFold(5) | stratified 75/25（112 / 38）+ StratifiedKFold(5) |
| 調參 | 三個模型各自 GridSearchCV（5 折） | 三個模型各自 GridSearchCV（5 折） |
| Decision Tree | RMSE 274、R² 0.78 | Accuracy 0.895 |
| Random Forest | RMSE 217、R² 0.86 | Accuracy 0.895（CV 最佳 0.964） |
| Gradient Boosting | **RMSE 191、R² 0.89、MAPE 12.0%（最佳）** | **Accuracy 0.921（測試最高）** |

讀法：
- 房價資料夠大（1,875 筆訓練），Random Forest、Gradient Boosting 明顯優於單棵樹，Gradient Boosting 最好。
- 鳶尾花很容易分（花瓣尺寸幾乎就能決定品種），單棵深度 3 的樹就有 0.955 的 CV Accuracy；測試集太小，模型間差異在 1 筆之內，視為平手。這時**可解釋的淺樹**是很好的選擇。
- 兩筆資料的訓練分數都明顯高於測試（RF 訓練 Accuracy 1.000；房價 RF 訓練 RMSE 78 vs 測試 217），所以一定要用 CV / 測試集評估。

### 7.4 從兩筆資料學到的事
1. 先介紹資料與目標、再判斷任務類型，後面的指標與切分方式才選得對（迴歸用 RMSE + KFold；多類別用 Accuracy + StratifiedKFold）。
2. 先檢查有沒有遺失：鳶尾花沒有遺失，整段診斷與插補都可省略；房價有遺失才需要，且整列刪除最差。
3. 每個模型都用 GridSearchCV 調參，並把搜尋範圍、最佳參數與固定參數寫進報告；最佳值落在範圍邊界時要擴大範圍。
4. 小資料（鳶尾花測試 38 筆）的分數差距要謹慎解讀；大資料（房價）的差距才有意義。

---

## 8. 參考資料

### 8.1 內容參考（方法與理論）
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592. —— MCAR / MAR / MNAR 定義（§4.1）。
- Little, R. J. A. (1988). A test of missing completely at random for multivariate data with missing values. *Journal of the American Statistical Association*, 83(404), 1198–1202. —— Little's MCAR 檢定（§4.2）。
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley. —— 遺失資料處理總論、整列刪除與單值插補的偏誤（§4）。
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–67. —— MICE（§4.3）。
- Troyanskaya, O., et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6), 520–525. —— KNN 插補（§4.3）。
- Stekhoven, D. J., & Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118. —— 以樹模型迭代插補（§4.3 MICE＋樹）。
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum. —— 效果量大小的經驗門檻（§3.3）。
- Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 21(1), 19–25. —— Kruskal-Wallis 的 η²(H)（§3.3）。
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of IJCAI-95*, 1137–1143. —— 分層 K 折交叉驗證（§5）。
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. —— 訓練／驗證／測試的角色、交叉驗證與模型選擇（§5）。
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830；以及 scikit-learn 使用者指南（Imputation of missing values、Cross-validation、GridSearchCV、Pipeline）—— 程式碼 API 與預設值（本文以 scikit-learn 1.9 驗證）。
- 樹模型本身（決策樹、隨機森林、梯度提升）的參考文獻見 `tree-based-models-guide.md` §8。

### 8.2 資料來源
- **鳶尾花 `classification_iris.csv`**：Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188。本 skill 取自 scikit-learn 內建的 `sklearn.datasets.load_iris()`（為 UCI Machine Learning Repository Iris 資料集的複本），由 `scripts/make_demo_data.py` 改成易讀的欄位名稱後存檔；欄位說明見 `assets/classification_iris_dictionary.csv`。
- **房價 `regression_house_price.csv`**：**模擬資料**，非真實成交紀錄。由 `scripts/make_demo_data.py` 以固定亂數種子（seed = 11）產生：價格依面積、屋齡、距市中心、距捷運、樓層與車位的設定公式加上隨機誤差生成，再依 MCAR（floor）、MAR（house_age、dist_mrt_km）、MNAR（mgmt_fee）機制注入遺失值；欄位說明見 `assets/regression_house_price_dictionary.csv`。
