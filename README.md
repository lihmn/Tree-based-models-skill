# Tree-Based Models Skill

Claude Skill：針對表格型資料（CSV），完成「資料與變數介紹 → 判斷分類／迴歸 → 敘述性統計 → 遺失值診斷與插補 → 資料切分 → 樹模型訓練與比較」的監督式學習流程，並輸出 HTML 報告、Excel 統計表與圖檔。

---

## 功能特色

- **自動判斷資料類型**：依目標欄位型態判斷二元分類、多類別分類或迴歸，並套用對應的指標與切分方式。
- **完整敘述性統計**：統計量、分佈圖（連續變數用直方圖、離散變數用長條圖）、特徵與目標的效果量與相關熱力圖等。
- **先檢查是否有遺失值、有才處理**：
  - 行／列遺失比例、Little's MCAR 檢定、MCAR / MAR / MNAR 機制推測
  - 比較 7 種插補方法（整列刪除、中位數、平均數、中位數＋遺失指標、KNN、MICE、MICE-ExtraTrees），差距在 1 個標準誤內時選最簡單的方法
- **評估流程**：25% 測試集保留到最後只評估一次；插補與調參都在 Pipeline 內，只用訓練集 5 折交叉驗證。
- **三個樹模型**：Decision Tree、Random Forest、Gradient Boosting，皆以 GridSearchCV 調參，並列出搜尋範圍與最佳參數。
- **特徵重要度**：採用測試集上的置換重要度（避免 impurity 重要度偏好高基數特徵）。

## 任務類型與評估指標

| | 二元分類 | 多類別分類 | 迴歸 |
|---|---|---|---|
| 調參／選模型指標 | AUC | Accuracy | RMSE |
| 其他報告指標 | PR-AUC、F1、ROC、混淆矩陣 | Macro-F1、AUC(ovr)、混淆矩陣 | MAE、R²、MAPE、殘差圖 |
| 切分方式 | Stratified + StratifiedKFold |  Stratified + StratifiedKFold | KFold |

## 檔案結構

```
├── README.md
├── tree-based-models/                       # Skill 本體（安裝時使用此資料夾）
│   ├── SKILL.md                             # Skill 主說明（Claude 讀取的入口）
│   ├── scripts/
│   │   ├── eda_pipeline.py                  # 主腳本：完整流程 + 報表輸出
│   │   ├── demo_tree_models.py              # 樹模型行為 demo
│   │   └── make_demo_data.py                # 產生 demo 資料與欄位說明檔
│   ├── references/
│   │   ├── eda-missing-data-guide.md        # 建模前流程教學與 demo 結果
│   │   ├── tree-based-models-guide.md       # 三個樹模型的原理與實務
│   │   └── *.txt                            # 腳本實際輸出紀錄
│   ├── assets/
│   │   ├── regression_house_price.csv            # 迴歸 demo（模擬房價，有遺失）
│   │   ├── regression_house_price_dictionary.csv
│   │   ├── classification_iris.csv               # 分類 demo（鳶尾花，無遺失）
│   │   └── classification_iris_dictionary.csv
│   ├── eda_regression_house_price/      # 房價 demo 的分析輸出
│   │   ├── report.html                  #   圖文報告
│   │   ├── descriptive_stats.xlsx       #   統計表
│   │   ├── figures/                     #   圖檔
│   │   ├── report.txt                   #   主控台輸出
│   │   └── summary.json                 #   關鍵數字
│   └── eda_classification_iris/         # 鳶尾花 demo 的分析輸出
│       ├── report.html                  #   圖文報告
│       ├── descriptive_stats.xlsx       #   統計表
│       ├── figures/                     #   圖檔
│       ├── report.txt                   #   主控台輸出
│       └── summary.json                 #   關鍵數字
├── materials/                               # Skill 說明報告
│   ├── tree_based_models_skill_report.qmd   # 報告原始檔（Quarto）
│   ├── tree_based_models_skill_report.html  # 報告輸出（可直接以瀏覽器開啟）
│   └── figures/                             # 報告使用的圖檔
├── breast_cancer_diagnosis.csv              # 未看過資料範例：乳癌診斷資料
├── 未看過資料_乳癌診斷_report.html            # 乳癌診斷資料的分析報告
└── 未看過資料_乳癌診斷_統計表.xlsx             # 乳癌診斷資料的統計表
```
 
- **`tree-based-models/`**：Skill 本體，安裝到 Claude 時使用此資料夾。
- **`materials/`**：完整說明報告，介紹方法、流程、檔案用途與示範結果；`.qmd` 可用 Quarto 重新產生 `.html`。
- **乳癌診斷相關檔案**：以 Skill 從未用過、且未附欄位說明檔的資料實測，驗證流程可套用到新資料。

## 程式說明與執行流程

主程式為 **`eda_pipeline.py`**，完整分析流程都在此執行；另外兩支為輔助程式。

| 程式 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| **`eda_pipeline.py`** | 主程式：EDA、遺失值處理、樹模型建模與報告輸出 | 任何表格 CSV + `--target` | `eda_<檔名>/` 資料夾 |
| `make_demo_data.py` | 產生兩筆 demo 資料（房價刻意放入 MCAR / MAR / MNAR 遺失） | 無（固定亂數種子） | `assets/` 的資料與欄位說明檔 |
| `demo_tree_models.py` | 教學用：示範過擬合、淺樹規則、OOB、MDI vs 置換重要度、early stopping | 分類 CSV | 主控台文字輸出 |

```
make_demo_data.py ──► assets/*.csv
                          │
                          ▼
              eda_pipeline.py --target Y
   1. 讀檔、判斷分類／迴歸、資料與變數介紹
   2. 敘述性統計
   3. 遺失值檢查（無遺失則跳過；有則診斷機制）
   4. 資料切分（25% 測試集 + 訓練集 5 折 CV）
   5. 插補方法比較（僅在有遺失時）
   6. DT / RF / GB 以 GridSearchCV 調參與比較
   7. 輸出 report.html、descriptive_stats.xlsx、figures/、report.txt、summary.json
                          │
                          ▼
   eda_regression_house_price/、eda_classification_iris/
```

`demo_tree_models.py` 為獨立的程式，不在主流程內。

## 安裝與使用

### 作為 Claude Skill

將 `tree-based-models/` 資料夾放進 Claude 的 skills 目錄，或在 Claude 的 Skill 設定中上傳 zip 檔。

### 直接執行

```bash
# 安裝套件
pip install scikit-learn pandas numpy scipy matplotlib openpyxl

# 執行 demo
python scripts/eda_pipeline.py assets/regression_house_price.csv --target price_wan
python scripts/eda_pipeline.py assets/classification_iris.csv --target species

# 分析自己的資料
python scripts/eda_pipeline.py my_data.csv --target y --dict my_data_dictionary.csv --out eda_my_data
```

| 參數 | 說明 |
|---|---|
| `--target` | 目標欄位（必填） |
| `--task` | `auto`（預設）/ `classification` / `regression` |
| `--dict` | 欄位說明檔；未指定時自動尋找 `<檔名>_dictionary.csv` |
| `--drop` | 額外排除的欄位（ID 欄位會自動排除） |
| `--out` | 輸出資料夾（預設 `eda_<檔名>`） |
| `--fast` | 跳過最慢的 MICE-ExtraTrees 插補 |
| `--no-grid` | 不做 GridSearchCV，改用預設參數 |

**輸出**：`report.html`（圖文報告）、`descriptive_stats.xlsx`（所有統計表）、`figures/*.png`、`report.txt`、`summary.json`。

## 示範結果

### Demo 資料
 
| 資料 | 任務 | 遺失值 | 最佳模型 | 測試表現 |
|---|---|---|---|---|
| 模擬房價（2,500 筆、10 特徵） | 迴歸 | 有 → 選用中位數插補 | Gradient Boosting | RMSE 191 萬元、R² 0.89 |
| 鳶尾花（150 筆、4 特徵） | 多類別分類 | 無 → 跳過 | 三模型相近 | Accuracy 0.895–0.921 |
 
### 未看過的資料：乳癌診斷
 
以 Skill 從未用過、且**未附欄位說明檔**的資料實測（569 筆、30 特徵，目標 `diagnosis`）。Skill 自動判斷為二元分類、以 malignant 為正類、無遺失值跳過插補，並偵測出高度相關特徵。
 
| 模型 | 測試 AUC | 測試 F1 | 測試 Accuracy |
|---|---|---|---|
| Decision Tree | 0.969 | 0.830 | 0.888 |
| Random Forest | 0.996 | 0.940 | 0.958 |
| **Gradient Boosting** | **0.998** | **0.950** | **0.965** |

## 主要參考資料

- Breiman, L., et al. (1984). *Classification and Regression Trees*. Wadsworth.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
- Little, R. J. A. (1988). A test of missing completely at random for multivariate data with missing values. *JASA*, 83(404), 1198–1202.
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–67.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

**資料來源**：

- 鳶尾花資料（Fisher, 1936）與乳癌診斷資料（Wolberg et al., 1995）皆取自 scikit-learn。
- 房價資料為 `make_demo_data.py` 以 seed = 11 產生的模擬資料。
