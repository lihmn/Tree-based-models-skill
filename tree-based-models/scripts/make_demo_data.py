"""產生兩筆 demo 資料（固定亂數種子，可重現），以及各自的欄位說明檔。

1. regression_house_price.csv   —— 迴歸：房價（目標 price_wan，單位：萬元）
       * 特徵全為數值（沒有類別變數）
       * 有遺失值，且刻意放入三種遺失機制，讓遺失值診斷與插補有東西可看：
           MCAR（完全隨機遺失）：floor
           MAR （隨機遺失，取決於其他已觀測欄位）：house_age（老公寓沒有屋齡）、dist_mrt_km（郊區未標示）
           MNAR（非隨機遺失，取決於值本身）：mgmt_fee（管理費越高越不願意寫）
2. classification_iris.csv      —— 分類：鳶尾花（目標 species，3 個品種）
       * Fisher (1936) 經典資料，取自 sklearn 內建；4 個數值特徵（花萼/花瓣的長與寬）
       * 沒有遺失值（流程會判斷「無遺失」並跳過遺失值診斷與處理）

*_dictionary.csv 是欄位說明（欄位、說明、單位、角色；欄位 __dataset__ 那列是整份資料的描述），給 eda_pipeline.py 的 --dict 用，不是資料。

用法:  python make_demo_data.py [輸出資料夾，預設 ../assets]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "assets"
out_dir.mkdir(parents=True, exist_ok=True)
sigmoid = lambda z: 1 / (1 + np.exp(-z))


# ====================================================================== 1. 迴歸：房價（全數值、有遺失）
def make_house(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    dist_center = np.round(rng.gamma(2.2, 3.2, n).clip(0.3, 35), 2)          # 距市中心 km
    total_floors = np.where(rng.random(n) < 0.35, rng.integers(2, 6, n), rng.integers(6, 31, n))
    floor = np.array([rng.integers(1, t + 1) for t in total_floors]).astype(float)
    old_walkup = total_floors <= 5
    house_age = np.where(old_walkup, rng.normal(38, 9, n), rng.gamma(2.5, 6, n))
    house_age = np.round(house_age.clip(0, 60), 1)
    area = np.round(rng.lognormal(np.log(85), 0.35, n).clip(20, 300), 1)       # ㎡
    bedrooms = np.clip(np.round(area / 30 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.6 + rng.normal(0.3, 0.5, n)), 1, 4).astype(int)
    dist_mrt = np.round(rng.exponential(0.25 + 0.06 * dist_center), 2)          # 越郊區離捷運越遠
    parking = np.where(old_walkup, rng.binomial(1, 0.15, n), rng.choice([0, 1, 2], n, p=[0.35, 0.55, 0.10]))
    mgmt_fee = np.where(old_walkup, rng.uniform(0, 600, n), 32 * area * rng.uniform(0.7, 1.4, n))
    mgmt_fee = np.round(mgmt_fee, -1)

    unit = 16.0 * np.exp(-0.055 * dist_center) + 2.5                          # 萬/㎡：離市中心越遠越便宜
    unit *= np.exp(-0.011 * house_age)                                         # 屋齡折舊
    unit *= 1 + 0.30 * np.exp(-dist_mrt / 0.5)                                 # 捷運溢價
    unit *= np.where(total_floors >= 6, 1 + 0.006 * floor, 0.95)              # 電梯大樓樓層溢價
    price = unit * area + 150 * parking
    price = np.round(price * rng.lognormal(0, 0.12, n), 0)

    df = pd.DataFrame({
        "listing_id": [f"H{i:05d}" for i in range(n)],
        "area_sqm": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "house_age": house_age,
        "floor": floor, "total_floors": total_floors, "dist_center_km": dist_center, "dist_mrt_km": dist_mrt,
        "parking_spaces": parking, "mgmt_fee": mgmt_fee, "price_wan": price,
    })
    # --- 注入遺失 ---
    df.loc[rng.random(n) < 0.05, "floor"] = np.nan                                        # MCAR 5%
    df.loc[rng.random(n) < np.where(old_walkup, 0.30, 0.03), "house_age"] = np.nan         # MAR：老公寓
    df.loc[rng.random(n) < np.where(dist_center > 12, 0.25, 0.03), "dist_mrt_km"] = np.nan  # MAR：郊區
    df.loc[rng.random(n) < sigmoid(-3.2 + 0.0006 * mgmt_fee), "mgmt_fee"] = np.nan         # MNAR：越貴越不寫

    dictionary = pd.DataFrame([
        ("__dataset__", "模擬的都會區房屋成交資料：2,500 筆物件，依面積、格局、屋齡、樓層、區位與管理費等 10 個數值變數，"
                        "預測成交總價（萬元）。其中 4 個變數有遺失值。", "—", "資料集"),
        ("listing_id", "物件編號（識別碼，建模前會排除）", "—", "ID"),
        ("area_sqm", "室內面積", "平方公尺", "特徵"),
        ("bedrooms", "房間數", "間", "特徵"),
        ("bathrooms", "衛浴數", "間", "特徵"),
        ("house_age", "屋齡（老公寓常沒有資料，MAR 遺失）", "年", "特徵"),
        ("floor", "所在樓層（隨機漏填，MCAR 遺失）", "樓", "特徵"),
        ("total_floors", "建物總樓層（≤5 多為無電梯老公寓）", "層", "特徵"),
        ("dist_center_km", "距市中心距離", "公里", "特徵"),
        ("dist_mrt_km", "距最近捷運站距離（郊區常未標示，MAR 遺失）", "公里", "特徵"),
        ("parking_spaces", "車位數", "個", "特徵"),
        ("mgmt_fee", "每月管理費（越高越不願意寫，MNAR 遺失）", "元/月", "特徵"),
        ("price_wan", "成交總價（預測目標）", "萬元", "目標"),
    ], columns=["欄位", "說明", "單位", "角色"])
    return df, dictionary


# ====================================================================== 2. 分類：鳶尾花（數值特徵、無遺失）
def make_iris():
    """Fisher (1936) 鳶尾花資料（sklearn 內建，不需下載）：依花萼與花瓣的長寬預測品種。"""
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    df = d.frame.rename(columns={
        "sepal length (cm)": "sepal_length_cm", "sepal width (cm)": "sepal_width_cm",
        "petal length (cm)": "petal_length_cm", "petal width (cm)": "petal_width_cm"})
    df["species"] = df.pop("target").map(dict(enumerate(d.target_names)))
    dictionary = pd.DataFrame([
        ("__dataset__", "鳶尾花資料集（Fisher, 1936）：150 朵鳶尾花，3 個品種各 50 朵；"
                        "依花萼與花瓣的長、寬（公分）預測花的品種。", "—", "資料集"),
        ("sepal_length_cm", "花萼長度", "公分", "特徵"),
        ("sepal_width_cm", "花萼寬度", "公分", "特徵"),
        ("petal_length_cm", "花瓣長度", "公分", "特徵"),
        ("petal_width_cm", "花瓣寬度", "公分", "特徵"),
        ("species", "鳶尾花品種（預測目標）：setosa（山鳶尾）/ versicolor（變色鳶尾）/ virginica（維吉尼亞鳶尾）", "—", "目標"),
    ], columns=["欄位", "說明", "單位", "角色"])
    return df, dictionary


for name, maker in [("regression_house_price", make_house), ("classification_iris", make_iris)]:
    df, dic = maker()
    df.to_csv(out_dir / f"{name}.csv", index=False)
    dic.to_csv(out_dir / f"{name}_dictionary.csv", index=False, encoding="utf-8-sig")
    miss = df.isna().mean().mul(100).round(1)
    print(f"saved {name}.csv: {df.shape}；遺失%: {miss[miss > 0].to_dict() or '無'}")
