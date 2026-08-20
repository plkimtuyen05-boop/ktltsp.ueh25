"""
analysis_segmentation.py
----------------------------
Phân khúc khách hàng bằng K-Means Clustering, trả lời Câu hỏi nghiên cứu 2
(Mục 2.2.iii đề cương): phân loại khách hàng dựa trên tần suất chụp hóa đơn,
tổng giá trị chi tiêu và danh mục hàng hóa yêu thích -> cung cấp insight
phân khúc khách hàng cho các thương hiệu đối tác (đầu ra dùng cho B2B Report).

Đặc trưng (features) xây dựng theo mô hình RFM mở rộng:
- frequency:        số hóa đơn (bill) khác nhau của khách hàng
- monetary_total:    tổng chi tiêu
- monetary_avg:      giá trị trung bình mỗi hóa đơn
- recency_days:      số ngày kể từ lần mua gần nhất tới thời điểm phân tích
- brand_diversity:   số thương hiệu khác nhau đã mua (mức độ trung thành/khám phá)
- category_diversity:số loại sản phẩm khác nhau đã mua

Cách dùng:
    python src/analysis_segmentation.py --k 4
    python src/analysis_segmentation.py --auto_k   # tự tìm k tối ưu bằng Elbow + Silhouette
"""

from __future__ import annotations
import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from db import get_connection, DB_PATH


def build_customer_features(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT user_id, bill_id, brand, product_type, price, purchase_date
        FROM invoice_items
        WHERE is_price_valid = 1 AND is_date_valid = 1
        """,
        conn,
    )
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df = df.dropna(subset=["purchase_date"])

    ref_date = df["purchase_date"].max() + pd.Timedelta(days=1)

    agg = df.groupby("user_id").agg(
        frequency=("bill_id", "nunique"),
        monetary_total=("price", "sum"),
        monetary_avg=("price", "mean"),
        last_purchase=("purchase_date", "max"),
        brand_diversity=("brand", "nunique"),
        category_diversity=("product_type", "nunique"),
    ).reset_index()

    agg["recency_days"] = (ref_date - agg["last_purchase"]).dt.days
    agg = agg.drop(columns=["last_purchase"])
    return agg


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 8)) -> tuple[int, pd.DataFrame]:
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    results_df = pd.DataFrame(results)
    best_k = int(results_df.loc[results_df["silhouette"].idxmax(), "k"])
    return best_k, results_df


def profile_clusters(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    profile = df.groupby("cluster")[feature_cols].mean().round(1)
    profile["n_customers"] = df.groupby("cluster").size()
    return profile.sort_values("monetary_total", ascending=False)


def label_segments(profile: pd.DataFrame) -> dict:
    """Gán nhãn nghiệp vụ dễ hiểu cho từng cluster dựa trên XẾP HẠNG TƯƠNG ĐỐI
    giữa các cụm đã tìm được (không so với median cố định, vì với k nhỏ,
    median dễ khiến nhiều cụm bị gắn cùng 1 nhãn). Nhãn dễ hiểu này phục vụ
    trực tiếp cho B2B Report (brand đối tác đọc hiểu ngay hơn là 'cluster 0')."""
    # Điểm tổng hợp = trung bình hạng percentile của chi tiêu và tần suất
    money_rank = profile["monetary_total"].rank(pct=True)
    freq_rank = profile["frequency"].rank(pct=True)
    score = (money_rank + freq_rank) / 2

    order = score.sort_values(ascending=False).index.tolist()
    n = len(order)

    labels = {}
    for pos, cluster_id in enumerate(order):
        pct = pos / max(n - 1, 1)  # 0 = tốt nhất, 1 = kém nhất
        row = profile.loc[cluster_id]
        if pct == 0:
            labels[cluster_id] = "Khách hàng VIP (chi tiêu cao, tần suất cao)"
        elif pct == 1:
            labels[cluster_id] = "Khách hàng ít tương tác / mới"
        elif money_rank[cluster_id] > freq_rank[cluster_id]:
            labels[cluster_id] = "Khách hàng chi tiêu cao, mua không thường xuyên"
        else:
            labels[cluster_id] = "Khách hàng thân thiết, chi tiêu vừa/thấp"
    return labels


def main():
    parser = argparse.ArgumentParser(description="Phân khúc khách hàng bằng K-Means")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--k", type=int, default=None, help="Số cluster cố định")
    parser.add_argument("--auto_k", action="store_true", help="Tự tìm k tối ưu bằng Silhouette Score")
    parser.add_argument("--output", default="../outputs/customer_segments.csv")
    args = parser.parse_args()

    conn = get_connection(args.db)
    df = build_customer_features(conn)
    print(f"📊 Xây dựng đặc trưng cho {len(df)} khách hàng")
    print(df.describe().round(1).to_string())

    feature_cols = ["frequency", "monetary_total", "monetary_avg",
                     "recency_days", "brand_diversity", "category_diversity"]
    X = df[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    if args.auto_k or args.k is None:
        best_k, k_search = find_optimal_k(X_scaled)
        print("\n=== Tìm K tối ưu (Elbow + Silhouette) ===")
        print(k_search.to_string(index=False))
        k = best_k
        print(f"\n👉 K tối ưu theo Silhouette Score: {k}")
    else:
        k = args.k

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    profile = profile_clusters(df, feature_cols)
    segment_labels = label_segments(profile)
    profile["segment_name"] = profile.index.map(segment_labels)
    df["segment_name"] = df["cluster"].map(segment_labels)

    print(f"\n=== HỒ SƠ TỪNG PHÂN KHÚC (k={k}) ===")
    print(profile.to_string())

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n💾 Đã lưu chi tiết phân khúc từng khách hàng vào: {args.output}")

    conn.close()
    return df, profile


if __name__ == "__main__":
    main()
