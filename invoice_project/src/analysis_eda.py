"""
analysis_eda.py
------------------
Phân tích khám phá dữ liệu (EDA) + xuất biểu đồ, phục vụ:
- Chương 4 khóa luận (Thực trạng và Kết quả nghiên cứu)
- Ý tưởng nội dung cho Dashboard nội bộ Vidimi

Cách dùng:
    python src/analysis_eda.py
"""

from __future__ import annotations
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from db import get_connection, DB_PATH

plt.rcParams["font.family"] = "DejaVu Sans"  # hỗ trợ tiếng Việt cơ bản trên matplotlib


def load_clean_data(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM invoice_items", conn)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    return df


def chart_data_quality(df: pd.DataFrame, outdir: str):
    total = len(df)
    counts = {
        "Hợp lệ hoàn toàn": ((df["is_date_valid"] == 1) & (df["is_price_valid"] == 1)).sum(),
        "Lỗi ngày (OCR)": (df["is_date_valid"] == 0).sum(),
        "Lỗi giá (OCR)": (df["is_price_valid"] == 0).sum(),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.keys(), counts.values(), color=["#2ca02c", "#d62728", "#ff7f0e"])
    ax.set_title(f"Chất lượng dữ liệu hóa đơn (n={total})")
    ax.set_ylabel("Số dòng")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + 2, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(f"{outdir}/chart_data_quality.png", dpi=150)
    plt.close()


def chart_revenue_over_time(df: pd.DataFrame, outdir: str):
    valid = df[(df["is_date_valid"] == 1) & (df["is_price_valid"] == 1)].dropna(subset=["purchase_date"])
    monthly = valid.set_index("purchase_date").resample("W")["price"].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    monthly.plot(ax=ax, marker="o", color="#1f77b4")
    ax.set_title("Tổng giá trị hóa đơn theo tuần")
    ax.set_ylabel("Tổng giá trị")
    ax.set_xlabel("Tuần")
    plt.tight_layout()
    plt.savefig(f"{outdir}/chart_revenue_over_time.png", dpi=150)
    plt.close()


def chart_top_brands(df: pd.DataFrame, outdir: str, top_n: int = 15):
    top = df["brand"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#9467bd")
    ax.set_title(f"Top {top_n} thương hiệu theo số hóa đơn")
    ax.set_xlabel("Số hóa đơn")
    plt.tight_layout()
    plt.savefig(f"{outdir}/chart_top_brands.png", dpi=150)
    plt.close()


def chart_product_type_share(df: pd.DataFrame, outdir: str):
    counts = df["product_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Cơ cấu loại sản phẩm (Product Type)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/chart_product_type_share.png", dpi=150)
    plt.close()


def chart_gender_spend(df: pd.DataFrame, outdir: str):
    valid = df[df["is_price_valid"] == 1]
    g = valid.groupby("gender")["price"].agg(["mean", "count"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index, g["mean"], color="#17becf")
    ax.set_title("Giá trị hóa đơn trung bình theo giới tính")
    ax.set_ylabel("Giá trị trung bình")
    for i, (idx, row) in enumerate(g.iterrows()):
        ax.text(i, row["mean"] + 1, f"n={int(row['count'])}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{outdir}/chart_gender_spend.png", dpi=150)
    plt.close()


def print_summary(df: pd.DataFrame):
    print("=== TỔNG QUAN DASHBOARD ===")
    print(f"Tổng số dòng hóa đơn (SKU-level):  {len(df)}")
    print(f"Số hóa đơn (bill) khác nhau:        {df['bill_id'].nunique()}")
    print(f"Số khách hàng (user) khác nhau:     {df['user_id'].nunique()}")
    print(f"Số thương hiệu (brand) khác nhau:   {df['brand'].nunique()}")
    valid = df[df["is_price_valid"] == 1]
    print(f"Tổng giá trị (dữ liệu hợp lệ):       {valid['price'].sum():,.0f}")
    print(f"Giá trị trung bình / hóa đơn:        {valid['price'].mean():,.1f}")
    if df["purchase_date"].notna().any():
        print(f"Khoảng thời gian dữ liệu hợp lệ:     "
              f"{df.loc[df['is_date_valid']==1,'purchase_date'].min()} "
              f"-> {df.loc[df['is_date_valid']==1,'purchase_date'].max()}")


def main():
    import os
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    conn = get_connection(DB_PATH)
    df = load_clean_data(conn)

    print_summary(df)
    print("\n📈 Đang xuất biểu đồ...")

    chart_data_quality(df, outdir)
    chart_revenue_over_time(df, outdir)
    chart_top_brands(df, outdir)
    chart_product_type_share(df, outdir)
    chart_gender_spend(df, outdir)

    print(f"✅ Đã lưu các biểu đồ vào thư mục: {outdir}")
    conn.close()


if __name__ == "__main__":
    main()
