"""
import_excel.py
-----------------
Import dữ liệu hóa đơn từ file Excel gốc (đã có cấu trúc sẵn) vào database
SQLite dùng chung. Đây là bước "nạp dữ liệu lịch sử ban đầu" — sau bước này,
mọi hóa đơn mới sẽ được LLM đọc từ ảnh và ghi thẳng vào cùng 1 database
(xem llm_ocr_reader.py).

Cách chạy:
    python src/import_excel.py --excel data/hoa_don_goc.xlsx --sheet "Data Drink"
"""

from __future__ import annotations
import argparse
import sqlite3
import pandas as pd
from db import init_db, DB_PATH

# Ánh xạ tên cột trong file Excel gốc -> tên cột chuẩn trong database.
# Chỉnh sửa dict này nếu file Excel của bạn có tên cột khác.
COLUMN_MAPPING = {
    "--": "category",
    "Bill ID": "bill_id",
    "User ID": "user_id",
    "Giới tính": "gender",
    "Price": "price",
    "Product": "product_type",
    "Tên SP": "product_name",
    "Product Brand": "brand",
    "Ngày trên bill": "purchase_date",
}

REQUIRED_DB_COLUMNS = [
    "bill_id", "user_id", "gender", "category", "product_type",
    "product_name", "brand", "price", "purchase_date"
]


def load_and_map(excel_path: str, sheet_name: str | int = 0) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.rename(columns=COLUMN_MAPPING)

    # Đảm bảo đủ cột, thiếu thì tạo cột rỗng
    for col in REQUIRED_DB_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[REQUIRED_DB_COLUMNS].copy()

    # Loại bỏ dòng hoàn toàn rỗng (thường là dòng cuối do Excel table format)
    df = df.dropna(how="all")
    df = df.dropna(subset=["bill_id"])  # bill_id là bắt buộc để định danh hóa đơn

    # Chuẩn hóa kiểu dữ liệu
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def insert_rows(conn: sqlite3.Connection, df: pd.DataFrame, source: str = "excel_import"):
    inserted, skipped_dup = 0, 0
    cur = conn.cursor()
    for _, row in df.iterrows():
        purchase_date_str = (
            row["purchase_date"].strftime("%Y-%m-%d %H:%M:%S")
            if pd.notnull(row["purchase_date"]) else None
        )
        try:
            cur.execute(
                """
                INSERT INTO invoice_items
                    (bill_id, user_id, gender, category, product_type,
                     product_name, brand, price, purchase_date, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["bill_id"]), str(row["user_id"]), row["gender"],
                    row["category"], row["product_type"], row["product_name"],
                    row["brand"],
                    float(row["price"]) if pd.notnull(row["price"]) else None,
                    purchase_date_str, source,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # Trùng (bill_id, product_name, price) -> đã tồn tại, bỏ qua
            skipped_dup += 1
    conn.commit()
    return inserted, skipped_dup


def main():
    parser = argparse.ArgumentParser(description="Import Excel hóa đơn -> SQLite")
    parser.add_argument("--excel", required=True, help="Đường dẫn file Excel")
    parser.add_argument("--sheet", default=0, help="Tên hoặc index sheet (mặc định sheet đầu)")
    parser.add_argument("--db", default=DB_PATH, help="Đường dẫn database SQLite")
    args = parser.parse_args()

    sheet = args.sheet
    try:
        sheet = int(sheet)
    except (TypeError, ValueError):
        pass

    conn = init_db(args.db)
    df = load_and_map(args.excel, sheet)
    inserted, skipped_dup = insert_rows(conn, df)

    print(f"📥 Đọc {len(df)} dòng hợp lệ từ '{args.excel}' (sheet={sheet!r})")
    print(f"✅ Đã ghi mới: {inserted} dòng")
    print(f"⏭️  Bỏ qua (trùng lặp): {skipped_dup} dòng")

    total = conn.execute("SELECT COUNT(*) FROM invoice_items").fetchone()[0]
    print(f"📊 Tổng số dòng hiện có trong database: {total}")
    conn.close()


if __name__ == "__main__":
    main()
