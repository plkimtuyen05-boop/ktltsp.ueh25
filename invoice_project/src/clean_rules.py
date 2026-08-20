"""
clean_rules.py
----------------
Bước làm sạch dữ liệu RULE-BASED (không dùng LLM) — chạy trước để phát hiện
và gắn cờ các dòng có khả năng bị lỗi OCR (ngày phi lý, giá bằng 0/âm,
brand/product bị thiếu...). Những dòng này sẽ là input ưu tiên cho bước
LLM cleaning (llm_data_cleaner.py) ở bước sau — dùng LLM cho toàn bộ dữ liệu
sẽ tốn chi phí API không cần thiết, trong khi rule-based lọc nhanh, miễn phí.

Đây chính là minh chứng thực nghiệm cho Câu hỏi nghiên cứu 1 của đề cương:
"Dữ liệu hóa đơn bị lỗi OCR trông như thế nào, và tần suất ra sao?"

Cách chạy:
    python src/clean_rules.py
"""

from __future__ import annotations
import sqlite3
from datetime import datetime
from db import get_connection, DB_PATH

# Ngưỡng hợp lý cho ngày mua hàng (điều chỉnh theo phạm vi dữ liệu thực tế)
MIN_VALID_YEAR = 2015
# Cho phép trễ tối đa vài ngày so với hiện tại (tránh lệch múi giờ nhẹ)
MAX_VALID_DATE = datetime.now()


def flag_invalid_dates(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT id, purchase_date FROM invoice_items")
    rows = cur.fetchall()
    n_flagged = 0
    for row in rows:
        is_valid = 1
        if row["purchase_date"] is None:
            is_valid = 0
        else:
            try:
                dt = datetime.strptime(row["purchase_date"], "%Y-%m-%d %H:%M:%S")
                if dt.year < MIN_VALID_YEAR or dt > MAX_VALID_DATE:
                    is_valid = 0
            except ValueError:
                is_valid = 0

        if is_valid == 0:
            conn.execute(
                "UPDATE invoice_items SET is_date_valid = 0, "
                "cleaning_notes = COALESCE(cleaning_notes || '; ', '') || ? WHERE id = ?",
                (f"invalid_date:{row['purchase_date']}", row["id"]),
            )
            n_flagged += 1
    conn.commit()
    return n_flagged


def flag_invalid_prices(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "UPDATE invoice_items SET is_price_valid = 0, "
        "cleaning_notes = COALESCE(cleaning_notes || '; ', '') || 'invalid_price' "
        "WHERE price IS NULL OR price <= 0"
    )
    conn.commit()
    return cur.rowcount


def flag_missing_core_fields(conn: sqlite3.Connection) -> int:
    """Đánh dấu các dòng thiếu brand hoặc product_name — cần LLM đọc lại/chuẩn hóa."""
    cur = conn.execute(
        "UPDATE invoice_items SET cleaning_notes = COALESCE(cleaning_notes || '; ', '') || 'missing_field' "
        "WHERE (brand IS NULL OR TRIM(brand) = '') "
        "   OR (product_name IS NULL OR TRIM(product_name) = '')"
    )
    conn.commit()
    return cur.rowcount


def summary_report(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM invoice_items").fetchone()[0]
    bad_date = conn.execute("SELECT COUNT(*) FROM invoice_items WHERE is_date_valid = 0").fetchone()[0]
    bad_price = conn.execute("SELECT COUNT(*) FROM invoice_items WHERE is_price_valid = 0").fetchone()[0]
    missing = conn.execute(
        "SELECT COUNT(*) FROM invoice_items WHERE cleaning_notes LIKE '%missing_field%'"
    ).fetchone()[0]
    needs_llm = conn.execute(
        "SELECT COUNT(*) FROM invoice_items WHERE is_date_valid = 0 OR is_price_valid = 0 "
        "OR cleaning_notes LIKE '%missing_field%'"
    ).fetchone()[0]
    return {
        "total_rows": total,
        "invalid_date_rows": bad_date,
        "invalid_price_rows": bad_price,
        "missing_field_rows": missing,
        "rows_needing_llm_review": needs_llm,
        "pct_needing_llm_review": round(100 * needs_llm / total, 2) if total else 0,
    }


def main():
    conn = get_connection(DB_PATH)
    n_date = flag_invalid_dates(conn)
    n_price = flag_invalid_prices(conn)
    n_missing = flag_missing_core_fields(conn)
    report = summary_report(conn)

    print("=== KẾT QUẢ RULE-BASED CLEANING ===")
    print(f"Ngày phi lý (trước {MIN_VALID_YEAR} hoặc tương lai): {n_date} dòng")
    print(f"Giá <= 0 hoặc thiếu:                                {n_price} dòng")
    print(f"Thiếu brand/tên sản phẩm:                           {n_missing} dòng")
    print()
    print("=== TỔNG QUAN CHẤT LƯỢNG DỮ LIỆU ===")
    for k, v in report.items():
        print(f"{k:30s}: {v}")
    conn.close()


if __name__ == "__main__":
    main()
