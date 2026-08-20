"""
db.py
------
Định nghĩa & khởi tạo database SQLite dùng chung cho toàn bộ pipeline, dùng
thư viện chuẩn `sqlite3` (không cần cài thêm gói ngoài -> chạy được ngay
trên Google Colab hoặc máy local mà không lo lỗi phụ thuộc).

Vì sao dùng SQLite thay vì tiếp tục dùng Excel làm database:
- Excel bị khóa file khi đang mở, dễ lỗi khi nhiều tiến trình ghi cùng lúc.
- Không có ràng buộc UNIQUE để tự chống trùng lặp hóa đơn khi LLM đọc lại
  1 ảnh nhiều lần.
- Khi dữ liệu lớn dần theo thời gian (liên tục có hóa đơn mới), Excel sẽ chậm.
- Vẫn export ra Excel/CSV bất cứ lúc nào để làm Dashboard / B2B Report.

Bảng chính:
- invoice_items: mỗi dòng là 1 sản phẩm (SKU-level) trích xuất từ hóa đơn.
- extraction_eval_log: log đánh giá độ chính xác trích xuất của LLM so với
  ground truth (phục vụ trả lời Câu hỏi nghiên cứu 1 trong đề cương).
"""

from __future__ import annotations
import sqlite3
import os

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "invoices.db")
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS invoice_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Khóa nghiệp vụ từ hệ thống Vidimi
    bill_id         TEXT NOT NULL,
    user_id         TEXT NOT NULL,

    -- Nhân khẩu học (nếu có)
    gender          TEXT,

    -- Thông tin sản phẩm / SKU-level
    category        TEXT,          -- vd: "Beverages"
    product_type    TEXT,          -- vd: "Nước ngọt", "Trà sữa"
    product_name    TEXT,          -- vd: "Matcha Strawberry Latte"
    brand           TEXT,          -- thương hiệu / cửa hàng

    -- Giao dịch
    price           REAL,
    purchase_date   TEXT,          -- lưu ISO format 'YYYY-MM-DD HH:MM:SS'

    -- Audit / nguồn gốc dữ liệu
    source          TEXT NOT NULL DEFAULT 'excel_import',  -- excel_import | llm_ocr | manual
    raw_ocr_text    TEXT,
    llm_model_used  TEXT,
    llm_confidence  REAL,

    -- Cờ chất lượng dữ liệu (set bởi bước làm sạch)
    is_date_valid   INTEGER DEFAULT 1,   -- 0/1 (SQLite không có kiểu BOOLEAN)
    is_price_valid  INTEGER DEFAULT 1,
    is_cleaned      INTEGER DEFAULT 0,
    cleaning_notes  TEXT,

    created_at      TEXT DEFAULT (datetime('now')),

    UNIQUE(bill_id, product_name, price)
);

CREATE INDEX IF NOT EXISTS ix_brand_date ON invoice_items(brand, purchase_date);
CREATE INDEX IF NOT EXISTS ix_user ON invoice_items(user_id);
CREATE INDEX IF NOT EXISTS ix_bill ON invoice_items(bill_id);

CREATE TABLE IF NOT EXISTS extraction_eval_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_item_id     INTEGER,
    field_name          TEXT,      -- vd: 'brand', 'price', 'purchase_date'
    llm_extracted_value TEXT,
    ground_truth_value  TEXT,
    is_correct          INTEGER,   -- 0/1
    evaluated_at        TEXT DEFAULT (datetime('now'))
);
"""


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


if __name__ == "__main__":
    conn = init_db()
    print(f"✅ Đã khởi tạo database tại: {DB_PATH}")
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Các bảng:", [r["name"] for r in cur.fetchall()])
    conn.close()
