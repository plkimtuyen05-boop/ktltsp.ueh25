"""
llm_data_cleaner.py
----------------------
Dùng LLM (Prompt Engineering) để chuẩn hóa lại các dòng dữ liệu đã được
clean_rules.py gắn cờ là nghi ngờ lỗi (ngày phi lý, giá bằng 0, brand/tên
sản phẩm viết tắt/lộn xộn do OCR). Đây chính là kỹ thuật "LLM for Data
Preprocessing" mô tả trong Mục 2.2.i của đề cương.

Khác với llm_ocr_reader.py (đọc TRỰC TIẾP từ ảnh), module này làm việc trên
dữ liệu VĂN BẢN đã có sẵn trong DB — rẻ hơn nhiều vì không cần gửi ảnh, chỉ
cần gửi text. Phù hợp để dọn dẹp một lượng lớn dữ liệu lịch sử.

Cách dùng:
    python src/llm_data_cleaner.py --limit 30 --commit
"""

from __future__ import annotations
import argparse
import json
import os
import sqlite3

from dotenv import load_dotenv
from openai import OpenAI

from db import get_connection, DB_PATH

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"  # dùng bản mini cho tác vụ text-cleaning để tiết kiệm chi phí

CLEANING_PROMPT_TEMPLATE = """\
Bạn là hệ thống chuẩn hóa dữ liệu hóa đơn bán lẻ Việt Nam. Dưới đây là MỘT
dòng dữ liệu hóa đơn có khả năng bị lỗi do quá trình OCR (đọc ảnh tự động).

Dữ liệu gốc (JSON):
{row_json}

Ghi chú lỗi được hệ thống rule-based phát hiện: {cleaning_notes}

Nhiệm vụ của bạn:
1. Nếu "purchase_date" phi lý (trước năm 2015, hoặc là ngày trong tương lai,
   hoặc null) và KHÔNG THỂ suy luận được ngày đúng từ dữ liệu hiện có, hãy
   đặt "purchase_date_fixed": null và giải thích ngắn gọn trong "reasoning".
   TUYỆT ĐỐI KHÔNG bịa ra một ngày ngẫu nhiên.
2. Nếu "price" <= 0 hoặc null, hãy đánh giá xem có thể suy luận giá hợp lý
   không (vd dựa trên product_name/brand quen thuộc). Nếu không chắc chắn,
   đặt "price_fixed": null.
3. Chuẩn hóa "brand" và "product_name": sửa lỗi chính tả rõ ràng do OCR (vd
   viết hoa/thường lộn xộn, ký tự thừa), KHÔNG đổi ý nghĩa/thương hiệu gốc.
4. Trả về DUY NHẤT JSON theo cấu trúc, không thêm markdown:

{{
  "brand_fixed": "<brand đã chuẩn hóa, hoặc giữ nguyên nếu đã đúng>",
  "product_name_fixed": "<tên sản phẩm đã chuẩn hóa>",
  "purchase_date_fixed": "<YYYY-MM-DD HH:MM:SS hoặc null>",
  "price_fixed": <số hoặc null>,
  "confidence": <0-1>,
  "reasoning": "<giải thích ngắn gọn các thay đổi>"
}}
"""


def fetch_rows_needing_review(conn: sqlite3.Connection, limit: int = 50):
    cur = conn.execute(
        """
        SELECT id, bill_id, brand, product_name, product_type, price,
               purchase_date, cleaning_notes
        FROM invoice_items
        WHERE (is_date_valid = 0 OR is_price_valid = 0
               OR cleaning_notes LIKE '%missing_field%')
          AND is_cleaned = 0
        LIMIT ?
        """,
        (limit,),
    )
    return cur.fetchall()


def clean_row_with_llm(row: sqlite3.Row) -> dict:
    row_dict = {
        "brand": row["brand"],
        "product_name": row["product_name"],
        "product_type": row["product_type"],
        "price": row["price"],
        "purchase_date": row["purchase_date"],
    }
    prompt = CLEANING_PROMPT_TEMPLATE.format(
        row_json=json.dumps(row_dict, ensure_ascii=False),
        cleaning_notes=row["cleaning_notes"] or "(không có ghi chú)",
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def apply_fix(conn: sqlite3.Connection, row_id: int, fix: dict):
    conn.execute(
        """
        UPDATE invoice_items
        SET brand = ?, product_name = ?, purchase_date = ?, price = ?,
            is_cleaned = 1,
            cleaning_notes = COALESCE(cleaning_notes || '; ', '') || ?
        WHERE id = ?
        """,
        (
            fix.get("brand_fixed"), fix.get("product_name_fixed"),
            fix.get("purchase_date_fixed"), fix.get("price_fixed"),
            f"llm_cleaned(conf={fix.get('confidence')}): {fix.get('reasoning', '')}",
            row_id,
        ),
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Dùng LLM để chuẩn hóa dữ liệu hóa đơn bị lỗi")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--commit", action="store_true", help="Ghi kết quả sửa vào DB")
    args = parser.parse_args()

    conn = get_connection(args.db)
    rows = fetch_rows_needing_review(conn, args.limit)
    print(f"🔎 Có {len(rows)} dòng cần LLM xem lại (giới hạn {args.limit})")

    for row in rows:
        print(f"\n--- Dòng id={row['id']} (bill_id={row['bill_id']}) ---")
        try:
            fix = clean_row_with_llm(row)
            print(json.dumps(fix, ensure_ascii=False, indent=2))
            if args.commit:
                apply_fix(conn, row["id"], fix)
                print("   ✅ Đã cập nhật vào DB")
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")

    conn.close()


if __name__ == "__main__":
    main()
