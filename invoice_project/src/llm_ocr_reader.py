"""
llm_ocr_reader.py
--------------------
Dùng GPT-4 Vision (OpenAI API) để đọc ảnh hóa đơn mới, trích xuất dữ liệu có
cấu trúc (JSON) và ghi thẳng vào cùng database SQLite với dữ liệu Excel gốc.

Đây là phần trả lời trực tiếp Mục tiêu cụ thể (iii) trong đề cương:
"Triển khai LLM thông qua API để tự động hóa khâu tiền xử lý, chuẩn hóa dữ
liệu hóa đơn bị lỗi (OCR errors)."

YÊU CẦU:
- Cần có OPENAI_API_KEY trong file .env (xem .env.example)
- Cần môi trường có mạng (chạy trên Google Colab như đề cương đề xuất, hoặc
  máy local có internet). Script KHÔNG chạy được trong môi trường sandbox
  không có mạng.

Cách dùng:
    python src/llm_ocr_reader.py --image_dir data/invoice_images --commit
    (bỏ --commit nếu chỉ muốn xem thử JSON output mà chưa ghi vào DB)
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from db import get_connection, DB_PATH

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o"  # model GPT-4 Vision hiện hành, có thể đổi tùy phiên bản khả dụng

EXTRACTION_PROMPT = """\
Bạn là một hệ thống trích xuất dữ liệu hóa đơn bán lẻ tại Việt Nam.
Hãy đọc ảnh hóa đơn được cung cấp và trả về DUY NHẤT một đối tượng JSON
(không thêm markdown, không giải thích, không có ```), theo đúng cấu trúc sau:

{
  "bill_id": null,
  "brand": "<tên cửa hàng/thương hiệu in trên hóa đơn>",
  "purchase_date": "<YYYY-MM-DD HH:MM:SS, hoặc null nếu không đọc được>",
  "items": [
    {
      "product_name": "<tên sản phẩm/món>",
      "product_type": "<loại sản phẩm, vd: Trà sữa, Nước ngọt, Cà phê...>",
      "price": <số, đơn vị nghìn đồng nếu hóa đơn dùng nghìn đồng, giữ nguyên đơn vị gốc>,
      "quantity": <số lượng, mặc định 1 nếu không ghi rõ>
    }
  ],
  "confidence": <số thực từ 0 đến 1, tự đánh giá độ tin cậy của việc đọc ảnh này>,
  "raw_text_extracted": "<toàn bộ văn bản bạn đọc được trên hóa đơn, giữ nguyên càng sát càng tốt>"
}

QUY TẮC QUAN TRỌNG:
- Nếu chữ mờ/không chắc chắn, vẫn điền giá trị khả dĩ nhất NHƯNG hạ "confidence" xuống thấp.
- Không được bịa số liệu nếu hoàn toàn không đọc được — trong trường hợp đó đặt giá trị null.
- purchase_date phải là ngày thực tế hợp lý (không được là ngày trong tương lai,
  không được là ngày quá xa như 1970).
- Không thêm bất kỳ trường nào ngoài cấu trúc trên.
"""


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_invoice_from_image(image_path: str) -> dict:
    """Gọi GPT-4 Vision để trích xuất dữ liệu có cấu trúc từ 1 ảnh hóa đơn."""
    b64_image = encode_image_base64(image_path)
    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1500,
        temperature=0,  # temperature=0 để kết quả ổn định, ít "sáng tạo" bịa số liệu
    )

    raw_text = response.choices[0].message.content.strip()
    # Phòng trường hợp model vẫn trả về kèm ```json ... ```
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM không trả về JSON hợp lệ cho {image_path}: {e}\nRaw: {raw_text}")

    return data


def save_extraction_to_db(
    conn: sqlite3.Connection,
    image_path: str,
    extracted: dict,
    default_user_id: str = "UNKNOWN_NEW_UPLOAD",
) -> tuple[int, int]:
    """Ghi kết quả trích xuất (có thể nhiều item/hóa đơn) vào database."""
    bill_id = extracted.get("bill_id") or f"LLM_{Path(image_path).stem}"
    brand = extracted.get("brand")
    purchase_date = extracted.get("purchase_date")
    confidence = extracted.get("confidence")
    raw_text = extracted.get("raw_text_extracted")

    inserted, skipped = 0, 0
    cur = conn.cursor()
    for item in extracted.get("items", []):
        try:
            cur.execute(
                """
                INSERT INTO invoice_items
                    (bill_id, user_id, category, product_type, product_name, brand,
                     price, purchase_date, source, raw_ocr_text, llm_model_used, llm_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'llm_ocr', ?, ?, ?)
                """,
                (
                    bill_id, default_user_id, "Beverages", item.get("product_type"),
                    item.get("product_name"), brand, item.get("price"),
                    purchase_date, raw_text, MODEL, confidence,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1
    conn.commit()
    return inserted, skipped


def process_directory(image_dir: str, db_path: str = DB_PATH, commit: bool = False):
    conn = get_connection(db_path)
    image_files = [
        p for p in Path(image_dir).glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]
    print(f"🔎 Tìm thấy {len(image_files)} ảnh hóa đơn trong '{image_dir}'")

    results = []
    for img_path in image_files:
        print(f"\n📄 Đang đọc: {img_path.name}")
        try:
            extracted = extract_invoice_from_image(str(img_path))
            print(json.dumps(extracted, ensure_ascii=False, indent=2))
            results.append({"image": str(img_path), "extracted": extracted})

            if commit:
                ins, skip = save_extraction_to_db(conn, str(img_path), extracted)
                print(f"   ✅ Ghi vào DB: {ins} dòng mới, {skip} dòng trùng bỏ qua")
        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {img_path.name}: {e}")

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Đọc ảnh hóa đơn bằng GPT-4 Vision -> ghi vào SQLite")
    parser.add_argument("--image_dir", required=True, help="Thư mục chứa ảnh hóa đơn mới")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--commit", action="store_true", help="Ghi kết quả vào database (mặc định chỉ preview)")
    args = parser.parse_args()

    process_directory(args.image_dir, args.db, args.commit)


if __name__ == "__main__":
    main()
