"""
run_pipeline.py
-------------------
Chạy toàn bộ pipeline theo đúng thứ tự (trừ các bước cần OpenAI API key +
mạng, sẽ được nhắc riêng). Dùng để demo nhanh toàn bộ luồng xử lý.

Cách dùng:
    python src/run_pipeline.py
"""
import subprocess
import sys


STEPS = [
    ("1. Khởi tạo database", ["python3", "db.py"]),
    ("2. Import Excel gốc -> SQLite", ["python3", "import_excel.py",
                                          "--excel", "../data/hoa_don_goc.xlsx",
                                          "--sheet", "Data Drink"]),
    ("3. Rule-based cleaning (phát hiện lỗi OCR)", ["python3", "clean_rules.py"]),
    ("4. EDA + xuất biểu đồ Dashboard", ["python3", "analysis_eda.py"]),
    ("5. Market Basket Analysis (Apriori)", ["python3", "analysis_market_basket.py",
                                                "--min_support", "0.02", "--min_confidence", "0.2"]),
    ("6. Phân khúc khách hàng (K-Means)", ["python3", "analysis_segmentation.py", "--auto_k"]),
    ("7. Mô hình dự báo (Classification)", ["python3", "analysis_prediction.py",
                                               "--target_brand", "HIGHLANDS COFFEE - MẠC THỊ BƯỞI"]),
]

LLM_STEPS_NOTE = """
--------------------------------------------------------------------------
CÁC BƯỚC CẦN OPENAI API KEY + MẠNG (chạy riêng trên Google Colab/local có internet):

  python src/llm_ocr_reader.py --image_dir data/invoice_images --commit
      -> Đọc ảnh hóa đơn mới bằng GPT-4 Vision, ghi thẳng vào database.

  python src/llm_data_cleaner.py --limit 30 --commit
      -> Dùng LLM chuẩn hóa các dòng dữ liệu đã bị clean_rules.py gắn cờ lỗi.

  python src/eval_llm_accuracy.py evaluate --ground_truth data/ground_truth.csv \\
      --predictions data/llm_predictions.csv
      -> Đánh giá định lượng độ chính xác trích xuất của LLM (cho Chương 4).
--------------------------------------------------------------------------
"""


def main():
    for title, cmd in STEPS:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Bước '{title}' thất bại, dừng pipeline.")
            sys.exit(1)

    print(LLM_STEPS_NOTE)


if __name__ == "__main__":
    main()
