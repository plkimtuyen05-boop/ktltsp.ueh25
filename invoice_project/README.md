# Ứng dụng LLM & Machine Learning trong tiền xử lý và phân tích dữ liệu hóa đơn tiêu dùng
### Nghiên cứu trường hợp Vidimi — Code khóa luận tốt nghiệp

Repo này triển khai đúng luồng công việc mô tả tại **Mục 2.2 (Phương pháp phân tích dữ liệu)**
của đề cương: kết hợp LLM (tiền xử lý) + Machine Learning truyền thống (Apriori, K-Means,
Classification), chạy trên dữ liệu hóa đơn thật của Vidimi (`data/hoa_don_goc.xlsx`, sheet
`Data Drink`, 587 dòng / 105 khách hàng / 392 thương hiệu).

## 1. Kiến trúc dữ liệu

Vấn đề với Excel làm database sống: khóa file khi mở, không constraint chống trùng lặp, chậm
khi liên tục có hóa đơn mới do LLM đọc thêm. Giải pháp:

```
Excel gốc (lịch sử)  ──┐
                        ├──►  SQLite (data/invoices.db)  ──►  Phân tích ML/LLM
Ảnh hóa đơn mới       ──┘        (bảng invoice_items)
(GPT-4 Vision đọc)
```

- **`invoice_items`**: bảng chính, mỗi dòng = 1 sản phẩm (SKU-level) trong 1 hóa đơn.
  Có `UNIQUE(bill_id, product_name, price)` để tự chống trùng khi LLM đọc lại cùng 1 ảnh.
  Có các cờ `is_date_valid`, `is_price_valid`, `is_cleaned` để theo dõi chất lượng dữ liệu
  qua từng bước xử lý.
- **`extraction_eval_log`**: log phục vụ đánh giá độ chính xác LLM so với ground truth.

## 2. Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env      # rồi điền OPENAI_API_KEY thật vào .env
```

> Các bước không dùng LLM (`db.py`, `import_excel.py`, `clean_rules.py`,
> `analysis_*.py`) chỉ cần `pandas`, `scikit-learn`, `matplotlib` — không cần
> API key hay mạng, chạy được ngay cả offline.
> Các bước dùng LLM (`llm_ocr_reader.py`, `llm_data_cleaner.py`) cần
> `OPENAI_API_KEY` + mạng — khuyến nghị chạy trên **Google Colab** đúng như
> đề cương đã chọn làm môi trường thực thi.

## 3. Chạy toàn bộ pipeline

```bash
cd src
python run_pipeline.py
```

Hoặc chạy từng bước (khuyến nghị để hiểu rõ từng giai đoạn):

| Bước | Lệnh | Trả lời câu hỏi nghiên cứu |
|---|---|---|
| 1. Khởi tạo DB | `python db.py` | — |
| 2. Import Excel gốc | `python import_excel.py --excel ../data/hoa_don_goc.xlsx --sheet "Data Drink"` | — |
| 3. Rule-based cleaning | `python clean_rules.py` | CH1 (định lượng % lỗi OCR hiện trạng) |
| 4. **LLM đọc ảnh hóa đơn mới** | `python llm_ocr_reader.py --image_dir ../data/invoice_images --commit` | CH1 |
| 5. **LLM chuẩn hóa dữ liệu lỗi** | `python llm_data_cleaner.py --limit 30 --commit` | CH1 |
| 6. Đánh giá độ chính xác LLM | `python eval_llm_accuracy.py evaluate --ground_truth ... --predictions ...` | CH1 (định lượng) |
| 7. EDA + biểu đồ Dashboard | `python analysis_eda.py` | CH3 |
| 8. Market Basket (Apriori) | `python analysis_market_basket.py` | CH2 |
| 9. Phân khúc khách hàng (K-Means) | `python analysis_segmentation.py --auto_k` | CH2 |
| 10. Mô hình dự báo (Classification) | `python analysis_prediction.py --target_brand "..."` | CH2 |

(Các bước in đậm cần API key + mạng — không chạy được trong môi trường demo hiện tại.)

## 4. Kết quả đã chạy thử trên dữ liệu thật (587 dòng)

**Chất lượng dữ liệu (rule-based, trước khi có LLM):**
- 10/587 dòng (1.7%) có ngày mua phi lý (trước 2015 hoặc trong tương lai) — lỗi OCR điển hình
- 18/587 dòng (3.1%) có giá ≤ 0 — lỗi OCR điển hình
- Tổng: 4.77% dữ liệu cần LLM xem lại → đây là baseline để so sánh "trước/sau khi có LLM" cho Chương 4

**Market Basket Analysis (giỏ hàng = tập thương hiệu/khách hàng, vì hóa đơn hiện là 1
sản phẩm/hóa đơn):** 56/105 khách hàng mua ở ≥2 thương hiệu khác nhau → tìm được 46 luật kết
hợp có ý nghĩa, vd khách mua ở *Highland Coffee* có 66.7% khả năng cũng mua ở *Aeon Tân Phú*
(lift = 17.5).

**Phân khúc khách hàng (K-Means, k=3 tối ưu theo Silhouette):**
- 12 khách VIP (chi tiêu TB 1.192, tần suất 26.7 hóa đơn)
- 89 khách thân thiết mức trung bình/thấp
- 2 khách gần như không quay lại

**Mô hình dự báo:** minh họa được phương pháp luận, nhưng phát hiện quan trọng: với cỡ mẫu
hiện tại (103 khách hàng, 392 thương hiệu), phần lớn thương hiệu chỉ có 1 khách hàng ghi
nhận → **đây là một hạn chế đáng đưa vào Chương 5 (Thảo luận)**: cần Vidimi thu thập thêm
dữ liệu (nhiều người dùng hơn) trước khi mô hình dự báo theo từng thương hiệu khả thi ở mức
production.

## 5. Ánh xạ vào kết cấu khóa luận

- **Chương 3 (Phương pháp NC):** mô tả lại kiến trúc DB, prompt trong
  `llm_ocr_reader.py`/`llm_data_cleaner.py`, công thức Apriori/K-Means dùng.
- **Chương 4 (Kết quả):** dùng trực tiếp số liệu ở mục 4 (README này) + các biểu đồ trong
  `outputs/`.
- **Chương 5 (Thảo luận):** dùng các "GIỚI HẠN CẦN NÊU RÕ" đã comment sẵn trong
  `analysis_prediction.py` và phần cảnh báo cỡ mẫu nhỏ.

## 6. Giới hạn cần nêu rõ trong khóa luận

1. Dữ liệu hiện tại chỉ có ngành hàng **Beverages** — cần mở rộng ngành hàng khác để
   B2B Report đa dạng hơn.
2. Mỗi hóa đơn hiện chỉ có **1 sản phẩm** → Market Basket được thiết kế lại ở cấp độ
   khách hàng (cross-brand) thay vì cấp độ hóa đơn (cross-SKU) truyền thống.
3. Chưa có dữ liệu **đổi voucher/sử dụng điểm thưởng** → phần classification cho bài toán
   này mới ở dạng scaffold (`train_voucher_redemption_model_TEMPLATE` trong
   `analysis_prediction.py`), sẵn sàng chạy khi có dữ liệu thật.
4. Cỡ mẫu 103 khách hàng là nhỏ cho Machine Learning → kết quả mang tính chứng minh
   phương pháp luận (proof-of-concept), khuyến nghị thu thập thêm trước khi triển khai
   production.
