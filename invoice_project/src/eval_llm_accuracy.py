"""
eval_llm_accuracy.py
------------------------
Đánh giá độ chính xác của LLM khi trích xuất/chuẩn hóa dữ liệu hóa đơn, so
với nhãn thực tế (ground truth) do người kiểm tra thủ công gán. Đây là phần
BẮT BUỘC để trả lời Câu hỏi nghiên cứu 1 của đề cương một cách định lượng
("LLM có thể tự động hóa và NÂNG CAO ĐỘ CHÍNH XÁC ... như thế nào?" — cần số
liệu cụ thể, không thể chỉ mô tả định tính).

QUY TRÌNH ĐỀ XUẤT CHO KHÓA LUẬN:
1. Chọn ngẫu nhiên N hóa đơn (ảnh) làm tập test, khuyến nghị N >= 50-100.
2. Con người đọc thủ công và điền đáp án đúng vào file ground_truth.csv
   (mẫu cấu trúc bên dưới, hàm `create_ground_truth_template`).
3. Chạy LLM (llm_ocr_reader.py) trên đúng N ảnh đó.
4. Chạy script này để so khớp kết quả LLM với ground_truth.csv, tính:
   - Accuracy từng trường (brand, product_name, price, purchase_date)
   - Character Error Rate (CER) cho các trường text (đo mức độ "gần đúng")
   - So sánh với baseline OCR truyền thống (nếu công ty đang dùng, vd Tesseract)
     để chứng minh LLM cải thiện bao nhiêu % so với hiện trạng.

Cách dùng:
    python src/eval_llm_accuracy.py --ground_truth data/ground_truth.csv --predictions data/llm_predictions.csv
"""

from __future__ import annotations
import argparse
import pandas as pd
from difflib import SequenceMatcher


GROUND_TRUTH_COLUMNS = [
    "image_filename", "bill_id", "brand", "product_name",
    "product_type", "price", "purchase_date",
]


def create_ground_truth_template(output_path: str, image_filenames: list[str]):
    """Tạo file CSV mẫu để người kiểm tra điền tay đáp án đúng cho từng ảnh."""
    df = pd.DataFrame({"image_filename": image_filenames})
    for col in GROUND_TRUTH_COLUMNS[1:]:
        df[col] = ""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"📝 Đã tạo template ground truth tại: {output_path}")
    print("   -> Điền tay đáp án đúng cho từng ảnh trước khi chạy đánh giá.")


def char_similarity(a: str, b: str) -> float:
    """Độ tương đồng ký tự (0-1), dùng cho các trường text như brand/product_name.
    1.0 = giống hệt, càng gần 0 càng khác biệt nhiều."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).strip().lower(), str(b).strip().lower()).ratio()


def evaluate(ground_truth_path: str, predictions_path: str, text_similarity_threshold: float = 0.85):
    gt = pd.read_csv(ground_truth_path)
    pred = pd.read_csv(predictions_path)

    merged = gt.merge(pred, on="image_filename", suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("Không khớp được image_filename nào giữa ground_truth và predictions. "
                          "Kiểm tra lại 2 file có cùng cột 'image_filename' không.")

    print(f"📊 Đánh giá trên {len(merged)} hóa đơn có đủ ground truth + prediction\n")

    report = []
    exact_match_fields = ["price"]  # trường số phải khớp tuyệt đối
    fuzzy_match_fields = ["brand", "product_name", "product_type"]  # trường text cho phép sai khác nhỏ
    date_fields = ["purchase_date"]

    for field in exact_match_fields:
        col_gt, col_pred = f"{field}_gt", f"{field}_pred"
        if col_gt not in merged.columns or col_pred not in merged.columns:
            continue
        correct = (
            pd.to_numeric(merged[col_gt], errors="coerce").round(2)
            == pd.to_numeric(merged[col_pred], errors="coerce").round(2)
        )
        acc = correct.mean()
        report.append({"field": field, "metric": "exact_match_accuracy", "value": round(acc, 4)})

    for field in fuzzy_match_fields:
        col_gt, col_pred = f"{field}_gt", f"{field}_pred"
        if col_gt not in merged.columns or col_pred not in merged.columns:
            continue
        sims = merged.apply(lambda r: char_similarity(r[col_gt], r[col_pred]), axis=1)
        acc_strict = (sims == 1.0).mean()
        acc_fuzzy = (sims >= text_similarity_threshold).mean()
        report.append({"field": field, "metric": "exact_match_accuracy", "value": round(acc_strict, 4)})
        report.append({"field": field, "metric": f"fuzzy_match_accuracy(>={text_similarity_threshold})",
                        "value": round(acc_fuzzy, 4)})
        report.append({"field": field, "metric": "avg_char_similarity", "value": round(sims.mean(), 4)})

    for field in date_fields:
        col_gt, col_pred = f"{field}_gt", f"{field}_pred"
        if col_gt not in merged.columns or col_pred not in merged.columns:
            continue
        gt_dates = pd.to_datetime(merged[col_gt], errors="coerce")
        pred_dates = pd.to_datetime(merged[col_pred], errors="coerce")
        correct = (gt_dates.dt.date == pred_dates.dt.date)
        report.append({"field": field, "metric": "exact_date_match_accuracy", "value": round(correct.mean(), 4)})

    report_df = pd.DataFrame(report)
    print(report_df.to_string(index=False))
    return report_df


def main():
    parser = argparse.ArgumentParser(description="Đánh giá độ chính xác trích xuất của LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    p_template = sub.add_parser("create_template", help="Tạo file ground truth mẫu")
    p_template.add_argument("--image_dir", required=True)
    p_template.add_argument("--output", default="../data/ground_truth.csv")

    p_eval = sub.add_parser("evaluate", help="So khớp predictions với ground truth")
    p_eval.add_argument("--ground_truth", required=True)
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--output_report", default="../outputs/llm_accuracy_report.csv")

    args = parser.parse_args()

    if args.command == "create_template":
        from pathlib import Path
        files = [p.name for p in Path(args.image_dir).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        create_ground_truth_template(args.output, files)
    elif args.command == "evaluate":
        report_df = evaluate(args.ground_truth, args.predictions)
        report_df.to_csv(args.output_report, index=False, encoding="utf-8-sig")
        print(f"\n💾 Đã lưu báo cáo đánh giá vào: {args.output_report}")


if __name__ == "__main__":
    main()
