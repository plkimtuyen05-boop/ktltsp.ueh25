"""
analysis_prediction.py
--------------------------
Mô hình dự báo (Classification), trả lời phần cuối Mục 2.2.iv đề cương:
"Dự báo khả năng một tập khách hàng sẽ mua một thương hiệu cụ thể hoặc sử
dụng điểm thưởng để đổi voucher."

TRẠNG THÁI DỮ LIỆU HIỆN TẠI (đọc kỹ trước khi dùng cho khóa luận):
- Dữ liệu hiện có KHÔNG có cột "đổi voucher" / "sử dụng điểm thưởng" ->
  bài toán "dự báo đổi voucher" chưa thể huấn luyện với dữ liệu thật. Phần
  `train_voucher_redemption_model()` bên dưới để dạng SCAFFOLD/TEMPLATE, sẵn
  sàng chạy ngay khi Vidimi bổ sung được cột nhãn thực tế (vd `redeemed_voucher`
  trong bảng users hoặc 1 bảng riêng `voucher_events`).
- Bài toán khả thi NGAY với dữ liệu hiện tại: dự báo khách hàng có PHẢI là
  khách hàng của 1 thương hiệu top (vd "Start Coffee") hay không, dựa trên
  hành vi mua sắm tổng thể (RFM + đa dạng thương hiệu/danh mục). Đây là proxy
  hợp lý cho bài toán target-marketing mà brand đối tác B2B thường cần
  ("dự đoán những khách hàng NÀO trong tập dữ liệu Vidimi có khả năng cao sẽ
  là khách hàng của thương hiệu chúng tôi, để nhắm mục tiêu quảng cáo").

GIỚI HẠN CẦN NÊU RÕ TRONG KHÓA LUẬN (Chương 5 - Thảo luận):
- Cỡ mẫu hiện tại nhỏ (103 khách hàng) -> mô hình mang tính minh họa
  phương pháp luận (proof-of-concept), cần thu thập thêm dữ liệu trước khi
  triển khai production.
- Cần đánh giá lại nguy cơ leakage khi brand_diversity được tính trên toàn
  bộ lịch sử (bao gồm cả các giao dịch với chính brand mục tiêu).

Cách dùng:
    python src/analysis_prediction.py --target_brand "Start Coffee"
"""

from __future__ import annotations
import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from db import get_connection, DB_PATH


def build_features_and_target(conn: sqlite3.Connection, target_brand: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT user_id, bill_id, brand, product_type, gender, price, purchase_date
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
        gender=("gender", lambda x: x.mode().iloc[0] if not x.mode().empty else "Không rõ"),
    ).reset_index()
    agg["recency_days"] = (ref_date - agg["last_purchase"]).dt.days
    agg = agg.drop(columns=["last_purchase"])

    target_users = set(df.loc[df["brand"] == target_brand, "user_id"])
    agg["target"] = agg["user_id"].isin(target_users).astype(int)

    agg = pd.get_dummies(agg, columns=["gender"], drop_first=True)
    return agg


def train_brand_purchase_model(df: pd.DataFrame, target_brand: str):
    feature_cols = [c for c in df.columns if c not in ("user_id", "target")]
    X = df[feature_cols].fillna(0)
    y = df["target"]

    n_positive = int(y.sum())
    print(f"🎯 Target: đã từng mua ở '{target_brand}'? "
          f"Positive: {n_positive} / {len(y)} khách hàng ({100*y.mean():.1f}%)")

    if n_positive < 4:
        print(
            f"\n❌ Không đủ dữ liệu để huấn luyện: chỉ có {n_positive} khách hàng từng mua ở "
            f"'{target_brand}'. Cần tối thiểu vài khách hàng ở MỖI lớp (mua/chưa mua) để "
            "train/test split và đánh giá mô hình có ý nghĩa thống kê.\n"
            "💡 Đây bản thân nó là một PHÁT HIỆN đáng đưa vào Chương 4/5 của khóa luận: "
            "với dữ liệu hiện tại (103 khách hàng, 392 thương hiệu), phần lớn thương hiệu "
            "chỉ có 1 khách hàng trung thành ghi nhận -> cần nhiều dữ liệu hơn (nhiều "
            "người dùng chụp hóa đơn hơn) trước khi mô hình dự báo theo từng thương hiệu "
            "khả thi ở mức production. Hãy thử --target_brand với 1 thương hiệu có nhiều "
            "khách hàng khác nhau hơn (in ra bằng: "
            "df.groupby('brand')['user_id'].nunique().sort_values(ascending=False))."
        )
        return None

    if n_positive < 8 or (len(y) - n_positive) < 8:
        print("⚠️  Cảnh báo: cỡ mẫu positive/negative vẫn khá nhỏ -> kết quả dưới đây "
              "CHỈ mang tính minh họa phương pháp luận (proof-of-concept), chưa đủ tin "
              "cậy để triển khai thực tế. Cần thu thập thêm dữ liệu.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42,
        stratify=y if y.sum() >= 2 and (len(y) - y.sum()) >= 2 else None,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        if y_proba is not None and len(set(y_test)) > 1:
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC: {auc:.3f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        if name == "Random Forest":
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            print("\nMức độ quan trọng của đặc trưng (feature importance):")
            print(importances.sort_values(ascending=False).round(3).to_string())

        results[name] = model

    return results


def train_voucher_redemption_model_TEMPLATE():
    """
    SCAFFOLD — chưa có dữ liệu thật để chạy.
    Khi Vidimi có bảng ghi nhận sự kiện đổi voucher (vd cột `redeemed_voucher`
    0/1 gắn với mỗi user, hoặc bảng `voucher_events(user_id, voucher_id, redeemed_at)`),
    chỉ cần:
      1. JOIN bảng đó với `agg` (kết quả của build_features_and_target, bỏ cột 'target' cũ)
      2. Đặt agg['target'] = agg['user_id'].isin(users_da_doi_voucher)
      3. Gọi lại train_brand_purchase_model(agg, target_brand="(đổi voucher)")

    Việc này không cần viết lại pipeline — chỉ cần thay nguồn nhãn (label source).
    """
    raise NotImplementedError(
        "Chưa có cột dữ liệu 'đổi voucher' trong database hiện tại. "
        "Xem docstring hàm này để biết cách bổ sung khi có dữ liệu thật."
    )


def main():
    parser = argparse.ArgumentParser(description="Mô hình dự báo khả năng mua 1 thương hiệu cụ thể")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--target_brand", default="Start Coffee",
                         help="Thương hiệu mục tiêu để dự báo (mặc định: thương hiệu xuất hiện nhiều nhất)")
    args = parser.parse_args()

    conn = get_connection(args.db)
    df = build_features_and_target(conn, args.target_brand)
    train_brand_purchase_model(df, args.target_brand)
    conn.close()


if __name__ == "__main__":
    main()
