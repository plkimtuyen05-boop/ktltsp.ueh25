"""
analysis_market_basket.py
----------------------------
Phân tích giỏ hàng (Market Basket Analysis) bằng thuật toán Apriori, trả lời
Câu hỏi nghiên cứu 2 của đề cương: "Những quy luật mua sắm nào có thể khai
phá từ dữ liệu hóa đơn?"

LƯU Ý QUAN TRỌNG VỀ THIẾT KẾ:
Dữ liệu hiện tại của Vidimi là hóa đơn 1-sản-phẩm/hóa-đơn (mỗi lần khách chụp
1 hóa đơn lẻ để tích điểm), KHÁC với market basket truyền thống (nhiều sản
phẩm trong CÙNG 1 hóa đơn siêu thị). Do đó "giỏ hàng" ở đây được định nghĩa
lại ở CẤP ĐỘ KHÁCH HÀNG: giỏ hàng của 1 user = tập hợp các THƯƠNG HIỆU mà
user đó đã mua trong suốt lịch sử. Điều này đúng với tinh thần đề cương nêu
rõ: "tỷ lệ khách hàng mua sản phẩm của thương hiệu A đồng thời mua sản phẩm
của thương hiệu đối thủ B" — tức là phân tích mua CHÉO thương hiệu, không
phải mua cùng lúc trong 1 hóa đơn.

Nếu dữ liệu tương lai có hóa đơn nhiều sản phẩm/hóa đơn, chỉ cần đổi
group_col='bill_id' thay vì 'user_id' để có market basket truyền thống.

Cách dùng:
    python src/analysis_market_basket.py --min_support 0.03 --min_confidence 0.3
"""

from __future__ import annotations
import argparse
import sqlite3
import pandas as pd
from itertools import combinations
from db import get_connection, DB_PATH

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


def build_basket_matrix(conn: sqlite3.Connection, group_col: str = "user_id") -> pd.DataFrame:
    """Ma trận nhị phân (0/1): mỗi dòng là 1 khách hàng, mỗi cột là 1 thương hiệu."""
    df = pd.read_sql(
        f"SELECT {group_col}, brand FROM invoice_items WHERE brand IS NOT NULL AND TRIM(brand) != ''",
        conn,
    )
    transactions = df.groupby(group_col)["brand"].apply(lambda x: list(set(x)))

    if HAS_MLXTEND:
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_array, columns=te.columns_, index=transactions.index)
    else:
        all_brands = sorted(set(b for lst in transactions for b in lst))
        basket_df = pd.DataFrame(0, index=transactions.index, columns=all_brands)
        for idx, brands in transactions.items():
            basket_df.loc[idx, brands] = 1
        basket_df = basket_df.astype(bool)

    return basket_df


def run_apriori_mlxtend(basket_df: pd.DataFrame, min_support: float, min_confidence: float):
    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return frequent_itemsets, rules


def run_apriori_manual(basket_df: pd.DataFrame, min_support: float, min_confidence: float, max_len: int = 2):
    """
    Cài đặt Apriori thủ công (fallback khi không có mlxtend) — chỉ tính tới
    itemset độ dài 2 (cặp thương hiệu), đủ để trả lời câu hỏi "khách mua A
    có xu hướng mua thêm B không?" mà đề cương đặt ra.
    """
    n_transactions = len(basket_df)
    item_counts = basket_df.sum(axis=0)
    frequent_items = item_counts[item_counts / n_transactions >= min_support].index.tolist()

    # Giới hạn số lượng brand để tránh bùng nổ tổ hợp (C(n,2) quá lớn)
    frequent_items = item_counts[frequent_items].sort_values(ascending=False).head(60).index.tolist()

    rules = []
    for a, b in combinations(frequent_items, 2):
        support_a = item_counts[a] / n_transactions
        support_b = item_counts[b] / n_transactions
        support_ab = (basket_df[a] & basket_df[b]).sum() / n_transactions
        if support_ab == 0:
            continue
        conf_a_to_b = support_ab / support_a if support_a > 0 else 0
        conf_b_to_a = support_ab / support_b if support_b > 0 else 0
        lift = support_ab / (support_a * support_b) if support_a * support_b > 0 else 0

        if conf_a_to_b >= min_confidence:
            rules.append({"antecedent": a, "consequent": b, "support": support_ab,
                          "confidence": conf_a_to_b, "lift": lift})
        if conf_b_to_a >= min_confidence:
            rules.append({"antecedent": b, "consequent": a, "support": support_ab,
                          "confidence": conf_b_to_a, "lift": lift})

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(["lift", "confidence"], ascending=False)

    itemsets_df = pd.DataFrame({
        "support": item_counts[frequent_items] / n_transactions,
        "itemsets": [{i} for i in frequent_items],
    }).reset_index(drop=True)

    return itemsets_df, rules_df


def main():
    parser = argparse.ArgumentParser(description="Market Basket Analysis (Apriori) theo thương hiệu")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--group_col", default="user_id", choices=["user_id", "bill_id"])
    parser.add_argument("--min_support", type=float, default=0.03)
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--output", default="../outputs/market_basket_rules.csv")
    args = parser.parse_args()

    conn = get_connection(args.db)
    basket_df = build_basket_matrix(conn, args.group_col)
    print(f"📊 Ma trận giỏ hàng: {basket_df.shape[0]} khách hàng x {basket_df.shape[1]} thương hiệu")

    if HAS_MLXTEND:
        print("✅ Dùng mlxtend.apriori")
        itemsets, rules = run_apriori_mlxtend(basket_df, args.min_support, args.min_confidence)
    else:
        print("⚠️  Không tìm thấy mlxtend -> dùng cài đặt Apriori thủ công (fallback, chỉ cặp 2 brand)")
        itemsets, rules = run_apriori_manual(basket_df, args.min_support, args.min_confidence)

    print(f"\nSố tập phổ biến (frequent itemsets): {len(itemsets)}")
    print(f"Số luật kết hợp (association rules) thỏa min_confidence={args.min_confidence}: {len(rules)}")

    if not rules.empty:
        print("\n🔝 Top 10 luật mua chéo mạnh nhất (theo lift):")
        print(rules.head(10).to_string(index=False))
        rules.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n💾 Đã lưu đầy đủ luật vào: {args.output}")
    else:
        print("\n⚠️  Không tìm được luật nào thỏa ngưỡng. Thử giảm --min_support / --min_confidence.")

    conn.close()


if __name__ == "__main__":
    main()
