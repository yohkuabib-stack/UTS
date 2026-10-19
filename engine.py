"""Data preparation and analysis helpers for the Streamlit BI application."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


RAW_COLS = [
    "Transaction_ID",
    "Transaction_Date",
    "Customer_ID",
    "Region",
    "Customer_Type",
    "Product_ID",
    "Product_Name",
    "Category",
    "Quantity",
    "Unit_Price",
    "Discount_Rate",
    "Line_Total",
    "Payment_Method",
]
MAX_QTY = 20
SILHOUETTE_SAMPLE_SIZE = 500

STRATEGY = {
    "Champions": "Berikan program loyalitas dan penawaran eksklusif; jaga pengalaman belanja pelanggan bernilai tinggi.",
    "Loyal Customers": "Berikan apresiasi loyalitas dan rekomendasi produk pelengkap.",
    "Potential Loyalists": "Tawarkan program keanggotaan dan promosi yang mendorong pembelian berulang.",
    "New Customers": "Sambut dengan edukasi produk dan insentif untuk pembelian kedua.",
    "Promising": "Kirim rekomendasi relevan dan promosi ringan untuk membangun kebiasaan belanja.",
    "Need Attention": "Hubungi dengan penawaran personal sebelum frekuensi belanja semakin turun.",
    "At Risk": "Lakukan kampanye reaktivasi dengan penawaran yang relevan dan tenggat terbatas.",
    "Cannot Lose Them": "Prioritaskan pendekatan personal untuk mempertahankan pelanggan bernilai tinggi.",
    "Hibernating": "Lakukan kampanye win-back berbiaya rendah dan ukur respons sebelum memberi diskon besar.",
    "Lost": "Uji satu penawaran reaktivasi; hentikan kampanye berbiaya tinggi bila tidak ada respons.",
    "Regular Customers": "Dorong peningkatan frekuensi dan nilai belanja melalui rekomendasi produk.",
}


def detect_sheets(sheets):
    """Find transaction and optional reference sheets using their column names."""
    raw = products = customers = None
    for frame in sheets.values():
        columns = set(frame.columns)
        if {"Transaction_ID", "Product_Name", "Quantity"}.issubset(columns):
            raw = frame.copy()
        if {"Product_ID", "Product_Name"}.issubset(columns):
            products = frame.copy()
        if {"Customer_ID", "Region"}.issubset(columns):
            customers = frame.copy()
    return raw, products, customers


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def profile(data):
    """Return the summary metrics displayed by the EDA and preprocessing pages."""
    d = data
    quantity = _numeric(d["Quantity"]) if "Quantity" in d else pd.Series(dtype=float)
    line_total = _numeric(d["Line_Total"]) if "Line_Total" in d else pd.Series(dtype=float)
    return {
        "Baris": int(len(d)),
        "Transaksi unik": int(d["Transaction_ID"].nunique()) if "Transaction_ID" in d else 0,
        "Pelanggan unik": int(d["Customer_ID"].nunique()) if "Customer_ID" in d else 0,
        "Produk unik (nama)": int(d["Product_Name"].nunique()) if "Product_Name" in d else 0,
        "Rata-rata Quantity": float(quantity.mean()) if quantity.notna().any() else 0.0,
        "Total penjualan (Rp)": float(line_total.sum()) if line_total.notna().any() else 0.0,
        "Baris duplikat": int(d.duplicated().sum()),
        "Kategori unik (penulisan)": int(d["Category"].nunique()) if "Category" in d else 0,
        "Metode bayar unik (penulisan)": int(d["Payment_Method"].nunique()) if "Payment_Method" in d else 0,
    }


def iqr_outliers(values):
    """Return lower/upper Tukey fences and the number of values above the upper fence."""
    values = _numeric(pd.Series(values)).dropna()
    if values.empty:
        return 0.0, 0.0, 0
    q1, q3 = values.quantile([0.25, 0.75])
    spread = q3 - q1
    lower, upper = float(q1 - 1.5 * spread), float(q3 + 1.5 * spread)
    return lower, upper, int((values > upper).sum())


def _reference_maps(products, customers):
    product_by_id = {}
    product_by_name = {}
    if products is not None and not products.empty:
        p = products.copy()
        if "Product_ID" in p and "Product_Name" in p:
            product_by_id = p.dropna(subset=["Product_ID"]).drop_duplicates("Product_ID").set_index("Product_ID").to_dict("index")
            product_by_name = p.dropna(subset=["Product_Name"]).drop_duplicates("Product_Name").set_index("Product_Name").to_dict("index")

    customer_by_id = {}
    if customers is not None and not customers.empty and "Customer_ID" in customers:
        customer_by_id = (
            customers.dropna(subset=["Customer_ID"])
            .drop_duplicates("Customer_ID")
            .set_index("Customer_ID")
            .to_dict("index")
        )
    return product_by_id, product_by_name, customer_by_id


def clean_data(raw, products=None, customers=None):
    """Clean transaction rows and add dashboard-friendly date and basket fields."""
    missing = [column for column in RAW_COLS if column not in raw.columns]
    if missing:
        raise ValueError(f"Kolom transaksi tidak lengkap: {', '.join(missing)}")

    d = raw[RAW_COLS].copy()
    rows_raw = len(d)
    log = []

    def record(stage, count, method, reason):
        log.append({"Tahap": stage, "Jumlah": int(count), "Metode": method, "Alasan": reason})

    duplicate_count = int(d.duplicated().sum())
    d = d.drop_duplicates().copy()
    record("Duplikat", duplicate_count, "Hapus baris duplikat persis", "Mencegah transaksi/baris produk dihitung lebih dari sekali.")

    text_columns = ["Transaction_ID", "Customer_ID", "Product_ID", "Product_Name", "Region",
                    "Customer_Type", "Category", "Payment_Method"]
    for column in text_columns:
        if d[column].dtype == object:
            d[column] = d[column].str.strip().replace("", np.nan)
        else:
            d[column] = d[column].replace(r"^\s*$", np.nan, regex=True)

    product_by_id, product_by_name, customer_by_id = _reference_maps(products, customers)
    before = d[["Product_ID", "Product_Name", "Category", "Unit_Price"]].copy()

    if product_by_id:
        product_ref = pd.DataFrame.from_dict(product_by_id, orient="index")
        name_ref = pd.DataFrame.from_dict(product_by_name, orient="index")
        no_id_match = ~d["Product_ID"].isin(product_ref.index)
        for target, source in [
            ("Product_ID", "Product_ID"),
            ("Product_Name", "Product_Name"),
            ("Category", "Category"),
            ("Unit_Price", "Reference_Unit_Price_Rp"),
        ]:
            reference_values = (
                pd.Series(product_ref.index, index=product_ref.index)
                if source == "Product_ID"
                else product_ref[source]
            )
            by_id = d["Product_ID"].map(reference_values)
            name_values = (
                pd.Series(name_ref.index, index=name_ref.index)
                if source == "Product_Name"
                else name_ref[source]
            )
            by_name = d["Product_Name"].map(name_values).where(no_id_match)
            d[target] = d[target].fillna(by_id).fillna(by_name)

    if customer_by_id:
        customer_ref = pd.DataFrame.from_dict(customer_by_id, orient="index")
        for target in ["Region", "Customer_Type"]:
            d[target] = d[target].fillna(d["Customer_ID"].map(customer_ref[target]))

    filled = int(before.isna().sum().sum() - d[["Product_ID", "Product_Name", "Category", "Unit_Price"]].isna().sum().sum())
    record("Missing", max(filled, 0), "Isi dari tabel referensi produk/pelanggan jika tersedia", "Mempertahankan baris yang masih dapat dilengkapi secara konsisten.")

    d["Transaction_Date"] = pd.to_datetime(d["Transaction_Date"], errors="coerce")
    for column in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        d[column] = _numeric(d[column])

    if not d.empty:
        d["Discount_Rate"] = d["Discount_Rate"].where(d["Discount_Rate"].notna(), 0)
        percent_discounts = d["Discount_Rate"].between(1, 100, inclusive="right")
        d.loc[percent_discounts, "Discount_Rate"] /= 100
        invalid_discount = ~d["Discount_Rate"].between(0, 1)
        discount_fixed = int(invalid_discount.sum())
        d.loc[invalid_discount, "Discount_Rate"] = 0
    else:
        discount_fixed = 0

    # Use catalog prices to repair impossible prices; leave valid high-value prices intact.
    catalog_price = d["Product_ID"].map({
        product_id: values.get("Reference_Unit_Price_Rp")
        for product_id, values in product_by_id.items()
    })
    bad_price = d["Unit_Price"].isna() | (d["Unit_Price"] <= 0) | (d["Unit_Price"] > 200000)
    repair_price = bad_price & catalog_price.notna() & (catalog_price > 0)
    d.loc[repair_price, "Unit_Price"] = catalog_price[repair_price]

    valid_required = (
        d["Transaction_ID"].notna()
        & d["Transaction_Date"].notna()
        & d["Customer_ID"].notna()
        & d["Product_Name"].notna()
        & d["Quantity"].between(1, MAX_QTY)
        & d["Unit_Price"].gt(0)
        & d["Unit_Price"].le(200000)
    )
    removed_invalid = int((~valid_required).sum())
    d = d.loc[valid_required].copy()
    record("Nilai tidak wajar", removed_invalid + discount_fixed, "Perbaiki nilai dari referensi; hapus baris yang tidak dapat dipulihkan", f"Quantity harus 1–{MAX_QTY}, tanggal dan ID wajib tersedia, harga harus positif dan masuk akal.")

    if not d.empty:
        d["Category"] = d["Category"].fillna("Tidak diketahui").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.title()
        d["Payment_Method"] = d["Payment_Method"].fillna("Tidak diketahui").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.title()
        d["Region"] = d["Region"].fillna("Tidak diketahui").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        d["Customer_Type"] = d["Customer_Type"].fillna("Tidak diketahui").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        d["Product_Name"] = d["Product_Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        expected_total = d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])
        bad_total = d["Line_Total"].isna() | d["Line_Total"].le(0)
        d.loc[bad_total, "Line_Total"] = expected_total[bad_total]
        record("Konsistensi", int(bad_total.sum()), "Hitung ulang Line_Total yang kosong atau tidak positif", "Line_Total harus konsisten dengan kuantitas, harga, dan diskon.")

        d["Quantity"] = d["Quantity"].astype(int)
        d["Date"] = d["Transaction_Date"]
        d["Bulan"] = d["Date"].dt.to_period("M").astype(str)
        d["Basket_Size"] = d.groupby("Transaction_ID")["Product_Name"].transform("nunique")

    d = d.reset_index(drop=True)
    summary = {"rows_raw": int(rows_raw), "rows_clean": int(len(d))}
    return d, pd.DataFrame(log, columns=["Tahap", "Jumlah", "Metode", "Alasan"]), summary


def build_rfm(clean):
    """Aggregate transaction data into one Recency/Frequency/Monetary row per customer."""
    date_column = "Date" if "Date" in clean.columns else "Transaction_Date"
    d = clean.copy()
    d[date_column] = pd.to_datetime(d[date_column], errors="coerce")
    reference_date = d[date_column].max() + pd.Timedelta(days=1)
    rfm = d.groupby("Customer_ID").agg(
        Last_Purchase=(date_column, "max"),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum"),
    )
    rfm["Recency"] = (reference_date - rfm["Last_Purchase"]).dt.days
    rfm = rfm[["Recency", "Frequency", "Monetary"]].reset_index()
    return rfm


def basket_matrix(clean):
    """Create a transaction-by-product boolean matrix for Apriori."""
    if clean.empty:
        return pd.DataFrame()
    basket = pd.crosstab(clean["Transaction_ID"], clean["Product_Name"]).astype(bool)
    basket.index.name = "Transaction_ID"
    return basket


def scale_rfm(rfm, use_log=True):
    """Optionally log-transform RFM features, then standardize each feature."""
    values = rfm[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)
    values = np.maximum(values, 0)
    if use_log:
        values = np.log1p(values)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    return (values - means) / stds


def _kmeans(X, k):
    return KMeans(n_clusters=int(k), random_state=42, n_init=10).fit(X)


def evaluate_k(X, min_k=2, max_k=10, sample_size=SILHOUETTE_SAMPLE_SIZE):
    """Calculate inertia and sampled silhouette scores for each viable cluster count."""
    X = np.asarray(X, dtype=float)
    if len(X) < 3:
        raise ValueError("Evaluasi K-Means membutuhkan sedikitnya 3 pelanggan.")
    upper = min(int(max_k), len(X) - 1)
    lower = min(int(min_k), upper)
    rows = []
    for k in range(lower, upper + 1):
        model = _kmeans(X, k)
        rows.append({
            "k": k,
            "Inertia (WCSS)": float(model.inertia_),
            "Silhouette": float(silhouette_score(
                X,
                model.labels_,
                sample_size=min(int(sample_size), len(X)),
                random_state=42,
            )),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X, k):
    """Fit deterministic K-Means and return customer cluster labels."""
    X = np.asarray(X, dtype=float)
    if len(X) < int(k):
        raise ValueError(f"Jumlah pelanggan ({len(X)}) lebih kecil dari jumlah klaster ({k}).")
    return _kmeans(X, k).labels_


def label_clusters(rfm, labels):
    """Name clusters using RFM performance relative to the customer medians."""
    grouped = rfm[["Recency", "Frequency", "Monetary"]].copy()
    grouped["Cluster"] = np.asarray(labels)
    means = grouped.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    medians = rfm[["Recency", "Frequency", "Monetary"]].median()
    names = {}
    for cluster, row in means.iterrows():
        recent = row["Recency"] <= medians["Recency"]
        frequent = row["Frequency"] >= medians["Frequency"]
        valuable = row["Monetary"] >= medians["Monetary"]
        if frequent and valuable and recent:
            name = "Champions"
        elif frequent and valuable:
            name = "Cannot Lose Them"
        elif frequent and recent:
            name = "Loyal Customers"
        elif valuable and recent:
            name = "Potential Loyalists"
        elif recent:
            name = "New Customers"
        elif frequent:
            name = "At Risk"
        elif valuable:
            name = "Hibernating"
        else:
            name = "Lost"
        names[cluster] = name
    return names


def run_apriori(basket, min_support, min_confidence, max_len=3):
    """Run a small, deterministic Apriori implementation and generate association rules."""
    if basket.empty:
        return (
            pd.DataFrame(columns=["Itemset", "Support", "Ukuran"]),
            pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Support Consequent"]),
        )

    matrix = basket.to_numpy(dtype=bool)
    n_transactions = len(basket)
    products = list(basket.columns)
    supports = {}
    frequent_by_size = {}
    itemset_rows = []

    for product, column in zip(products, matrix.T):
        support = float(column.mean())
        itemset = (product,)
        if support >= min_support:
            supports[itemset] = support
            frequent_by_size[1] = frequent_by_size.get(1, []) + [itemset]
            itemset_rows.append({"Itemset": product, "Support": support, "Ukuran": 1})

    for size in range(2, max(1, int(max_len)) + 1):
        previous = frequent_by_size.get(size - 1, [])
        if not previous:
            break
        candidates = set()
        previous_set = set(previous)
        for left, right in combinations(previous, 2):
            candidate = tuple(sorted(set(left) | set(right), key=str))
            if len(candidate) != size:
                continue
            if all(tuple(subset) in previous_set for subset in combinations(candidate, size - 1)):
                candidates.add(candidate)

        current = []
        for candidate in sorted(candidates, key=lambda value: tuple(map(str, value))):
            positions = [products.index(item) for item in candidate]
            support = float(matrix[:, positions].all(axis=1).sum() / n_transactions)
            if support >= min_support:
                supports[candidate] = support
                current.append(candidate)
                itemset_rows.append({"Itemset": " + ".join(map(str, candidate)), "Support": support, "Ukuran": size})
        frequent_by_size[size] = current

    rule_rows = []
    for size, itemsets in frequent_by_size.items():
        if size < 2:
            continue
        for itemset in itemsets:
            support = supports[itemset]
            for antecedent_size in range(1, size):
                for antecedent in combinations(itemset, antecedent_size):
                    consequent = tuple(item for item in itemset if item not in antecedent)
                    support_antecedent = supports.get(tuple(antecedent))
                    support_consequent = supports.get(tuple(consequent))
                    if not support_antecedent or not support_consequent:
                        continue
                    confidence = support / support_antecedent
                    if confidence >= min_confidence:
                        rule_rows.append({
                            "Antecedent": " + ".join(map(str, antecedent)),
                            "Consequent": " + ".join(map(str, consequent)),
                            "Support": support,
                            "Confidence": confidence,
                            "Lift": confidence / support_consequent,
                            "Support Consequent": support_consequent,
                        })

    itemsets_frame = pd.DataFrame(itemset_rows, columns=["Itemset", "Support", "Ukuran"])
    if not itemsets_frame.empty:
        itemsets_frame = itemsets_frame.sort_values(["Ukuran", "Support"], ascending=[True, False]).reset_index(drop=True)
    rules_frame = pd.DataFrame(
        rule_rows,
        columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Support Consequent"],
    )
    if not rules_frame.empty:
        rules_frame = rules_frame.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    return itemsets_frame, rules_frame
