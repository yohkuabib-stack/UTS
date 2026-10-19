# -*- coding: utf-8 -*-
"""
SISTEM ANALISIS DATA MINING & BUSINESS INTELLIGENCE - UTS BUSINESS INTELLIGENCE
Yohana Antanik Kuabib | NPM 51230082 | Universitas Timor

SATU FILE: backend, mesin analisis, dan seluruh tampilan (HTML/CSS/JS, foto, logo) ada di sini.

Cara menjalankan:
    pip install flask pandas numpy scikit-learn openpyxl
    python uts_bi.py
Lalu buka http://127.0.0.1:5082  |  username: Yohana  |  password: 51230082
"""
import base64
import io
import os
import re
import sys
import uuid
import zlib
from itertools import combinations

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_file, session
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =============================================================== BAGIAN 1: MESIN ANALISIS


REQUIRED_COLS = [
    "Transaction_ID", "Transaction_Date", "Customer_ID", "Region", "Customer_Type",
    "Product_ID", "Product_Name", "Category", "Quantity", "Unit_Price",
    "Discount_Rate", "Line_Total", "Payment_Method",
]
QTY_CAP = 50  # batas kewajaran jumlah beli per baris (ritel eceran)
UNKNOWN = "Tidak Diketahui"
NO_CUST = "Tidak Teridentifikasi"


# --------------------------------------------------------------------------- util
def rp(x):
    try:
        return "Rp {:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return "-"


def num(x, d=0):
    try:
        return ("{:,.%df}" % d).format(float(x)).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def clean_json(o):
    """Ubah numpy/pandas/NaN menjadi tipe JSON yang aman."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d %H:%M")
    if isinstance(o, pd.Period):
        return str(o)
    if o is pd.NaT:
        return None
    if isinstance(o, pd.Series):
        return clean_json(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return clean_json(o.to_dict(orient="records"))
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
    return o


def df_records(df, cols=None, n=None):
    d = df if cols is None else df[cols]
    if n:
        d = d.head(n)
    return clean_json(d.astype(object).where(d.notna(), None).to_dict(orient="records"))


# --------------------------------------------------------------------------- load
def load_workbook(file_storage, filename):
    """Baca file unggahan -> dict {raw, produk, pelanggan, sheets}."""
    name = filename.lower()
    out = {"produk": None, "pelanggan": None, "sheets": []}
    if name.endswith(".csv"):
        raw = pd.read_csv(file_storage)
        out["sheets"] = [{"nama": "CSV", "baris": len(raw), "kolom": raw.shape[1]}]
    else:
        xl = pd.ExcelFile(file_storage)
        names = xl.sheet_names
        raw_name = "Data_Raw" if "Data_Raw" in names else names[0]
        raw = xl.parse(raw_name)
        for nm in names:
            try:
                n_rows = xl.parse(nm).shape
            except Exception:
                n_rows = (0, 0)
            out["sheets"].append({"nama": nm, "baris": int(n_rows[0]), "kolom": int(n_rows[1]),
                                  "dipakai": nm == raw_name or nm in ("Data_Produk", "Data_Pelanggan")})
        if "Data_Produk" in names:
            out["produk"] = xl.parse("Data_Produk")
        if "Data_Pelanggan" in names:
            out["pelanggan"] = xl.parse("Data_Pelanggan")
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))
    out["raw"] = raw[REQUIRED_COLS].copy()
    return out


def build_catalog(raw, produk):
    """Katalog produk: dari sheet Data_Produk, atau diturunkan dari data jika tidak ada."""
    if produk is not None and {"Product_ID", "Product_Name", "Category", "Reference_Unit_Price_Rp"} <= set(produk.columns):
        cat = produk[["Product_ID", "Product_Name", "Category", "Reference_Unit_Price_Rp"]].copy()
        cat.columns = ["Product_ID", "Product_Name", "Category", "Ref_Price"]
        return cat, "sheet Data_Produk"
    d = raw.dropna(subset=["Product_ID", "Product_Name"]).copy()
    d["Category"] = d["Category"].astype(str).str.strip().str.title()
    valid_price = d[(d["Unit_Price"] > 0)]
    price = valid_price.groupby("Product_ID")["Unit_Price"].median().rename("Ref_Price")
    cat = (d.groupby("Product_ID").agg(Product_Name=("Product_Name", lambda s: s.mode().iat[0]),
                                       Category=("Category", lambda s: s.mode().iat[0])).join(price).reset_index())
    return cat, "diturunkan dari data (modus/median)"


# --------------------------------------------------------------------------- EDA
def _iqr(s):
    s = s.dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(q1), float(q3), float(lo), float(hi), int(((s < lo) | (s > hi)).sum())


def eda(raw, catalog):
    d = raw.copy()
    d["Transaction_Date"] = pd.to_datetime(d["Transaction_Date"], errors="coerce")
    n_rows, n_cols = d.shape
    dtypes = []
    for c in d.columns:
        dtypes.append({"kolom": c, "tipe": str(d[c].dtype), "terisi": int(d[c].notna().sum()),
                       "kosong": int(d[c].isna().sum()), "unik": int(d[c].nunique(dropna=True)),
                       "contoh": str(d[c].dropna().iloc[0]) if d[c].notna().any() else "-"})
    tmin, tmax = d["Transaction_Date"].min(), d["Transaction_Date"].max()
    valid_q = d["Quantity"][(d["Quantity"] > 0) & (d["Quantity"] <= QTY_CAP)]
    stats = {
        "baris": n_rows, "kolom": n_cols,
        "tanggal_awal": tmin.strftime("%d %b %Y"), "tanggal_akhir": tmax.strftime("%d %b %Y"),
        "rentang_hari": int((tmax - tmin).days) + 1,
        "transaksi_unik": int(d["Transaction_ID"].nunique()),
        "pelanggan_unik": int(d["Customer_ID"].nunique()),
        "produk_unik": int(d["Product_ID"].nunique()),
        "rata_qty_mentah": float(d["Quantity"].mean()),
        "rata_qty_valid": float(valid_q.mean()),
        "median_qty": float(d["Quantity"].median()),
        "total_penjualan_mentah": float(d["Line_Total"].sum()),
        "rata_item_per_transaksi": float(n_rows / d["Transaction_ID"].nunique()),
    }
    miss = [{"kolom": c, "jumlah": int(d[c].isna().sum()), "persen": float(d[c].isna().mean() * 100)}
            for c in d.columns if d[c].isna().sum() > 0]
    dup_full = int(d.duplicated().sum())
    dup_key = int(d.drop_duplicates().duplicated(["Transaction_ID", "Product_ID"]).sum())

    # inkonsistensi teks
    cat_ok = set(catalog["Category"].unique())
    cat_bad = d["Category"].astype(str).value_counts()
    cat_variants = [{"nilai": repr(k), "jumlah": int(v)} for k, v in cat_bad.items() if k not in cat_ok]
    pay = d["Payment_Method"].astype("string")
    pay_counts = pay.value_counts(dropna=False)
    pay_variants = [{"nilai": repr(k) if not pd.isna(k) else "(kosong)", "jumlah": int(v)} for k, v in pay_counts.items()]
    pay_key = pay.str.strip().str.lower().nunique()

    # outlier
    q1, q3, lo, hi, n_iqr_q = _iqr(d["Quantity"])
    _, _, plo, phi, n_iqr_p = _iqr(d["Unit_Price"])
    out = {
        "qty_nonpositif": int((d["Quantity"] <= 0).sum()),
        "qty_ekstrem": int((d["Quantity"] > QTY_CAP).sum()),
        "qty_iqr_batas_atas": hi, "qty_iqr_jumlah": n_iqr_q,
        "qty_maks_wajar": float(valid_q.max()),
        "harga_nonpositif": int((d["Unit_Price"] <= 0).sum()),
        "harga_ekstrem": int((d["Unit_Price"] > 10 * catalog["Ref_Price"].max()).sum()),
        "harga_iqr_jumlah": n_iqr_p,
    }
    # visual
    d["YM"] = d["Transaction_Date"].dt.to_period("M").astype(str)
    monthly = d.groupby("YM").agg(penjualan=("Line_Total", "sum"), transaksi=("Transaction_ID", "nunique")).reset_index()
    top_prod = (d.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False).head(10)).reset_index()
    d["Cat_norm"] = d["Category"].astype(str).str.strip().str.title()
    cat_sales = d.groupby("Cat_norm")["Line_Total"].sum().sort_values(ascending=False).reset_index()
    qty_hist = d["Quantity"].dropna().clip(upper=10).round().astype(int).value_counts().sort_index()
    qty_hist = [{"qty": ("≥10" if k >= 10 else str(k)), "jumlah": int(v)} for k, v in qty_hist.items()]

    # temuan awal (otomatis dari data)
    mm = monthly.loc[monthly["penjualan"].idxmax()]
    ml = monthly.loc[monthly["penjualan"].idxmin()]
    top1 = top_prod.iloc[0]
    share_top = float(top_prod["Line_Total"].sum() / d["Line_Total"].sum() * 100)
    cat1 = cat_sales.iloc[0]
    mem = d.dropna(subset=["Customer_Type"]).groupby("Customer_Type")["Transaction_ID"].nunique()
    n_missing_cells = int(d.isna().sum().sum())
    findings = [
        f"Kualitas data belum siap dianalisis: {num(n_missing_cells)} sel kosong, {num(dup_full)} baris duplikat, "
        f"{num(out['qty_nonpositif'] + out['qty_ekstrem'])} nilai Quantity tidak valid dan {num(out['harga_nonpositif'] + out['harga_ekstrem'])} nilai Unit_Price tidak valid. "
        "Preprocessing wajib dilakukan sebelum Apriori dan K-Means.",
        f"Penjualan bulanan tertinggi terjadi pada {mm['YM']} ({rp(mm['penjualan'])}), terendah pada {ml['YM']} ({rp(ml['penjualan'])}); "
        "fluktuasi ini perlu diperhatikan untuk perencanaan stok dan promosi.",
        f"Produk dengan penjualan tertinggi adalah {top1['Product_Name']} ({rp(top1['Line_Total'])}); 10 produk teratas menyumbang {num(share_top, 1)}% dari total penjualan "
        f"dan kategori terbesar adalah {cat1['Cat_norm']} ({num(cat1['Line_Total'] / d['Line_Total'].sum() * 100, 1)}%).",
        f"Satu transaksi rata-rata berisi {num(stats['rata_item_per_transaksi'], 2)} baris produk, sehingga analisis keranjang belanja (Apriori) layak dilakukan.",
    ]
    if len(mem) == 2:
        findings.append(f"Transaksi Member = {num(mem.get('Member', 0))} dan Non-Member = {num(mem.get('Non-Member', 0))}; "
                        "porsi kedua tipe pelanggan relatif seimbang sehingga program keanggotaan punya ruang untuk ditingkatkan.")
    return {
        "stats": stats, "dtypes": dtypes, "missing": miss, "dup_full": dup_full, "dup_key": dup_key,
        "cat_variants": cat_variants, "pay_variants": pay_variants, "pay_unik_setelah_normalisasi": int(pay_key),
        "outlier": out, "monthly": monthly, "top_products": top_prod, "cat_sales": cat_sales,
        "qty_hist": qty_hist, "findings": findings,
        "preview": df_records(raw, n=8),
    }


# --------------------------------------------------------------------------- preprocessing
def preprocess(raw, produk, pelanggan):
    """Bersihkan data. Data_Raw tidak ditimpa (bekerja pada salinan)."""
    catalog, cat_src = build_catalog(raw, produk)
    df = raw.copy()
    n0 = len(df)
    log = []
    changed = {}

    def add(masalah, jumlah, tindakan, alasan):
        log.append({"no": len(log) + 1, "masalah": masalah, "jumlah": int(jumlah), "tindakan": tindakan,
                    "alasan": alasan, "baris_setelah": int(len(df))})

    # sebelum: ringkasan kondisi awal
    def snapshot(x, tag):
        return {
            "kondisi": tag, "baris": int(len(x)), "kolom": int(x.shape[1]),
            "sel_kosong": int(x.isna().sum().sum()),
            "baris_duplikat": int(x.duplicated().sum()),
            "qty_tidak_valid": int(((x["Quantity"] <= 0) | (x["Quantity"] > QTY_CAP) | x["Quantity"].isna()).sum()),
            "harga_tidak_valid": int(((x["Unit_Price"] <= 0) | x["Unit_Price"].isna() | (x["Unit_Price"] > 10 * catalog["Ref_Price"].max())).sum()),
            "kategori_unik": int(x["Category"].nunique()),
            "metode_bayar_unik": int(x["Payment_Method"].nunique()),
            "transaksi_unik": int(x["Transaction_ID"].nunique()),
            "pelanggan_unik": int(x["Customer_ID"].nunique()),
            "total_penjualan": float(x["Line_Total"].sum()),
            "rata_qty": float(x["Quantity"].mean()),
        }
    before = snapshot(raw, "Sebelum")

    # 1. tanggal
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    bad_date = int(df["Transaction_Date"].isna().sum())
    df = df.dropna(subset=["Transaction_Date"])
    add("Format tanggal tidak valid / tidak terbaca", bad_date,
        "Konversi ke datetime; baris yang gagal dikonversi dihapus",
        "Tanggal dibutuhkan untuk tren bulanan dan Recency; nilai tidak terbaca tidak dapat diperbaiki.")

    # 2. duplikat
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    add("Baris duplikat persis (seluruh kolom sama)", dup, "Hapus duplikat, simpan kemunculan pertama",
        "Duplikat menggandakan penjualan dan frekuensi; satu nota tidak mungkin mencatat baris identik dua kali.")

    # 3. teks tidak konsisten
    for c in ["Region", "Customer_Type", "Customer_ID", "Product_ID", "Product_Name", "Payment_Method", "Category"]:
        df[c] = df[c].astype("string").str.strip()
    cat_ok = set(catalog["Category"].unique())
    n_cat_bad = int((~df["Category"].isin(cat_ok)).sum())
    # Pembayaran: seragamkan huruf besar-kecil mengikuti bentuk paling sering
    key = df["Payment_Method"].str.lower()
    canon = df.groupby(key)["Payment_Method"].agg(lambda s: s.mode().iat[0])
    # bentuk baku = bentuk dengan kapitalisasi yang paling umum (QRIS, Tunai, Debit, Transfer)
    pay_raw_variants = raw["Payment_Method"].astype("string")
    n_pay_bad = int((pay_raw_variants.notna() & (pay_raw_variants != pay_raw_variants.str.strip().map(
        lambda v: v if pd.isna(v) else canon.get(v.lower(), v)))).sum())
    df["Payment_Method"] = key.map(canon)
    changed["Category"] = n_cat_bad
    changed["Payment_Method"] = n_pay_bad
    add("Teks kategori tidak konsisten (huruf besar/kecil, spasi berlebih)", n_cat_bad,
        "Strip spasi lalu petakan Category dari katalog produk (Product_ID/Product_Name)",
        "Katalog adalah sumber kebenaran; varian ejaan membuat satu kategori terhitung beberapa.")
    add("Metode pembayaran tidak konsisten (QRIS/Qris/qris/'QRIS ')", n_pay_bad,
        "Strip spasi dan seragamkan ke satu ejaan baku",
        "Menghindari satu metode terpecah menjadi beberapa kategori pada tabel/grafik.")

    # 4. Product_ID kosong -> dari Product_Name
    name2id = catalog.drop_duplicates("Product_Name").set_index("Product_Name")["Product_ID"]
    m = df["Product_ID"].isna()
    df.loc[m, "Product_ID"] = df.loc[m, "Product_Name"].map(name2id)
    n_pid = int(m.sum())
    still = int(df["Product_ID"].isna().sum())
    df = df.dropna(subset=["Product_ID"])
    changed["Product_ID"] = n_pid
    # sinkronkan nama/kategori dari katalog
    id2name = catalog.set_index("Product_ID")["Product_Name"]
    id2cat = catalog.set_index("Product_ID")["Category"]
    df["Product_Name"] = df["Product_ID"].map(id2name).fillna(df["Product_Name"])
    df["Category"] = df["Product_ID"].map(id2cat).fillna(df["Category"].astype("string").str.title())
    add("Product_ID kosong", n_pid, f"Isi dari Product_Name melalui katalog ({cat_src}); {still} baris yang tetap tidak dikenali dihapus",
        "Nama produk selalu terisi dan 1:1 dengan Product_ID, sehingga pengisian dari katalog akurat (bukan tebakan).")

    # 5. Customer_ID kosong
    n_c = int(df["Customer_ID"].isna().sum())
    fill = df.groupby("Transaction_ID")["Customer_ID"].transform(lambda s: s.dropna().iat[0] if s.notna().any() else pd.NA)
    df["Customer_ID"] = df["Customer_ID"].fillna(fill)
    n_c_left = int(df["Customer_ID"].isna().sum())
    df["Customer_ID"] = df["Customer_ID"].fillna(NO_CUST)
    changed["Customer_ID"] = n_c
    add("Customer_ID kosong", n_c,
        f"Isi dari baris lain pada Transaction_ID yang sama ({n_c - n_c_left} terisi); sisanya {n_c_left} diberi label '{NO_CUST}' "
        "dan dikecualikan dari RFM/K-Means",
        "Satu nota hanya dimiliki satu pelanggan. ID tidak boleh ditebak/diimputasi dengan modus karena akan mengacaukan RFM.")

    # 6. Region & Customer_Type kosong
    n_r = int(df["Region"].isna().sum()); n_t = int(df["Customer_Type"].isna().sum())
    if pelanggan is not None and {"Customer_ID", "Region", "Customer_Type"} <= set(pelanggan.columns):
        mp_r = pelanggan.set_index("Customer_ID")["Region"]; mp_t = pelanggan.set_index("Customer_ID")["Customer_Type"]
        src = "master Data_Pelanggan"
    else:
        known = df[df["Customer_ID"] != NO_CUST]
        mp_r = known.dropna(subset=["Region"]).groupby("Customer_ID")["Region"].agg(lambda s: s.mode().iat[0])
        mp_t = known.dropna(subset=["Customer_Type"]).groupby("Customer_ID")["Customer_Type"].agg(lambda s: s.mode().iat[0])
        src = "modus per pelanggan dari data"
    df["Region"] = df["Region"].fillna(df["Customer_ID"].map(mp_r)).fillna(UNKNOWN)
    df["Customer_Type"] = df["Customer_Type"].fillna(df["Customer_ID"].map(mp_t)).fillna(UNKNOWN)
    changed["Region"] = n_r; changed["Customer_Type"] = n_t
    add("Region dan Customer_Type kosong", n_r + n_t, f"Isi berdasarkan Customer_ID dari {src}; yang tidak ditemukan diberi '{UNKNOWN}'",
        "Atribut pelanggan bersifat tetap per pelanggan sehingga dapat dipulihkan secara pasti.")

    # 7. Payment kosong
    n_p = int(df["Payment_Method"].isna().sum())
    df["Payment_Method"] = df["Payment_Method"].fillna(UNKNOWN)
    add("Payment_Method kosong", n_p, f"Isi dengan '{UNKNOWN}' (tidak diimputasi modus)",
        "Metode bayar tidak dipakai di Apriori/K-Means; imputasi modus akan memalsukan distribusi metode bayar.")

    # 8. diskon tidak valid
    bad_disc = int(((df["Discount_Rate"] < 0) | (df["Discount_Rate"] > 0.5) | df["Discount_Rate"].isna()).sum())
    df.loc[(df["Discount_Rate"] < 0) | (df["Discount_Rate"] > 0.5) | df["Discount_Rate"].isna(), "Discount_Rate"] = 0.0
    add("Discount_Rate tidak valid (<0, >50%, kosong)", bad_disc, "Set 0 (tanpa diskon)",
        "Diskon wajar berada pada 0-15%; nilai di luar itu dianggap salah input sehingga tidak diberi diskon.")

    # 9. harga tidak valid
    ref = df["Product_ID"].map(catalog.set_index("Product_ID")["Ref_Price"])
    bad_price = df["Unit_Price"].isna() | (df["Unit_Price"] <= 0) | (df["Unit_Price"] > 5 * ref) | (df["Unit_Price"] < 0.2 * ref)
    n_price = int(bad_price.sum())
    df["Flag_Harga_Diperbaiki"] = bad_price
    df.loc[bad_price, "Unit_Price"] = ref[bad_price]
    changed["Unit_Price"] = n_price
    add("Unit_Price kosong / negatif / tidak wajar (mis. -100, 999.999)", n_price,
        "Ganti dengan harga referensi katalog produk",
        "Pada baris valid, 100% Unit_Price sama dengan harga referensi katalog, sehingga harga katalog adalah nilai pengganti paling tepat.")

    # 10. quantity tidak valid
    bad_q = df["Quantity"].isna() | (df["Quantity"] <= 0) | (df["Quantity"] > QTY_CAP)
    n_q = int(bad_q.sum())
    est = df["Line_Total"] / (df["Unit_Price"] * (1 - df["Discount_Rate"]))
    ok_rec = bad_q & est.notna() & (est.round() >= 1) & (est.round() <= QTY_CAP) & ((est - est.round()).abs() < 0.01)
    df["Flag_Qty_Diperbaiki"] = ok_rec
    df.loc[ok_rec, "Quantity"] = est[ok_rec].round()
    n_rec = int(ok_rec.sum())
    drop_q = df["Quantity"].isna() | (df["Quantity"] <= 0) | (df["Quantity"] > QTY_CAP)
    n_drop_q = int(drop_q.sum())
    df = df[~drop_q].copy()
    changed["Quantity"] = n_rec
    add("Quantity kosong / nol / negatif / 999", n_q,
        f"Pulihkan dari Line_Total ÷ (Unit_Price × (1 − diskon)) → {n_rec} baris pulih; {n_drop_q} baris yang tidak dapat dipulihkan dihapus",
        "Line_Total konsisten 100% dengan rumus pada baris normal, sehingga dapat dipakai membalik hitungan Quantity. 999 adalah kode placeholder, bukan pembelian nyata.")

    # 11. outlier kontekstual (IQR vs konteks bisnis)
    q1, q3, lo, hi, n_iqr = _iqr(df["Quantity"])
    add(f"Nilai ekstrem Quantity menurut IQR (batas atas {num(hi,1)})", n_iqr,
        f"Dipertahankan (Quantity maksimum {num(df['Quantity'].max(),0)}); hanya nilai > {QTY_CAP} dianggap tidak wajar",
        "Membeli 4-6 unit barang eceran adalah perilaku normal pelanggan (stok rumah tangga). Menghapusnya menghilangkan pembeli bernilai tinggi; "
        "batas IQR bukan batas bisnis.")

    # 12. Line_Total
    calc = (df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])).round(2)
    lt_miss = int(df["Line_Total"].isna().sum())
    lt_bad = int((df["Line_Total"].notna() & ((df["Line_Total"] - calc).abs() > 1)).sum())
    df["Flag_Total_Diperbaiki"] = df["Line_Total"].isna() | ((df["Line_Total"] - calc).abs() > 1)
    df["Line_Total"] = calc
    changed["Line_Total"] = lt_miss + lt_bad
    add("Line_Total kosong / tidak sesuai rumus Quantity × Harga × (1 − Diskon)", lt_miss + lt_bad,
        f"Hitung ulang seluruh Line_Total dengan rumus ({lt_miss} kosong, {lt_bad} tidak sesuai)",
        "Total harus konsisten dengan komponennya agar KPI penjualan dan Monetary akurat.")

    # 12b. duplikat tersembunyi: muncul setelah nilai rusak diperbaiki
    key_cols = REQUIRED_COLS
    dup2 = int(df.duplicated(key_cols).sum())
    df = df.drop_duplicates(key_cols).reset_index(drop=True)
    add("Duplikat tersembunyi (baris kembar yang salah satunya rusak; baru terlihat setelah perbaikan)", dup2,
        "Hapus setelah nilai diperbaiki, simpan kemunculan pertama",
        "Sebelum diperbaiki, kedua baris tampak berbeda karena salah satu nilainya kosong/rusak; setelah diperbaiki keduanya identik.")

    # 13. transformasi
    df["Quantity"] = df["Quantity"].astype(int)
    df["Unit_Price"] = df["Unit_Price"].astype(float)
    df["Tanggal"] = df["Transaction_Date"].dt.normalize()
    df["Year_Month"] = df["Transaction_Date"].dt.to_period("M").astype(str)
    df["Hour"] = df["Transaction_Date"].dt.hour
    for c in ["Region", "Customer_Type", "Customer_ID", "Product_ID", "Product_Name", "Payment_Method", "Category"]:
        df[c] = df[c].astype(str)
    add("Transformasi tipe & fitur turunan", len(df),
        "Tanggal → datetime, Quantity → integer, harga → float, tambah Year_Month dan Hour; kolom flag untuk baris yang diperbaiki",
        "Format yang konsisten memudahkan agregasi bulanan dan Recency; flag menjaga transparansi perubahan.")

    # 14. duplikat (Transaction_ID, Product_ID)
    dk = int(df.duplicated(["Transaction_ID", "Product_ID"]).sum())
    add("Produk yang sama muncul >1 kali dalam satu Transaction_ID (bukan duplikat persis)", dk,
        "Dipertahankan; basket_size dan matriks Apriori memakai produk unik per transaksi",
        "Barisnya punya nilai berbeda (mis. Quantity) sehingga bukan duplikasi; dampak pada keranjang dinetralkan oleh penghitungan produk unik.")

    df = df.reset_index(drop=True)
    after = snapshot(df.assign(), "Sesudah")
    after["kolom"] = int(df.shape[1])
    after["sel_kosong"] = int(df[REQUIRED_COLS].isna().sum().sum())
    after["baris_duplikat"] = int(df[REQUIRED_COLS].duplicated().sum())
    after["qty_tidak_valid"] = int(((df["Quantity"] <= 0) | (df["Quantity"] > QTY_CAP)).sum())
    after["harga_tidak_valid"] = int((df["Unit_Price"] <= 0).sum())
    after["pelanggan_unik"] = int((df["Customer_ID"] != NO_CUST).sum() and df.loc[df["Customer_ID"] != NO_CUST, "Customer_ID"].nunique())

    removed = n0 - len(df)
    documentation = {
        "baris_awal": n0, "baris_dipertahankan": int(len(df)), "baris_dihapus": int(removed),
        "persen_dipertahankan": float(len(df) / n0 * 100),
        "dihapus_rinci": [
            {"alasan": "Duplikat persis", "jumlah": dup},
            {"alasan": "Tanggal tidak valid", "jumlah": bad_date},
            {"alasan": "Product_ID tak dikenali", "jumlah": still},
            {"alasan": "Quantity tidak valid & tidak dapat dipulihkan", "jumlah": n_drop_q},
            {"alasan": "Duplikat tersembunyi (setelah perbaikan)", "jumlah": dup2},
        ],
        "diubah_rinci": [{"kolom": k, "sel_diubah": int(v)} for k, v in changed.items() if v],
        "baris_flag_harga": int(df["Flag_Harga_Diperbaiki"].sum()),
        "baris_flag_qty": int(df["Flag_Qty_Diperbaiki"].sum()),
        "baris_flag_total": int(df["Flag_Total_Diperbaiki"].sum()),
        "pelanggan_tak_teridentifikasi_baris": int((df["Customer_ID"] == NO_CUST).sum()),
        "sumber_katalog": cat_src,
    }
    return {"clean": df, "catalog": catalog, "log": log, "before": before, "after": after, "doc": documentation}


# --------------------------------------------------------------------------- features
def build_features(df):
    """basket_size, RFM, matriks Apriori, matriks K-Means (terstandardisasi)."""
    tx = (df.groupby("Transaction_ID").agg(
        Transaction_Date=("Transaction_Date", "min"), Customer_ID=("Customer_ID", "first"),
        basket_size=("Product_ID", "nunique"), jumlah_baris=("Product_ID", "size"),
        total_qty=("Quantity", "sum"), total_belanja=("Line_Total", "sum")).reset_index())
    ref_date = df["Tanggal"].max() + pd.Timedelta(days=1)
    known = df[df["Customer_ID"] != NO_CUST]
    rfm = known.groupby("Customer_ID").agg(
        last=("Tanggal", "max"), Frequency=("Transaction_ID", "nunique"), Monetary=("Line_Total", "sum")).reset_index()
    rfm["Recency"] = (ref_date - rfm["last"]).dt.days
    rfm = rfm[["Customer_ID", "Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    onehot = pd.crosstab(df["Transaction_ID"], df["Product_Name"]).clip(upper=1).astype(bool)
    return {"tx": tx, "rfm": rfm, "ref_date": ref_date, "scaler": scaler, "X": X, "onehot": onehot}


# --------------------------------------------------------------------------- Apriori
def apriori(onehot, min_support=0.01, max_len=3):
    n = len(onehot)
    items = list(onehot.columns)
    mat = onehot.values
    sup1 = mat.mean(axis=0)
    freq = {(i,): float(s) for i, s in enumerate(sup1) if s >= min_support}
    allf = dict(freq)
    cur = freq
    k = 2
    while k <= max_len and cur:
        keys = sorted(cur)
        cand = set()
        for a, b in combinations(keys, 2):
            if a[:-1] == b[:-1]:
                c = a + (b[-1],)
                if all(tuple(x for x in c if x != y) in cur for y in c):
                    cand.add(c)
        nxt = {}
        for c in cand:
            s = float(mat[:, list(c)].all(axis=1).mean())
            if s >= min_support:
                nxt[c] = s
        allf.update(nxt)
        cur = nxt
        k += 1
    return items, allf, n


def association_rules(items, allf, min_conf=0.2, rank_by="lift"):
    rules = []
    for iset, s in allf.items():
        if len(iset) < 2:
            continue
        for r in range(1, len(iset)):
            for ant in combinations(iset, r):
                cons = tuple(x for x in iset if x not in ant)
                sa, sc = allf.get(ant), allf.get((cons[0],) if len(cons) == 1 else cons)
                if not sa or not sc:
                    continue
                conf = s / sa
                if conf < min_conf:
                    continue
                lift = conf / sc
                lev = s - sa * sc
                conv = (1 - sc) / (1 - conf) if conf < 1 else np.inf
                rules.append({
                    "antecedent": " + ".join(items[i] for i in ant), "consequent": " + ".join(items[i] for i in cons),
                    "ant_items": [items[i] for i in ant], "cons_items": [items[i] for i in cons],
                    "support": s, "confidence": conf, "lift": lift, "leverage": lev,
                    "conviction": conv if np.isfinite(conv) else None, "n_items": len(iset)})
    if not rules:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    r = pd.DataFrame(rules)
    key = {"lift": ["lift", "confidence", "support"], "confidence": ["confidence", "lift", "support"],
           "support": ["support", "lift", "confidence"]}[rank_by]
    return r.sort_values(key, ascending=False).reset_index(drop=True)


def run_apriori(onehot, min_support, min_conf, max_len=3, top_n=10, rank_by="lift"):
    items, allf, n = apriori(onehot, min_support, max_len)
    rules = association_rules(items, allf, min_conf, rank_by)
    pos = rules[rules["lift"] > 1].copy() if len(rules) else rules
    top = pos.head(top_n)
    fi = [{"itemset": " + ".join(items[i] for i in k), "ukuran": len(k), "support": v, "jumlah_transaksi": int(round(v * n))}
          for k, v in allf.items()]
    fi = sorted(fi, key=lambda x: (-(x["ukuran"] > 1), -x["support"]))
    # heatmap lift pasangan produk (15 produk terpopuler)
    sup = onehot.mean(axis=0).sort_values(ascending=False)
    top_items = list(sup.head(15).index)
    M = onehot[top_items].values.astype(float)
    co = (M.T @ M) / n
    s = np.array([sup[i] for i in top_items])
    lift = co / np.outer(s, s)
    np.fill_diagonal(lift, np.nan)
    return {
        "n_transaksi": n, "n_produk": len(items), "n_frequent": len(allf),
        "n_frequent_by_size": {str(k): int(sum(1 for x in allf if len(x) == k)) for k in range(1, max_len + 1)},
        "n_rules": int(len(rules)), "n_rules_lift_gt1": int(len(pos)),
        "frequent_itemsets": fi[:20], "rules_top": top, "rules_all": rules,
        "heat_items": top_items, "heat_lift": lift,
        "support_stat": {"min": float(sup.min()), "median": float(sup.median()), "max": float(sup.max())},
    }


# --------------------------------------------------------------------------- K-Means
def _knee(ks, inertia):
    x = np.array(ks, float); y = np.array(inertia, float)
    xn = (x - x.min()) / (x.max() - x.min()); yn = (y - y.min()) / (y.max() - y.min())
    d = np.abs(yn - (1 - xn)) / np.sqrt(2)  # jarak ke garis lurus dari titik awal ke akhir
    return int(ks[int(np.argmax(d))])


def kmeans_eval(X, kmin=2, kmax=8):
    ks = list(range(kmin, kmax + 1))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(X, km.labels_)))
    k_elbow = _knee(ks, inertia)
    k_sil = ks[int(np.argmax(sil))]
    # rekomendasi: silhouette tertinggi pada rentang di sekitar siku (k_elbow-1 .. k_elbow+1), minimal 3 segmen
    rng = [k for k in range(k_elbow - 1, k_elbow + 2) if k in ks and k >= 3] or [k_elbow]
    k_rec = max(rng, key=lambda k: sil[ks.index(k)])
    return {"ks": ks, "inertia": inertia, "silhouette": sil, "k_elbow": k_elbow, "k_silhouette": k_sil,
            "k_rekomendasi": int(k_rec), "rentang_siku": rng}


STRATEGI = {
    "Loyal Customers": {
        "strategi": "Program loyalitas & penghargaan: poin/tier member, akses awal produk baru, bundling premium. Jaga frekuensi, jangan beri diskon besar yang menggerus margin.",
        "kanal": "Kartu member, WhatsApp pribadi"},
    "Potential Customers": {
        "strategi": "Dorong naik kelas: promo 'beli lagi dalam 14 hari', bundling dari aturan Apriori, upgrade ke member. Targetkan menambah frekuensi dan nilai keranjang.",
        "kanal": "Promo di kasir, voucher terbatas waktu"},
    "Churn Risk": {
        "strategi": "Reaktivasi: voucher 'kami rindu Anda' berbatas waktu, survei singkat penyebab berhenti, tawaran produk yang dulu sering dibeli.",
        "kanal": "WhatsApp/SMS terarah"},
    "Occasional Customers": {
        "strategi": "Naikkan frekuensi dan nilai keranjang: bundling dari aturan Apriori, promo 'belanja kedua', rekomendasi produk pendamping di kasir, dan ajakan menjadi Member.",
        "kanal": "Struk belanja, promo di kasir"},
}


def label_clusters(rfm_k):
    ov = rfm_k[["Recency", "Frequency", "Monetary"]]
    mu, sd = ov.mean(), ov.std(ddof=0).replace(0, 1)
    prof = rfm_k.groupby("Cluster").agg(
        Pelanggan=("Customer_ID", "count"), Recency_mean=("Recency", "mean"), Recency_median=("Recency", "median"),
        Frequency_mean=("Frequency", "mean"), Frequency_median=("Frequency", "median"),
        Monetary_mean=("Monetary", "mean"), Monetary_median=("Monetary", "median"), Monetary_total=("Monetary", "sum")).reset_index()
    prof["r_z"] = (prof["Recency_mean"] - mu["Recency"]) / sd["Recency"]
    prof["f_z"] = (prof["Frequency_mean"] - mu["Frequency"]) / sd["Frequency"]
    prof["m_z"] = (prof["Monetary_mean"] - mu["Monetary"]) / sd["Monetary"]
    labels = []
    for _, r in prof.iterrows():
        if r["r_z"] > 0.5:
            lab = "Churn Risk"
        elif r["f_z"] > 0.3 and r["m_z"] > 0.3:
            lab = "Loyal Customers"
        elif r["f_z"] > 0 or r["m_z"] > 0:
            lab = "Potential Customers"
        else:
            lab = "Occasional Customers"
        labels.append(lab)
    prof["Base"] = labels
    prof["val"] = prof["f_z"] + prof["m_z"] - prof["r_z"]
    # label unik
    lab_final = {}
    for base, grp in prof.groupby("Base"):
        grp = grp.sort_values("val", ascending=False)
        for i, (_, r) in enumerate(grp.iterrows()):
            lab_final[int(r["Cluster"])] = base if len(grp) == 1 else f"{base} {chr(65 + i)}"
    prof["Segmen"] = prof["Cluster"].map(lab_final)
    prof["Share_Pelanggan"] = prof["Pelanggan"] / prof["Pelanggan"].sum() * 100
    prof["Share_Revenue"] = prof["Monetary_total"] / prof["Monetary_total"].sum() * 100

    def desc(r):
        return (f"Rata-rata terakhir belanja {num(r['Recency_mean'],0)} hari lalu "
                f"({'lebih lama' if r['r_z'] > 0 else 'lebih baru'} dari rata-rata pelanggan {num(ov['Recency'].mean(),0)} hari), "
                f"{num(r['Frequency_mean'],1)} transaksi ({'di atas' if r['f_z'] > 0 else 'di bawah'} rata-rata {num(ov['Frequency'].mean(),1)}) "
                f"dan belanja {rp(r['Monetary_mean'])} ({'di atas' if r['m_z'] > 0 else 'di bawah'} rata-rata {rp(ov['Monetary'].mean())}).")
    prof["Karakteristik"] = prof.apply(desc, axis=1)
    prof["Strategi"] = prof["Base"].map(lambda b: STRATEGI[b]["strategi"])
    prof["Kanal"] = prof["Base"].map(lambda b: STRATEGI[b]["kanal"])
    prof = prof.sort_values("Cluster").reset_index(drop=True)
    return prof, lab_final


def run_kmeans(feat, k=None):
    X, rfm = feat["X"], feat["rfm"]
    ev = kmeans_eval(X)
    k_use = int(k) if k else ev["k_rekomendasi"]
    km = KMeans(n_clusters=k_use, n_init=10, random_state=42).fit(X)
    rfm_k = rfm.copy()
    rfm_k["Cluster"] = km.labels_
    prof, lab = label_clusters(rfm_k)
    rfm_k["Segmen"] = rfm_k["Cluster"].map(lab)
    sil_k = float(silhouette_score(X, km.labels_))
    # prioritas segmen: segmen non-loyal dengan kontribusi belanja terbesar (masih punya ruang tumbuh/dipertahankan)
    cand = prof[prof["Base"] != "Loyal Customers"]
    pr = (cand if len(cand) else prof).sort_values("Monetary_total", ascending=False).iloc[0]
    tipe = {"Churn Risk": "dipertahankan sebelum hilang", "Potential Customers": "ditingkatkan nilainya",
            "Occasional Customers": "ditingkatkan frekuensi dan nilai keranjangnya"}.get(pr["Base"], "ditingkatkan nilainya")
    alasan = (f"Segmen {pr['Segmen']} berisi {num(pr['Pelanggan'])} pelanggan ({num(pr['Share_Pelanggan'], 1)}%) yang menyumbang "
              f"{num(pr['Share_Revenue'], 1)}% total belanja ({rp(pr['Monetary_total'])}). Di antara segmen yang belum loyal, inilah yang paling besar kontribusinya "
              f"dan masih bisa {tipe}, sehingga tindakan pemasaran memberi dampak terbesar.")
    risk = prof[prof["Base"] == "Churn Risk"]
    if len(risk) and pr["Base"] != "Churn Risk":
        rr = risk.sort_values("Monetary_total", ascending=False).iloc[0]
        alasan += (f" Segmen {rr['Segmen']} ({num(rr['Pelanggan'])} pelanggan, {num(rr['Share_Revenue'], 1)}% belanja) paling mendesak secara waktu "
                   f"karena terakhir belanja rata-rata {num(rr['Recency_mean'])} hari lalu, tetapi kontribusi belanjanya lebih kecil.")
    # titik pusat dalam satuan asli
    cent = feat["scaler"].inverse_transform(km.cluster_centers_)
    cents = pd.DataFrame(cent, columns=["Recency", "Frequency", "Monetary"])
    cents["Cluster"] = range(k_use)
    cents["Segmen"] = cents["Cluster"].map(lab)
    return {"eval": ev, "k": k_use, "silhouette_k": sil_k, "rfm_k": rfm_k, "profile": prof, "prioritas": {"segmen": pr["Segmen"], "alasan": alasan},
            "centroid": cents, "labels": lab}


# --------------------------------------------------------------------------- dashboard
def apply_filters(df, rfm_k, start=None, end=None, categories=None, segments=None):
    d = df
    if start:
        d = d[d["Tanggal"] >= pd.to_datetime(start)]
    if end:
        d = d[d["Tanggal"] <= pd.to_datetime(end)]
    if categories:
        d = d[d["Category"].isin(categories)]
    if segments and rfm_k is not None:
        ids = rfm_k.loc[rfm_k["Segmen"].isin(segments), "Customer_ID"]
        d = d[d["Customer_ID"].isin(ids)]
    return d


def dashboard(df, rfm_k, prof, params, start=None, end=None, categories=None, segments=None):
    d = apply_filters(df, rfm_k, start, end, categories, segments)
    if d.empty:
        return {"empty": True}
    n_tx = d["Transaction_ID"].nunique()
    basket = d.groupby("Transaction_ID")["Product_ID"].nunique()
    known = d[d["Customer_ID"] != NO_CUST]
    kpi = {"total_penjualan": float(d["Line_Total"].sum()), "transaksi": int(n_tx),
           "pelanggan": int(known["Customer_ID"].nunique()), "avg_basket": float(basket.mean()),
           "aov": float(d["Line_Total"].sum() / n_tx), "baris": int(len(d))}
    monthly = d.groupby("Year_Month").agg(penjualan=("Line_Total", "sum"), transaksi=("Transaction_ID", "nunique")).reset_index()
    top_p = d.groupby("Product_Name").agg(penjualan=("Line_Total", "sum"), qty=("Quantity", "sum")).sort_values("penjualan", ascending=False).head(10).reset_index()
    cat = d.groupby("Category")["Line_Total"].sum().sort_values(ascending=False).reset_index()
    reg = d.groupby("Region")["Line_Total"].sum().sort_values(ascending=False).reset_index()
    # Apriori pada data terfilter
    oh = pd.crosstab(d["Transaction_ID"], d["Product_Name"]).clip(upper=1).astype(bool)
    ap = run_apriori(oh, params["min_support"], params["min_conf"], params.get("max_len", 3), 10, params.get("rank_by", "lift")) if len(oh) >= 30 else None
    # K-Means pada data terfilter
    ids = known["Customer_ID"].unique()
    seg_f = rfm_k[rfm_k["Customer_ID"].isin(ids)]
    cnt = seg_f.groupby("Segmen").size().reset_index(name="pelanggan")
    prof_f = (seg_f.groupby("Segmen").agg(Pelanggan=("Customer_ID", "count"), Recency=("Recency", "mean"),
                                           Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean")).reset_index())
    # Bonus: aturan per segmen
    bonus = []
    seg_names = sorted(rfm_k["Segmen"].unique())
    for sg in seg_names:
        ids_s = rfm_k.loc[rfm_k["Segmen"] == sg, "Customer_ID"]
        ds = d[d["Customer_ID"].isin(ids_s)]
        if ds["Transaction_ID"].nunique() < 30:
            continue
        ohs = pd.crosstab(ds["Transaction_ID"], ds["Product_Name"]).clip(upper=1).astype(bool)
        a = run_apriori(ohs, params["min_support"], params["min_conf"], 2, 3, "lift")
        topp = ds.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False).head(3)
        bonus.append({"segmen": sg, "transaksi": int(ds["Transaction_ID"].nunique()),
                      "produk_top": [{"produk": k, "penjualan": float(v)} for k, v in topp.items()],
                      "rules": df_records(a["rules_top"][["antecedent", "consequent", "support", "confidence", "lift"]]) if len(a["rules_top"]) else []})
    return {"empty": False, "kpi": kpi, "monthly": monthly, "top_products": top_p, "cat": cat, "region": reg,
            "apriori": None if ap is None else {"rules": ap["rules_top"], "n_tx": ap["n_transaksi"], "n_rules": ap["n_rules_lift_gt1"]},
            "seg_count": cnt, "seg_profile": prof_f, "bonus": bonus}


# --------------------------------------------------------------------------- rekomendasi
def recommendations(df, feat, ap, km, assump):
    tx = feat["tx"]; rfm_k = km["rfm_k"]; prof = km["profile"]
    n_tx = len(tx)
    total = float(df["Line_Total"].sum())
    top_rules = ap["rules_top"].head(5)
    avg_line = df.groupby("Product_Name")["Line_Total"].mean()
    adopt, disc, react, voucher, uplift, margin = (assump[k] for k in ["adopt", "bundle_disc", "react", "voucher", "uplift", "margin"])

    # A. penempatan/bundling
    placement, seen = [], set()
    for _, r in top_rules.iterrows():
        pair = tuple(sorted([r["antecedent"], r["consequent"]]))
        if pair in seen:
            continue
        seen.add(pair)
        na = r["support"] / r["confidence"] * n_tx
        placement.append({
            "a": r["antecedent"], "b": r["consequent"], "support": r["support"], "confidence": r["confidence"], "lift": r["lift"],
            "bukti": f"{r['antecedent']} → {r['consequent']}: support {num(r['support'] * 100, 2)}% ({num(round(r['support'] * n_tx))} transaksi), "
                     f"confidence {num(r['confidence'] * 100, 1)}%, lift {num(r['lift'], 2)}. Pembeli {r['antecedent']} {num(r['lift'], 2)}× lebih mungkin "
                     f"membeli {r['consequent']} dibanding pembeli acak.",
            "tindakan": f"Paket bundling '{r['antecedent']} + {r['consequent']}' dan tempatkan berdekatan di rak/etalase.",
            "na": float(na)})
    # estimasi bundling
    b_rev = 0.0
    for p in placement[:3]:
        b = p["b"].split(" + ")[0]
        b_rev += p["na"] * (1 - p["confidence"]) * adopt * float(avg_line.get(b, avg_line.mean()))
    b_cost = b_rev * disc
    b_profit = b_rev * margin - b_cost

    # B. retensi
    churn = prof[prof["Base"] == "Churn Risk"].sort_values("Monetary_total", ascending=False)
    loyal = prof[prof["Base"] == "Loyal Customers"].sort_values("Monetary_total", ascending=False)
    ch = churn.iloc[0] if len(churn) else prof.sort_values("Recency_mean", ascending=False).iloc[0]
    lo = loyal.iloc[0] if len(loyal) else prof.sort_values("Monetary_total", ascending=False).iloc[0]
    r_rev = ch["Pelanggan"] * react * ch["Monetary_mean"]
    r_cost = ch["Pelanggan"] * react * voucher  # voucher hanya dibiayai bila ditukar (pelanggan yang kembali)
    r_profit = r_rev * margin - r_cost
    l_rev = lo["Monetary_total"] * uplift
    l_cost = lo["Pelanggan"] * voucher * 0.5
    l_profit = l_rev * margin - l_cost
    retention = [
        {"segmen": ch["Segmen"], "bukti": f"{int(ch['Pelanggan'])} pelanggan; terakhir belanja rata-rata {num(ch['Recency_mean'],0)} hari lalu; belanja total {rp(ch['Monetary_total'])} ({num(ch['Share_Revenue'],1)}% dari total belanja pelanggan teridentifikasi).",
         "tindakan": "Kirim voucher reaktivasi berbatas waktu + survei singkat; prioritaskan pelanggan dengan Monetary tertinggi di segmen ini."},
        {"segmen": lo["Segmen"], "bukti": f"{int(lo['Pelanggan'])} pelanggan; {num(lo['Frequency_mean'],1)} transaksi dan {rp(lo['Monetary_mean'])} per pelanggan; menyumbang {num(lo['Share_Revenue'],1)}% total belanja.",
         "tindakan": "Program poin/tier member dan akses awal promo untuk menjaga frekuensi tanpa diskon besar."},
    ]
    # C. prioritas
    items = [
        {"nama": "Reaktivasi segmen berisiko churn", "urgensi": 5, "manfaat": 4, "biaya": 2, "kemudahan": 4,
         "alasan": f"Pelanggan {ch['Segmen']} makin lama makin jauh (Recency {num(ch['Recency_mean'],0)} hari); semakin lama dibiarkan, semakin kecil peluang kembali."},
        {"nama": "Bundling & penempatan produk berdasarkan aturan Apriori", "urgensi": 3, "manfaat": 4, "biaya": 1, "kemudahan": 5,
         "alasan": "Hanya butuh penataan rak dan paket harga; tidak butuh sistem baru; berbasis aturan dengan lift > 1."},
        {"nama": "Program loyalitas untuk segmen bernilai tinggi", "urgensi": 3, "manfaat": 5, "biaya": 3, "kemudahan": 3,
         "alasan": f"Segmen {lo['Segmen']} menyumbang {num(lo['Share_Revenue'],1)}% belanja; kehilangan sedikit pelanggan saja berdampak besar."},
        {"nama": "Konversi pelanggan Non-Member menjadi Member", "urgensi": 2, "manfaat": 3, "biaya": 2, "kemudahan": 4,
         "alasan": "Memudahkan identifikasi pelanggan sehingga data RFM lebih lengkap dan promosi lebih terarah."},
    ]
    for it in items:
        it["skor"] = round((it["urgensi"] + it["manfaat"] + it["kemudahan"] + (6 - it["biaya"])) / 4, 2)
    items.sort(key=lambda x: -x["skor"])
    for i, it in enumerate(items, 1):
        it["rank"] = i

    # D. estimasi
    est = [
        {"nama": "Bundling & penempatan (3 aturan teratas)",
         "rumus": "Σ [transaksi berisi antecedent × (1 − confidence) × tingkat adopsi × rata-rata nilai produk consequent]",
         "pendapatan": b_rev, "biaya": b_cost, "laba": b_profit},
        {"nama": f"Reaktivasi {ch['Segmen']}", "rumus": "pendapatan = pelanggan × tingkat reaktivasi × rata-rata belanja; biaya = nilai voucher × pelanggan yang kembali",
         "pendapatan": r_rev, "biaya": r_cost, "laba": r_profit},
        {"nama": f"Loyalitas {lo['Segmen']}", "rumus": "total belanja segmen × kenaikan belanja",
         "pendapatan": l_rev, "biaya": l_cost, "laba": l_profit},
    ]
    tot_rev = sum(e["pendapatan"] for e in est)
    tot_profit = sum(e["laba"] for e in est)
    # F. KPI
    cust_tx = tx[tx["Customer_ID"] != NO_CUST].groupby("Customer_ID").size()
    kpis = [
        {"kpi": "Rata-rata basket_size", "baseline": f"{num(tx['basket_size'].mean(),2)} produk", "cara": "Rata-rata produk unik per transaksi, dibandingkan sebelum dan sesudah bundling"},
        {"kpi": "Penjualan produk pasangan bundling", "baseline": ", ".join(f"{p['b']}" for p in placement[:3]) or "-", "cara": "Penjualan produk consequent pada toko uji vs toko kontrol"},
        {"kpi": f"Tingkat reaktivasi {ch['Segmen']}", "baseline": "0% (belum ada program)", "cara": "% pelanggan segmen yang bertransaksi dalam 30 hari setelah voucher"},
        {"kpi": "Frekuensi belanja rata-rata", "baseline": f"{num(cust_tx.mean(),2)} transaksi/pelanggan/tahun", "cara": "Transaksi per pelanggan per bulan (kelompok program vs kelompok kontrol)"},
        {"kpi": "Pangsa transaksi Member", "baseline": f"{num((df.drop_duplicates('Transaction_ID')['Customer_Type'] == 'Member').mean() * 100,1)}%", "cara": "Persentase transaksi dengan Customer_Type = Member per bulan"},
        {"kpi": "Rata-rata nilai transaksi (AOV)", "baseline": rp(total / n_tx), "cara": "Total penjualan ÷ jumlah transaksi per bulan"},
    ]
    plan = {
        "jangka_pendek": [
            "Minggu 1-2: ubah penataan rak untuk pasangan produk teratas dan pasang label paket bundling.",
            f"Minggu 2-4: kirim voucher reaktivasi ke {int(ch['Pelanggan'])} pelanggan {ch['Segmen']} (uji pada sebagian dulu).",
            "Bulan 1: catat penjualan harian produk bundling dan jumlah voucher yang ditukar sebagai data awal evaluasi."],
        "jangka_menengah": [
            "Bulan 2-3: luncurkan program poin/tier untuk segmen loyal dan kampanye 'jadi Member' di kasir.",
            "Bulan 3-6: jalankan ulang Apriori dan K-Means setiap kuartal dengan data baru; perbarui bundling dan segmen.",
            "Bulan 6: evaluasi dengan uji A/B (toko/kelompok pelanggan kontrol) lalu putuskan perluasan."],
    }
    return {
        "fakta": {"total_penjualan": total, "transaksi": n_tx, "pelanggan": int(rfm_k["Customer_ID"].nunique())},
        "placement": placement, "retention": retention, "prioritas": items,
        "estimasi": est, "estimasi_total": {"pendapatan": tot_rev, "laba": tot_profit,
                                            "persen_dari_total": tot_rev / total * 100 if total else 0},
        "kpi": kpis, "rencana": plan, "asumsi": assump,
    }


A = sys.modules[__name__]  # akses fungsi analisis dengan awalan A.


# =============================================================== BAGIAN 2: SERVER FLASK

# ----------------------------------------------------------------- konfigurasi akun & profil
LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "yohana").strip().lower()
LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "51230082")
PROFIL = {"nama": "Yohana Antanik Kuabib", "npm": "51230082", "matkul": "Business Intelligence",
          "kampus": "Universitas Timor", "prodi": "Teknologi Informasi"}

DEFAULT_AP = {"min_support": 0.01, "min_conf": 0.20, "max_len": 3, "top_n": 10, "rank_by": "lift"}
DEFAULT_ASSUMP = {"adopt": 0.05, "bundle_disc": 0.05, "react": 0.10, "voucher": 10000, "uplift": 0.05, "margin": 0.20}


class DatasetNotUploadedError(ValueError):
    """Raised when an analysis endpoint is called before a dataset is uploaded."""


app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("SECRET_KEY", "uts-bi-yohana-unimor-2026")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
STATE = {}  # state per sesi (disimpan di memori server)


def st():
    sid = session.get("sid")
    if not sid:
        sid = session["sid"] = uuid.uuid4().hex
    return STATE.setdefault(sid, {})


def J(obj):
    return jsonify(A.clean_json(obj))


@app.errorhandler(DatasetNotUploadedError)
def on_dataset_not_uploaded(e):
    return jsonify({"error": str(e)}), 400


@app.before_request
def guard():
    p = request.path
    if p.startswith("/api/") and p not in ("/api/login", "/api/me") and not session.get("user"):
        return jsonify({"error": "Belum login"}), 401


@app.errorhandler(Exception)
def on_error(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------- halaman & autentikasi
@app.route("/")
def index():
    return Response(asset("index.html"), mimetype="text/html")


@app.route("/static/<path:p>")
def static_files(p):
    mt = {"js": "application/javascript", "css": "text/css", "png": "image/png", "jpg": "image/jpeg"}.get(p.rsplit(".", 1)[-1], "application/octet-stream")
    try:
        return Response(asset(p), mimetype=mt)
    except KeyError:
        return jsonify({"error": "tidak ditemukan"}), 404


@app.route("/api/login", methods=["POST"])
def login():
    d = request.get_json(force=True, silent=True) or {}
    u = str(d.get("username", "")).strip().lower()
    p = str(d.get("password", "")).strip()
    if u == LOGIN_USERNAME and p == LOGIN_PASSWORD:
        old_sid = session.get("sid")
        if old_sid:
            STATE.pop(old_sid, None)
        session.clear()
        session["user"] = LOGIN_USERNAME
        session["sid"] = uuid.uuid4().hex
        STATE[session["sid"]] = {}
        return J({"ok": True, "profil": PROFIL})
    return jsonify({"ok": False, "error": "Username atau password salah."}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    STATE.pop(session.get("sid"), None)
    session.clear()
    return J({"ok": True})


@app.route("/api/me")
def me():
    if session.get("user"):
        s = st()
        return J({"login": True, "profil": PROFIL, "uploaded": "raw" in s, "filename": s.get("filename")})
    return J({"login": False})


# ----------------------------------------------------------------- helper tahapan
def need_raw():
    s = st()
    if "raw" not in s:
        raise DatasetNotUploadedError("Unggah file dataset terlebih dahulu (langkah 1).")
    return s


def ensure_clean():
    s = need_raw()
    if "pre" not in s:
        s["pre"] = A.preprocess(s["raw"], s["produk"], s["pelanggan"])
    return s


def ensure_feat():
    s = ensure_clean()
    if "feat" not in s:
        s["feat"] = A.build_features(s["pre"]["clean"])
    return s


def ensure_ap(params=None):
    s = ensure_feat()
    if params is not None or "ap" not in s:
        p = dict(DEFAULT_AP)
        p.update(params or {})
        s["ap_params"] = p
        s["ap"] = A.run_apriori(s["feat"]["onehot"], p["min_support"], p["min_conf"], p["max_len"], p["top_n"], p["rank_by"])
    return s


def ensure_km(k=None):
    s = ensure_feat()
    if k is not None or "km" not in s:
        s["km"] = A.run_kmeans(s["feat"], k)
    return s


# ----------------------------------------------------------------- 1. upload
@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "File tidak ditemukan."}), 400
    data = io.BytesIO(f.read())
    w = A.load_workbook(data, f.filename)
    sid = session.get("sid") or uuid.uuid4().hex
    session["sid"] = sid
    STATE[sid] = {"raw": w["raw"], "produk": w["produk"], "pelanggan": w["pelanggan"], "filename": f.filename}
    raw = w["raw"]
    payload = {
        "filename": f.filename, "sheets": w["sheets"], "baris": len(raw), "kolom": raw.shape[1],
        "kolom_nama": list(raw.columns), "preview": A.df_records(raw, n=10),
        "produk_ada": w["produk"] is not None, "pelanggan_ada": w["pelanggan"] is not None,
    }
    STATE[sid]["upinfo"] = A.clean_json(payload)
    return J(payload)


@app.route("/api/upload_info")
def upload_info():
    s = need_raw()
    return jsonify(s["upinfo"])


# ----------------------------------------------------------------- 2. business understanding
@app.route("/api/business")
def business():
    s = need_raw()
    raw = s["raw"]
    ctx = {"baris": len(raw), "transaksi": int(raw["Transaction_ID"].nunique()),
           "pelanggan": int(raw["Customer_ID"].nunique()), "produk": int(raw["Product_ID"].nunique())}
    return J({
        "ctx": ctx,
        "masalah": [
            {"judul": "Data hanya dipakai untuk laporan rutin", "isi": "Manajemen belum memanfaatkan data transaksi untuk memahami pola belanja, sehingga keputusan promosi dan stok masih berbasis perkiraan."},
            {"judul": "Pola kombinasi produk belum diketahui", "isi": "Tidak ada informasi produk apa yang sering dibeli bersamaan, padahal informasi ini dasar bundling, cross-selling, dan penataan rak."},
            {"judul": "Pelanggan diperlakukan seragam", "isi": "Tanpa segmentasi, pelanggan loyal, pelanggan baru, dan pelanggan yang hampir berhenti mendapat perlakuan pemasaran yang sama."},
            {"judul": "Kualitas data belum terjamin", "isi": "Data berpotensi memuat nilai kosong, duplikat, nilai tidak wajar, dan format tidak konsisten yang bisa menyesatkan analisis jika tidak dibersihkan."},
        ],
        "pertanyaan": [
            {"q": "Produk apa saja yang sering dibeli bersamaan dalam satu transaksi?", "jawab": "Tahap Apriori (Market Basket Analysis)"},
            {"q": "Segmen pelanggan mana yang paling bernilai dan segmen mana yang berisiko berhenti belanja?", "jawab": "Tahap RFM dan K-Means"},
            {"q": "Kapan penjualan meningkat atau menurun, dan produk/kategori apa yang paling menyumbang pendapatan?", "jawab": "Tahap EDA dan Dashboard BI"},
            {"q": "Berapa besar kualitas data yang bermasalah dan bagaimana dampaknya pada angka penjualan?", "jawab": "Tahap EDA dan Preprocessing (perbandingan sebelum-sesudah)"},
            {"q": "Tindakan pemasaran apa yang paling layak diprioritaskan dan bagaimana mengukur keberhasilannya?", "jawab": "Tahap Rekomendasi Keputusan"},
        ],
        "manfaat": [
            {"judul": "Meningkatkan penjualan", "isi": "Aturan asosiasi (lift > 1) dipakai untuk paket bundling, penempatan rak berdekatan, dan rekomendasi produk di kasir sehingga nilai keranjang naik."},
            {"judul": "Mempertahankan pelanggan", "isi": "Segmentasi RFM membedakan pelanggan loyal dan berisiko churn, sehingga voucher reaktivasi dan program loyalitas diarahkan tepat sasaran dengan biaya lebih efisien."},
            {"judul": "Keputusan berbasis bukti", "isi": "Dashboard menyajikan KPI dan tren yang dapat difilter, sehingga manajemen memantau hasil tindakan dan mengevaluasi dampaknya."},
        ],
    })


# ----------------------------------------------------------------- 3. EDA
@app.route("/api/eda")
def api_eda():
    s = need_raw()
    if "eda" not in s:
        cat, _ = A.build_catalog(s["raw"], s["produk"])
        s["eda"] = A.eda(s["raw"], cat)
    return J(s["eda"])


# ----------------------------------------------------------------- 4. preprocessing
@app.route("/api/preprocess")
def api_preprocess():
    s = ensure_clean()
    p = s["pre"]
    clean = p["clean"]
    return J({"log": p["log"], "before": p["before"], "after": p["after"], "doc": p["doc"],
              "preview": A.df_records(clean, ["Transaction_ID", "Tanggal", "Customer_ID", "Region", "Customer_Type", "Product_ID",
                                                "Product_Name", "Category", "Quantity", "Unit_Price", "Discount_Rate", "Line_Total",
                                                "Payment_Method", "Year_Month"], n=10),
              "catalog": p["catalog"]})


# ----------------------------------------------------------------- 5. feature engineering
@app.route("/api/features")
def api_features():
    s = ensure_feat()
    f, p = s["feat"], s["pre"]
    tx, rfm, onehot, X = f["tx"], f["rfm"], f["onehot"], f["X"]
    bs = tx["basket_size"].value_counts().sort_index()
    top_items = list(onehot.mean().sort_values(ascending=False).head(10).index)
    sample_oh = onehot[top_items].head(8).astype(int).reset_index()
    raw_cols = ["Recency", "Frequency", "Monetary"]
    stat = lambda d: [{"fitur": c, "min": d[c].min(), "mean": d[c].mean(), "std": d[c].std(ddof=0), "max": d[c].max()} for c in raw_cols]
    scaled = pd.DataFrame(X, columns=raw_cols)
    cust_txt = int((p["clean"]["Customer_ID"] == A.NO_CUST).sum())
    return J({
        "ref_date": f["ref_date"].strftime("%d %B %Y"),
        "basket": {"mean": tx["basket_size"].mean(), "median": tx["basket_size"].median(), "max": int(tx["basket_size"].max()),
                   "dist": [{"basket": int(k), "transaksi": int(v)} for k, v in bs.items()],
                   "sample": A.df_records(tx, ["Transaction_ID", "Customer_ID", "jumlah_baris", "basket_size", "total_qty", "total_belanja"], n=8)},
        "rfm": {"n": len(rfm), "stat": stat(rfm), "sample": A.df_records(rfm.sort_values("Monetary", ascending=False), n=8),
                "tidak_teridentifikasi": cust_txt},
        "apriori_matrix": {"shape": [int(onehot.shape[0]), int(onehot.shape[1])], "density": float(onehot.values.mean() * 100),
                           "sample": A.df_records(sample_oh), "kolom": ["Transaction_ID"] + top_items},
        "kmeans_matrix": {"shape": [int(X.shape[0]), int(X.shape[1])], "fitur": raw_cols,
                          "before": stat(rfm), "after": [{"fitur": c, "min": scaled[c].min(), "mean": scaled[c].mean(), "std": scaled[c].std(ddof=0), "max": scaled[c].max()} for c in raw_cols],
                          "sample_scaled": A.df_records(pd.concat([rfm[["Customer_ID"]].reset_index(drop=True), scaled.round(3)], axis=1), n=8)},
        "funnel": [{"tahap": "Baris item transaksi (data mentah)", "n": int(len(s["raw"]))},
                   {"tahap": "Baris data bersih", "n": int(len(p["clean"]))},
                   {"tahap": "Transaksi (basket) untuk Apriori", "n": int(len(tx))},
                   {"tahap": "Pelanggan (RFM) untuk K-Means", "n": int(len(rfm))}],
        "doc": p["doc"],
    })


# ----------------------------------------------------------------- 6. Apriori
def ap_payload(s):
    ap = s["ap"]
    return {"params": s["ap_params"], "n_transaksi": ap["n_transaksi"], "n_produk": ap["n_produk"], "n_frequent": ap["n_frequent"],
            "n_frequent_by_size": ap["n_frequent_by_size"], "n_rules": ap["n_rules"], "n_rules_lift_gt1": ap["n_rules_lift_gt1"],
            "frequent_itemsets": ap["frequent_itemsets"], "rules": ap["rules_top"], "heat_items": ap["heat_items"],
            "heat_lift": ap["heat_lift"], "support_stat": ap["support_stat"]}


@app.route("/api/apriori", methods=["GET", "POST"])
def api_apriori():
    body = request.get_json(silent=True) or {}
    params = None
    if request.method == "POST":
        params = {}
        for k, caster in (("min_support", float), ("min_conf", float), ("max_len", int), ("top_n", int)):
            if k in body:
                params[k] = caster(body[k])
        params["top_n"] = max(1, min(10, params.get("top_n", 10)))
        if body.get("rank_by") in ("lift", "confidence", "support"):
            params["rank_by"] = body["rank_by"]
        if not (0 < params.get("min_support", .01) <= 1) or not (0 < params.get("min_conf", .2) <= 1):
            return jsonify({"error": "Support dan confidence harus di antara 0 dan 1."}), 400
    s = ensure_ap(params)
    return J(ap_payload(s))


# ----------------------------------------------------------------- 7. K-Means
@app.route("/api/kmeans", methods=["GET", "POST"])
def api_kmeans():
    body = request.get_json(silent=True) or {}
    k = body.get("k") if request.method == "POST" else None
    s = ensure_km(int(k) if k else None)
    km = s["km"]
    rk = km["rfm_k"]
    sample = rk.sample(min(len(rk), 1500), random_state=1) if len(rk) > 1500 else rk
    pts = A.df_records(sample[["Customer_ID", "Recency", "Frequency", "Monetary", "Segmen", "Cluster"]])
    return J({
        "eval": km["eval"], "k": km["k"], "silhouette_k": km["silhouette_k"],
        "profile": km["profile"], "prioritas": km["prioritas"], "points": pts, "n": len(rk),
        "centroid": km["centroid"],
        "hasil": A.df_records(rk.sort_values(["Cluster", "Monetary"], ascending=[True, False]), n=15),
        "fitur": ["Recency (hari sejak transaksi terakhir)", "Frequency (jumlah transaksi unik)", "Monetary (total belanja bersih)"],
    })


# ----------------------------------------------------------------- 8. dashboard
@app.route("/api/dashboard/meta")
def dash_meta():
    s = ensure_km()
    s = ensure_ap()
    df = s["pre"]["clean"]
    return J({"min": df["Tanggal"].min().strftime("%Y-%m-%d"), "max": df["Tanggal"].max().strftime("%Y-%m-%d"),
              "categories": sorted(df["Category"].unique()), "segments": sorted(s["km"]["rfm_k"]["Segmen"].unique()),
              "ap_params": s["ap_params"]})


@app.route("/api/dashboard", methods=["POST"])
def dash():
    b = request.get_json(silent=True) or {}
    s = ensure_ap()
    s = ensure_km()
    out = A.dashboard(s["pre"]["clean"], s["km"]["rfm_k"], s["km"]["profile"], s["ap_params"],
                      b.get("start") or None, b.get("end") or None, b.get("categories") or None, b.get("segments") or None)
    if not out.get("empty"):
        rk = s["km"]["rfm_k"]
        if b.get("segments"):
            rk = rk[rk["Segmen"].isin(b["segments"])]
        ids = set(A.apply_filters(s["pre"]["clean"], s["km"]["rfm_k"], b.get("start"), b.get("end"), b.get("categories"), b.get("segments"))["Customer_ID"])
        rk = rk[rk["Customer_ID"].isin(ids)]
        out["points"] = A.df_records(rk[["Recency", "Frequency", "Monetary", "Segmen"]])
    return J(out)


# ----------------------------------------------------------------- 9. rekomendasi
@app.route("/api/recommendations", methods=["GET", "POST"])
def recs():
    s = ensure_ap()
    s = ensure_km()
    assump = dict(DEFAULT_ASSUMP)
    if request.method == "POST":
        b = request.get_json(silent=True) or {}
        for k in assump:
            if k in b:
                assump[k] = float(b[k])
    r = A.recommendations(s["pre"]["clean"], s["feat"], s["ap"], s["km"], assump)
    r["prioritas_segmen"] = s["km"]["prioritas"]
    return J(r)


# ----------------------------------------------------------------- 10. unduhan
def csv_resp(df, name):
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=name)


@app.route("/api/download/<name>")
def download(name):
    s = ensure_km()
    s = ensure_ap()
    p, f, km, ap = s["pre"], s["feat"], s["km"], s["ap"]
    clean = p["clean"].copy()
    clean["Tanggal"] = clean["Tanggal"].dt.strftime("%Y-%m-%d")
    clean["Transaction_Date"] = clean["Transaction_Date"].dt.strftime("%Y-%m-%d %H:%M")
    if name == "data_clean.csv":
        return csv_resp(clean, name)
    if name == "rfm_cluster.csv":
        return csv_resp(km["rfm_k"], name)
    if name == "basket.csv":
        return csv_resp(f["tx"], name)
    if name == "apriori_rules.csv":
        return csv_resp(ap["rules_all"].drop(columns=["ant_items", "cons_items"], errors="ignore"), name)
    if name == "hasil_analisis.xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            clean.to_excel(xw, sheet_name="Data_Clean", index=False)
            pd.DataFrame(p["log"]).to_excel(xw, sheet_name="Dokumentasi_Preprocessing", index=False)
            pd.DataFrame([p["before"], p["after"]]).to_excel(xw, sheet_name="Sebelum_vs_Sesudah", index=False)
            f["tx"].to_excel(xw, sheet_name="Transaksi_Basket", index=False)
            km["rfm_k"].to_excel(xw, sheet_name="RFM_KMeans", index=False)
            km["profile"].to_excel(xw, sheet_name="Profil_Segmen", index=False)
            ap["rules_all"].drop(columns=["ant_items", "cons_items"], errors="ignore").to_excel(xw, sheet_name="Apriori_Rules", index=False)
            pd.DataFrame(ap["frequent_itemsets"]).to_excel(xw, sheet_name="Frequent_Itemsets", index=False)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=name,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return jsonify({"error": "File tidak dikenal"}), 404




# =============================================================== ASET FRONTEND (HTML/CSS/JS/gambar, terkompresi)
_ASSETS_B64 = {
    "index.html": (
        "eNqVV9ty2zYQffdXIHzIpBNTtJ0mzTgUWzuXjsZW4qnkhzyuSIiEBQIMAFrxTB/6Ef2XvvdT+iVdgHeJvsQztgHsBYvF2bNg+OzDl/fLr1cfSWZyHh2E9h/h"
        "INKpxxLPLlBI8F9ODZA4A6WpmXqlWftvvWZZQE6n3i2j20Iq45FYCkMFqm1ZYrJpQm9ZTH03OSRMMMOA+zoGTqfH1olhhtPofEaWciPJf3/9TT4Aup2jqkjJ"
        "c3Jeaiao1mSGfjlnKRUxJX+S68+z+Zc/wqCyPwg5ExuiKMfQMQSPZIqup16gDRgWByxPAy5TOSlE6g20C0VRX9DYNDaZMYU+DYI1nkRPUilTTqFgehLL/Adt"
        "q92dIYmV1FoqljLROHl8vyDW+uTXNeSM300XUsHpNs3Mb6+Pjt69wd9fjo6et8JSxfTlAoR++arS+hk1+poJ0wWHu6neQuFVJ9DmjlOdUWqGJ+ut72QSAwqc"
        "dIIjaxTUIFnJ5C46OAif+T6Zjv2Qyy+/zz7fI/N9dKExk0wKwpKph7fF8BpjDlrXM5KxJKECNyUkTNjtQOiDMk4yJlspEEktRTmigWgV3wMPAhzhe4lTci3Y"
        "LVWaGdBkyXKphgH51qZzi9s242EQbnu/FMyL9jyGwSNmulx50ZWSqYKcLEyZMLKkGyFtBFgUa6ly0GzgpjfpD3uuba0qYMbfKsRC/wR7GorZkhn6H03gWho5"
        "uSmaBH7CKfkqMxBAzoQBwTbkooQVW3m7e3gPxlqlepvJ0Thx3bcc5EWjew3j3rGzBOZFn6/m5PXxyaujo7cn5N9/xhlnLKX1YBSN9lp8zRLanM0udMD+hLMd"
        "LMWgEkxdaSTWfcGpQVqV63V36Ow4WjBtaI5HBM400+FKRUsEiYaNZo5AsRqPW4NiB6xYp17k2DWv2DUBMX7aUxQpRkqRppDh2MAhKWi+ssjFJONEciArir3i"
        "BsiLs0IxZLafDommaY7sj4BEfdtJUtzixYU/pxillUNeAEOO2UjUS6zehhalKTWISRgULpsueA7onmDWsOFoqqorvq5HYeDEHR4F+nDZbXWJuSswg4Z+RwpD"
        "2otpJnlC0d0cdLnZYFydrqLfSqZo4tK/lnGpRwMpMJtbidcUXdWj3UD6BbQdFtcgzNZTHWY3Hw+1kzeh9tyuSmOQNStP1cSrdtkuZYrdxOvFZOoVvF/wXfBT"
        "b4m3wvhgn+gSL9qEQeVuh1j6p20h/VGpHj86/FO71LL2sBQHQSPH5cy0xisjCP76CKsc1J0br7iMN17kMoKYIdqVwl6AI4UokV+ul4t7XhJY8AuaU3SmyJtB"
        "ldsj9Mo8DOr29HCTO7u6nF2cLWYP9bkmb1AU7ZlxPOhvYMnDadnBCrrUNvN9prQSP3Nl/iOtrnpKDfw/tbM5XfcE85o33AMNzWm7fjZ44yH63pHz2SMtrBoL"
        "qFKH91XoLuRqNsR0BoVFNBUpVTcADoFo3njql6qSa8ZpTcGucqqV93bhx7teZd1VXewXGa6P99nYdxw1njGUPrW57Rnia4667nbatrenGMzt1VyUnEF2+mgv"
        "HH9rhIFD715rzIGJBrYWplhytcTIooN0xw72KrBPlOemewraub+yC/3rnuMqxo5/H6IE3KZB6zAJDaQuWLyhXaltqmm0m7fspLVYVu6w8Z6MpGR/e3AcovsB"
        "aItT689BLmMdKVSoxJUeiVrtffKvaViWpp+rhkjTTGqE2AXlJah7SL0aVtfS1Ii9MOe6/qprHbdzgzgUCf0+9fxjG521GJBmVbttko0E3blxs90Ggd8BsWKF"
        "GdbajQ7sF6iZlHkyudEuE04tulcfKfXpmie7qkH1QYMpcR/H/wMGH9CC"
    ),
    "css/style.css": (
        "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"
    ),
    "js/chart.umd.js": (
        "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"
    ),
    "js/app.js": (
        "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"
    ),
    "js/app2.js": (
        "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"
    ),
    "img/foto.jpg": (
        "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"
        "kpg="
    ),
    "img/logo.png": (
        "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"
        "AAAAAElFTkSuQmCCnaQdWg=="
    ),
}
_ASSET_CACHE = {}


def asset(name):
    """Ambil aset frontend. Nama diterima dengan atau tanpa awalan folder (js/, css/, img/)."""
    if name not in _ASSET_CACHE:
        content = zlib.decompress(base64.b64decode(_ASSETS_B64[name]))
        if name == "index.html":
            content = content.replace(b"Semester 6", b"Semester 7")
        elif name in ("js/app.js", "js/app2.js"):
            content = re.sub(rb"ref\s*:\s*['\"]Soal\b[^'\"]*['\"]\s*,?", b"", content)
            if name == "js/app.js":
                content = re.sub(
                    rb"(kicker\s*:\s*['\"][^'\"]*?)\s*(?:\xc2\xb7|-)\s*Soal\b[^'\"]*",
                    rb"\1",
                    content,
                )
        _ASSET_CACHE[name] = content
    return _ASSET_CACHE[name]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5082))
    print(f"\n  Buka di browser: http://127.0.0.1:{port}\n  Login -> username: Yohana | password: 51230082\n")
    app.run(host="127.0.0.1", port=port, debug=False)