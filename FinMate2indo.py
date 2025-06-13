import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import defaultdict

def apply_budget_rule(prediction_series, income):
    # Kelompok kategori
    group_map = {
        'food': 'needs',
        'transportation': 'needs',
        'books_supplies': 'needs',
        'technology': 'needs',
        'entertainment': 'wants',
        'personal_care': 'wants',
        'health_wellness': 'savings',
        'miscellaneous': 'savings'
    }

    group_ratios = {
        'needs': 0.5,
        'wants': 0.3,
        'savings': 0.2
    }

    # Kelompokkan prediksi ke dalam needs/wants/savings
    grouped_values = defaultdict(dict)
    for cat, val in prediction_series.items():
        group = group_map.get(cat)
        if group:
            grouped_values[group][cat] = val

    # Hitung alokasi akhir berdasarkan proporsi dalam grup
    final_allocation = {}
    for group, items in grouped_values.items():
        group_total = sum(items.values())
        if group_total > 0:
            for cat, val in items.items():
                prop = val / group_total
                final_allocation[cat] = round(prop * income * group_ratios[group], 2)
        else:
            for cat in items:
                final_allocation[cat] = 0.0

    return final_allocation


# ======== Load Model dan Scaler =========
model = load_model("finmate_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Daftar semua kolom fitur yang sesuai dengan training
feature_columns = [
    'monthly_income', 'financial_aid',
    'age',
    'gender_Female', 'gender_Male', 'gender_Non-binary',
    'year_in_school_Freshman', 'year_in_school_Junior', 'year_in_school_Senior', 'year_in_school_Sophomore',
    'major_Biology', 'major_Computer Science', 'major_Economics', 'major_Engineering', 'major_Psychology',
    'preferred_payment_method_Cash', 'preferred_payment_method_Credit/Debit Card', 'preferred_payment_method_Mobile Payment App'
]


# Target kolom (pengeluaran per kategori)
target_cols = [
    'food', 'transportation', 'books_supplies', 'technology',
    'entertainment', 'personal_care', 'health_wellness', 'miscellaneous'
]

# ======== UI Streamlit =========
st.title("💸 FinMate - Rekomendasi Pengeluaran Mahasiswa")

income = st.number_input("Pemasukan bulanan (Rp)", value=3000000)
financial_aid = st.selectbox("Mendapat bantuan keuangan?", ["Ya", "Tidak"])
gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
year = st.selectbox("Tingkat Pendidikan", ["Tahun Keempat", "Tahun Ketiga", "Tahun Kedua", "Tahun Pertama"])
major = st.selectbox("Jurusan", ["Teknik", "Bisnis", "Ilmu Sosial", "Psikologi", "Ilmu Komputer", "Biologi", "Ekonomi", "Matematika", "Sistem Informasi", "Lainnya"])
payment = st.selectbox("Metode Pembayaran Favorit", ["Tunai", "E-Wallet", "Kartu Kredit"])

if st.button("Rekomendasi Pengeluaran"):
    # Buat dictionary input user
    user_input = {
        'monthly_income': income,
        'financial_aid': 1 if financial_aid == "Ya" else 0,
        'gender_Male': 1 if gender == "Pria" else 0,
        'year_in_school_Senior': 1 if year == "Tahun Keempat" else 0,
        'major_Engineering': 1 if major == "Teknik" else 0,
        'preferred_payment_method_Cash': 1 if payment == "Tunai" else 0,
    }

    # Tambahkan kolom lain yang mungkin kosong dengan 0
    for col in feature_columns:
        if col not in user_input:
            user_input[col] = 0

    # Konversi ke DataFrame dan normalisasi
    df_input = pd.DataFrame([user_input])
    df_input[scaler.feature_names_in_] = scaler.transform(df_input[scaler.feature_names_in_])

    # Pastikan urutan kolom sesuai training
    df_input = df_input[feature_columns]

    # Prediksi
    prediction = model.predict(df_input)[0]
    prediction = pd.Series(prediction[:len(target_cols)], index=target_cols)
    final_budget = apply_budget_rule(prediction, user_input['monthly_income'])
    total_pred = prediction.sum()
    if total_pred > 0:
        scaling_factor = user_input['monthly_income'] / total_pred
        prediction = prediction * scaling_factor

    # Kamus terjemahan
    category_translation = {
    'food': 'Makanan',
    'transportation': 'Transportasi',
    'books_supplies': 'Buku & Perlengkapan',
    'technology': 'Teknologi',
    'entertainment': 'Hiburan',
    'personal_care': 'Perawatan Pribadi',
    'health_wellness': 'Kesehatan & Kebugaran',
    'miscellaneous': 'Lain-lain'
    }

    st.subheader("📊 Rekomendasi Pengeluaran :")
    for cat, val in final_budget.items():
        nama_indonesia = category_translation.get(cat, cat)
        st.write(f"- {nama_indonesia}: Rp {val:,.2f}")



