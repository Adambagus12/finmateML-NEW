from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import defaultdict

app = Flask(__name__)   
model = load_model("finmate_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

feature_columns = [
    'monthly_income', 'financial_aid',
    'age',
    'gender_Female', 'gender_Male', 'gender_Non-binary',
    'year_in_school_Freshman', 'year_in_school_Junior', 'year_in_school_Senior', 'year_in_school_Sophomore',
    'major_Biology', 'major_Computer Science', 'major_Economics', 'major_Engineering', 'major_Psychology',
    'preferred_payment_method_Cash', 'preferred_payment_method_Credit/Debit Card', 'preferred_payment_method_Mobile Payment App'
]

target_cols = [
    'food', 'transportation', 'books_supplies', 'technology',
    'entertainment', 'personal_care', 'health_wellness', 'miscellaneous'
]

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

def apply_budget_rule(prediction_series, income):
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

    group_ratios = {'needs': 0.5, 'wants': 0.3, 'savings': 0.2}

    grouped_values = defaultdict(dict)
    for cat, val in prediction_series.items():
        group = group_map.get(cat)
        if group:
            grouped_values[group][cat] = val

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

@app.route("/predict", methods=["POST"])
def predict_budget():
    data = request.json

    user_input = {col: 0 for col in feature_columns}
    user_input["monthly_income"] = data.get("monthly_income", 0)
    user_input["financial_aid"] = 1 if data.get("financial_aid") == "Ya" else 0
    user_input[f"gender_{data.get('gender')}"] = 1
    user_input[f"year_in_school_{data.get('year')}"] = 1
    user_input[f"major_{data.get('major')}"] = 1
    user_input[f"preferred_payment_method_{data.get('payment_method')}"] = 1

    df_input = pd.DataFrame([user_input])
    df_input[scaler.feature_names_in_] = scaler.transform(df_input[scaler.feature_names_in_])
    df_input = df_input[feature_columns]

    prediction = model.predict(df_input)[0]
    prediction = pd.Series(prediction[:len(target_cols)], index=target_cols)

    final_budget = apply_budget_rule(prediction, user_input['monthly_income'])

    result = {
        category_translation.get(k, k): round(v, 2)
        for k, v in final_budget.items()
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
