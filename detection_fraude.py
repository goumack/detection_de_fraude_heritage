import streamlit as st
import requests
import pickle
import numpy as np
from pathlib import Path

# Configuration du modèle
deployed_model_name = "fraud"
infer_endpoint = "https://fraud-serve-lab-ai-models.apps.origins.heritage.africa"
infer_url = f"{infer_endpoint}/v2/models/{deployed_model_name}/infer"

# Chargement du scaler
scaler_path = "scaler.pkl"
with open(scaler_path, 'rb') as handle:
    scaler = pickle.load(handle)

# Fonction d'inférence via REST API
def rest_request(data):
    json_data = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": [1, 5],
                "datatype": "FP32",
                "data": data
            }
        ]
    }
    
    response = requests.post(infer_url, json=json_data, verify=False)
    response_dict = response.json()
    return response_dict['outputs'][0]['data']

# 🎨 Mise en page professionnelle
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# 🌟 Titre principal
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #004080;">💳 Example de Détection de Fraude Bancaire par Accel-Tech sur origins</h1>
        <p style="color: #555;">Ceci est un exemlpe d'Analyse des transactions pour une sécurité optimale.</p>
        <hr>
    </div>
    """, unsafe_allow_html=True
)

# 📋 Saisie des caractéristiques
st.subheader("📌 Veuillez entrer les détails de la transaction :")

col1, col2 = st.columns(2)

with col1:
    distance_from_last_transaction = st.number_input("📏 Distance avec la dernière transaction (en km)", min_value=0.0, step=0.01)
    ratio_to_median_price = st.number_input("💰 Ratio par rapport au prix médian", min_value=0.0, step=0.01)
    used_chip = st.radio("💳 La puce a-t-elle été utilisée ?", ["Oui", "Non"])

with col2:
    used_pin_number = st.radio("🔢 Le code PIN a-t-il été saisi ?", ["Oui", "Non"])
    online_order = st.radio("🛒 Est-ce un achat en ligne ?", ["Oui", "Non"])

# Convertir les valeurs en 0/1 pour le modèle
used_chip = 1 if used_chip == "Oui" else 0
used_pin_number = 1 if used_pin_number == "Oui" else 0
online_order = 1 if online_order == "Oui" else 0

# 🔎 Bouton de prédiction stylisé
if st.button("🔍 Vérifier la transaction", use_container_width=True):
    data = [distance_from_last_transaction, ratio_to_median_price, used_chip, used_pin_number, online_order]
    scaled_data = scaler.transform([data]).tolist()[0]
    prediction = rest_request(scaled_data)
    
    threshold = 0.95
    result = "Fraude" if prediction[0] > threshold else "Aucune fraude"

    # Affichage du résultat avec des icônes et des couleurs adaptées
    if result == "Fraude":
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; background-color: #ffcccc; border-radius: 10px;">
                <h2 style="color: red;">⚠️ ALERTE FRAUDE ⚠️</h2>
                <p style="color: black;">Cette transaction semble suspecte. Vérification requise.</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; background-color: #ccffcc; border-radius: 10px;">
                <h2 style="color: green;">✅ Transaction Sécurisée</h2>
                <p style="color: black;">Aucune anomalie détectée sur cette transaction.</p>
            </div>
            """, unsafe_allow_html=True
        )
