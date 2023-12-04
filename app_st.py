# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from joblib import load
import pandas as pd
import numpy as np
import streamlit as st
import joblib


def user_input_cancer():
    columns = st.columns(2)  # Diviser la page en 2 colonnes

    # Curseurs pour la première colonne
    with columns[0]:
        perimeter_mean = st.slider('Périmètre moyen', 43.79, 188.5, 91.62)
        area_mean = st.slider('Surface moyenne', 143.5, 2501.0, 654.89)
        concavity_mean = st.slider(
            'Concavité moyenne', 0.0, 0.4268, 0.0888)
        concave_points_mean = st.slider(
            'Points concaves moyens', 0.0, 0.2012, 0.0489)
        area_se = st.slider('Surface (erreur)', 6.802, 542.2, 40.47)

    # Curseurs pour la deuxième colonne
    with columns[1]:
        radius_worst = st.slider('Rayon (pire)', 7.93, 36.04, 16.27)
        texture_worst = st.slider('Texture (pire)', 12.02, 49.54, 25.68)
        perimeter_worst = st.slider(
            'Périmètre (pire)', 50.41, 251.2, 107.2)
        area_worst = st.slider('Surface (pire)', 185.2, 4254.0, 880.6)
        concavity_worst = st.slider('Concavité (pire)', 0.0, 1.252, 0.207)
        concave_points_worst = st.slider(
            'Points concaves (pire)', 0.0, 0.291, 0.116)

    data_cancer = {
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'concavity_mean': concavity_mean,
        'concave_points_mean': concave_points_mean,
        'area_se': area_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'concavity_worst': concavity_worst,
        'concave_points_worst': concave_points_worst,
    }

    cancer_parameters = pd.DataFrame(data_cancer, index=[0])
    return cancer_parameters


df = user_input_cancer()

# Ajout d'un bouton pour déclencher la prédiction
if st.button('Obtenir la prédiction'):
    # Charger le modèle sauvegardé
    SVM = joblib.load("model_svc")

# Prédiction avec le modèle SVM
    prediction = SVM.predict(df)
    prediction_label = 'Maligne' if prediction[0] == 1 else 'Bénigne'

    st.subheader("La catégorie du diagnostic de la tumeur est :")
    st.write(prediction_label)
