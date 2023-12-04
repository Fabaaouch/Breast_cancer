# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from joblib import dump, load
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, fbeta_score


def balance_dataframe(df, target_column, option):
    """
    Équilibre un DataFrame en utilisant différentes méthodes d'équilibrage.

    Args:
        df (DataFrame): Le DataFrame à équilibrer.
        target_column (str): Le nom de la colonne cible.
        option (str): L'option de l'équilibrage ('over' pour le sur-échantillonnage avec RandomOverSampler,
                      'under' pour le sous-échantillonnage avec RandomUnderSampler,
                      'smote_nc' pour le sur-échantillonnage avec SMOTE-NC).

    Returns:
        DataFrame: Le DataFrame équilibré.
    """
    if option == 'over':
        oversampler = RandomOverSampler(
            sampling_strategy='auto', random_state=42)
        X, y = oversampler.fit_resample(
            df.drop(columns=[target_column]), df[target_column])
    elif option == 'under':
        undersampler = RandomUnderSampler(
            sampling_strategy='auto', random_state=42)
        X, y = undersampler.fit_resample(
            df.drop(columns=[target_column]), df[target_column])
    elif option == 'smote_nc':
        # Assuming that categorical features are labeled as 'object' dtype
        cat_columns = df.select_dtypes(include=['object']).columns.tolist()
        smote_nc = SMOTENC(categorical_features=cat_columns, random_state=42)
        X, y = smote_nc.fit_resample(
            df.drop(columns=[target_column]), df[target_column])
    else:
        raise ValueError("L'option doit être 'over', 'under' ou 'smote_nc'.")

    if option != 'smote_nc':
        balanced_df = pd.concat([pd.DataFrame(X, columns=df.drop(
            columns=[target_column]).columns), pd.Series(y, name=target_column)], axis=1)
    else:
        balanced_df = pd.concat([pd.DataFrame(X, columns=df.drop(columns=[
                                target_column]).columns), pd.DataFrame(y, columns=[target_column])], axis=1)

    return balanced_df


df = pd.read_csv('data.csv')

st.sidebar.title('Sommaire')

pages = ["Contexte du projet", "Exploration des données",
         "Analyse de données", "Modélisation", "Application 1", "Application 2"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:

    st.write('<span style="color:red; font-size:20px">Contexte du projet</span>',
             unsafe_allow_html=True)

    st.write("- Client : département chirurgie de l’Université du Wisconsin")

    st.write("- Data : à partir images de cellules de tumeurs du sein")

    st.write(
        "- technique du snake : contour du noyau cellulaire pour définir ses caractéristiques")

    st.write("- Calcul de différents paramètres (forme noyau, densité,taille…)")

    st.image("breast_cancer_image.jpg")

    st.write('<span style="color:red; font-size:20px">Objectifs</span>',
             unsafe_allow_html=True)

    st.write("- Modèle de prédiction du risque de cancer du sein après 	biopsie et analyse des images (Machine Learning)")

    st.write(
        "- Application directement utilisable par les médecins (Déploiement local & cloud)")

elif page == pages[1]:
    st.write("### Exploration des données")

    st.dataframe(df.head())

    st.write("Dimensions du dataframe :")

    st.write(df.shape)
    st.write("Variables du dataframe:")

    st.write(
        "- Une variable cible:‘diagnosis’ : Target M = malignant = 1, B = benign= 0")
    st.write("- 30 variables descriptives des noyaux cellulaires : numériques")
    st.write("- ce set de données est deséquilibré: 63% de B et 37% de M")
    st.image("desequilibre.png", width=300)
    if st.checkbox("Afficher le nombre des variables numériques"):
        st.write("29 variables numériques")
    if st.checkbox("Afficher le nombre des variables catégorielles"):
        st.write("une seule variable catégorielle: 'diagnosis'=Target")

    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
    if st.checkbox("Afficher les doublons"):
        st.write(df.duplicated().sum())
elif page == pages[2]:
    st.write("### Analyse de données")

    df.diagnosis = df.diagnosis.replace(["M", "B"], [1, 0])

# Liste des variables à afficher
    list_var = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Titre de l'application Streamlit
    st.write('Distribution des variables par rapport à la target')

# Sélection de la variable à afficher via un sélecteur déroulant
    selected_var = st.selectbox('Sélectionnez une variable', list_var)

# Affichage du graphique en fonction de la variable sélectionnée
    plt.figure(figsize=(10, 8))
    sns.kdeplot(df.loc[df['diagnosis'] == 0, selected_var],
                label='diagnosis == B', color='blue')
    sns.kdeplot(df.loc[df['diagnosis'] == 1, selected_var],
                label='diagnosis == M', color='orange')
# Labeling du graphique
    plt.xlabel(selected_var)
    plt.ylabel('Density')
    plt.title('Distribution de ' + selected_var)
    plt.legend()  # Ajouter une légende au graphique

# Afficher le graphique dans l'application Streamlit
    st.pyplot(plt)


# Streamlit selectbox to choose variables
    selected_var = st.selectbox("Choose a variable", list_var)

# Create subplots for each variable
    fig, ax = plt.subplots()

# Filter data for 'M' and 'B' diagnoses
    data_M = df[df['diagnosis'] == 1][selected_var]
    data_B = df[df['diagnosis'] == 0][selected_var]

# Create bar plots for the selected variable based on 'diagnosis' values
    ax.bar(['M', 'B'], [data_M.mean(), data_B.mean()], color=['blue', 'orange'])
    ax.set_title(f'Average {selected_var} by diagnosis')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Mean Value')

# Show the Matplotlib plot in Streamlit
    st.pyplot(fig)


# Titre de l'application Streamlit
    st.write('Relation entre la variable cible et les autres variables')

# Sélection de la variable à afficher sur l'axe y via un sélecteur déroulant
    selected_var_y = st.selectbox(
        'Sélectionnez une variable pour l\'axe Y', list_var)

# Graphique interactif avec une variable sélectionnée sur l'axe Y et la variable cible sur l'axe X
    fig = px.scatter(df, x="diagnosis", y=selected_var_y,
                     title=f"Relation entre 'diagnosis' et '{selected_var_y}'")
    st.plotly_chart(fig)

    st.write("- Les tumeurs bénignes présentent des perimètres, des rayons et des surfaces plus réduits généralement")

    st.write(
        "- Valeurs plus élevées pour concavity et compactness dans le cas de tumeurs malignes")
    st.write(
        "- Les variables qui permettent une meilleure classification pour les catégories de la target: radius_mean,perimeter_mean, area_mean, concavity_mean, concave points_mean, perimeter_worst, area_worst, concavity_worst, concave points_worst")

  # Création d'une palette de couleurs rose avec des nuances
    palette = sns.diverging_palette(250, 10, as_cmap=True)

# Création d'une nouvelle figure pour la matrice de corrélation
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(df[list_var].corr(), cmap=palette, annot=True, vmin=-1, vmax=1)
    plt.title('Matrice de corrélation')

# Affichage de la matrice de corrélation dans l'application Streamlit
    st.pyplot(fig)

    st.write("- Correlation/target:radius/périmètre/aire (+ avec les worst)compactness/concavity/concave_points")

    st.write(
        "- les variables les plus corrélées entre elles:Area / radius / périmètre/Compactness/concavity/concave_points")

    st.write("- Peu corrélée: Texture et symmetry")

elif page == pages[3]:
    st.write("### Modélisation")
    st.subheader(
        "Métriques pour l'évaluation des modèles de classification du cancer du sein")
    st.write("Les métriques suivantes sont couramment utilisées pour évaluer les modèles de classification pour le cancer du sein :")
    st.write("- **Accuracy**: Mesure la précision globale du modèle en pourcentage de prédictions correctes.")
    st.write("- **Recall (Sensibilité)**: Mesure la capacité du modèle à détecter correctement les faux négatifs: diagnostic bénin pour un cas malin")
    st.write("- **AUC (Area Under the Curve)**: Mesure la capacité du modèle à distinguer entre les classes positives et négatives.")
    st.write("- **F-beta Score**: Mesure la précision du modèle avec un biais particulier pour le rappel, qui est important pour minimiser les faux négatifs dans le diagnostic du cancer du sein.")

    # Conseils pour les métriques
    st.write("Lorsqu'il s'agit de minimiser les faux négatifs (patients diagnostiqués sans cancer alors qu'ils en ont effectivement),")
    st.write(
        "on peut se fier au **Recall**, car il identifie correctement les faux négatifs.")
    st.write("Il est également important de contrôler l'**Accuracy** pour avoir une vue d'ensemble de la précision globale du modèle.")
    st.subheader("Eclatement en set d'entrainement, validation et test")
    df_bc = pd.read_csv('data.csv')
    df_bc.diagnosis = df_bc.diagnosis.replace(["M", "B"], [1, 0])
    liste_to_keep_rfe = ['perimeter_mean', 'area_mean', 'concavity_mean', 'concave_points_mean',
                         'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                         'area_worst', 'concavity_worst', 'concave_points_worst', 'diagnosis']
    dfe_rfe = df_bc[liste_to_keep_rfe]
    df_train, df_test_perf = train_test_split(
        dfe_rfe, test_size=0.20, random_state=100)

    df_bc_o = balance_dataframe(df_train, 'diagnosis', option='over')
    y6 = df_bc_o["diagnosis"]
    X6 = df_bc_o .drop("diagnosis", axis=1)

    X_train6, X_test6, y_train6, y_test6 = train_test_split(
        X6, y6, test_size=0.2, random_state=100)

    st.write("le shape du set d'entrainement", X_train6.shape)
    st.write("le shape du set de test:", X_test6.shape)
    st.write("Le shape du set de validation:", df_test_perf.shape)

    st.subheader("Modèles")

    scaler = StandardScaler()
    X_train6 = scaler.fit_transform(X_train6)
    X_test6 = scaler.transform(X_test6)

    RFC = joblib.load("model_rfc")
    lr = joblib.load("model_lr")
    SVM = joblib.load("model_svc")
    dtc = joblib.load("model_dtc")
    lgbm = joblib.load("model_lgbm")
    Xgb = joblib.load("model_Xgb")
    KNN = joblib.load("model_KNN")

    y_pred_rfc = RFC.predict(X_test6)
    y_pred_lr = lr.predict(X_test6)
    y_pred_knn = KNN.predict(X_test6)
    y_pred_svm = SVM.predict(X_test6)
    y_pred_dtc = dtc.predict(X_test6)
    y_pred_lgbm = lgbm.predict(X_test6)
    y_pred_xgb = Xgb.predict(X_test6)

    model_choisi = st.selectbox(label="Modèle", options=[
        'SVC', 'Random Forest', 'KNN', 'LightGBM', 'XGBoost', 'LogisticRegression', 'DecisionTree'])

    def train_model(model_choisi):
        if model_choisi == 'SVC':
            y_pred = y_pred_svm
        elif model_choisi == 'Random Forest':
            y_pred = y_pred_rfc
        elif model_choisi == 'KNN':
            y_pred = y_pred_knn
        elif model_choisi == 'LightGBM':
            y_pred = y_pred_lgbm
        elif model_choisi == 'XGBoost':
            y_pred = y_pred_xgb
        elif model_choisi == 'LogisticRegression':
            y_pred = y_pred_lr
        elif model_choisi == 'DecisionTree':
            y_pred = y_pred_dtc
        accuracy = round(accuracy_score(y_test6, y_pred), 2)
        recall = round(recall_score(y_test6, y_pred), 2)
        auc = round(roc_auc_score(y_test6, y_pred), 2)
        fbeta = round(fbeta_score(y_test6, y_pred, beta=8), 2)

        return accuracy, recall, auc, fbeta

    accuracy, recall, auc, fbeta = train_model(model_choisi)

    st.write("Accuracy:", accuracy)
    st.write("Recall:", recall)
    st.write("AUC:", auc)
    st.write("F-beta:", fbeta)

elif page == pages[4]:
    st.header("Les paramètres d'entrée")

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

elif page == pages[5]:
    st.header("Les paramètres d'entrée")

    def user_input_cancer():
        columns = st.columns(2)  # Diviser la page en 2 colonnes

        # Champs de texte pour la première colonne
        with columns[0]:
            perimeter_mean = st.text_input('Périmètre moyen', '91.62')
            area_mean = st.text_input('Surface moyenne', '654.89')
            concavity_mean = st.text_input('Concavité moyenne', '0.0888')
            concave_points_mean = st.text_input(
                'Points concaves moyens', '0.0489')
            area_se = st.text_input('Surface (erreur)', '40.47')

        # Champs de texte pour la deuxième colonne
        with columns[1]:
            radius_worst = st.text_input('Rayon (pire)', '16.27')
            texture_worst = st.text_input('Texture (pire)', '25.68')
            perimeter_worst = st.text_input('Périmètre (pire)', '107.2')
            area_worst = st.text_input('Surface (pire)', '880.6')
            concavity_worst = st.text_input('Concavité (pire)', '0.207')
            concave_points_worst = st.text_input(
                'Points concaves (pire)', '0.116')

        # Convertir les entrées en nombres
        perimeter_mean = float(perimeter_mean)
        area_mean = float(area_mean)
        concavity_mean = float(concavity_mean)
        concave_points_mean = float(concave_points_mean)
        area_se = float(area_se)
        radius_worst = float(radius_worst)
        texture_worst = float(texture_worst)
        perimeter_worst = float(perimeter_worst)
        area_worst = float(area_worst)
        concavity_worst = float(concavity_worst)
        concave_points_worst = float(concave_points_worst)

        data_cancer = {
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'concavity_mean': [concavity_mean],
            'concave_points_mean': [concave_points_mean],
            'area_se': [area_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'concavity_worst': [concavity_worst],
            'concave_points_worst': [concave_points_worst],
        }

        cancer_parameters = pd.DataFrame(data_cancer)
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
