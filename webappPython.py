# app.py
# ----------------------------
# WebApp Streamlit pour saisir les données patient, charger le modèle et afficher la prédiction.

import streamlit as st  # on importe Streamlit pour faire l’interface
import joblib           # pour recharger la pipeline sauvegardée
import pandas as pd     # pour créer un DataFrame à partir des valeurs utilisateur

# 1) On charge la pipeline (modèle + prétraitement)
try:
    pipeline = joblib.load("diabetes_pipeline.pkl")  # on lit le fichier .pkl
    st.sidebar.success("Modèle chargé ")           # message si tout va bien
except FileNotFoundError:
    st.sidebar.error("Erreur : diabetes_pipeline.pkl introuvable")  # sinon on affiche une erreur

# 2) Titre de l’application
st.title(" 🔮Prédiction du diabète (Version débutant)")

# 3) Sidebar pour recueillir les infos patient
st.sidebar.header("🩺 Paramètres patient")
pregnancies    = st.sidebar.number_input("Grossesses", min_value=0, max_value=20, value=1)
glucose        = st.sidebar.slider("Glucose", 0, 200, 120)
blood_pressure = st.sidebar.slider("Tension artérielle", 0, 140, 70)
skin_thickness = st.sidebar.slider("Épaisseur de peau", 0, 100, 20)
insulin        = st.sidebar.slider("Insuline", 0, 900, 79)
bmi            = st.sidebar.slider("IMC", 0.0, 70.0, 25.0)
dpf            = st.sidebar.slider("DiabetesPedigreeFunc", 0.0, 3.0, 0.5)
age            = st.sidebar.slider("Âge", 10, 100, 30)

# 4) On crée un DataFrame à partir des valeurs saisies
input_data = {
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
}
input_df = pd.DataFrame(input_data)  # transforme la dict en tableau

st.write("### Données saisies")
st.write(input_df)  # on affiche ce qu’on va prédire

# 5) Bouton de prédiction : quand on clique, on calcule et on affiche
if st.button("🩸 Prédire"):
    prob = pipeline.predict_proba(input_df)[0, 1]  # récupère la probabilité
    pred = pipeline.predict(input_df)[0]           # récupère la classe (0 ou 1)

    # on affiche la probabilité avec 1 décimale
    st.write(f"**Probabilité d’être diabétique** : {prob * 100:.1f}%")
    # on affiche un message coloré selon le résultat
    if pred == 1:
        st.error("👉 Résultat : **Diabétique**")
    else:
        st.success("👉 Résultat : **Non diabétique**")

# 6) Pour comprendre predict/predict_proba, voilà la doc :
st.markdown(
    "ℹ️ *sickit learn* : "
    "https://scikit-learn.org/stable/modules/generated/"
    "sklearn.ensemble.RandomForestClassifier.html"
)



st.markdown(
    "🤔 *Le sucre nous rend-il bêtes ?*  *ARTE* : "
    "https://www.youtube.com/watch?v=XcZ7CnAEgB0&t=139s"
    
)

#a entrer dans le terminal : PS C:\Users\aliox\Downloads\data ndb1> streamlit run webappPython.py
