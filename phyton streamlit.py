# train_model.py
# ----------------------------
# Ce script sert à charger les données, entraîner un modèle, évaluer et sauvegarder la pipeline.

import pandas as pd  # on importe pandas pour manipuler les tableaux (DataFrame)
from sklearn.model_selection import train_test_split  # pour découper les données en train/test
from sklearn.impute import SimpleImputer  # pour remplacer les valeurs manquantes
from sklearn.preprocessing import StandardScaler  # pour normaliser les données
from sklearn.ensemble import RandomForestClassifier  # pour utiliser un modèle de forêt aléatoire
from sklearn.metrics import classification_report, roc_auc_score  # pour évaluer la qualité du modèle
from sklearn.pipeline import Pipeline  # pour chaîner imputation, mise à l’échelle et modèle
import joblib  # pour sauvegarder et recharger la pipeline  

# 1) On charge le fichier CSV dans un DataFrame
df = pd.read_csv("C:/Users/aliox/Downloads/diabetes.csv")  # attention à bien mettre le bon chemin
print("✅ Fichier chargé, aperçu :")
print(df.head())  # on affiche les 5 premières lignes pour vérifier

# 2) On sépare les variables explicatives (X) de la cible (y)
X = df.drop("Outcome", axis=1)  # X = toutes les colonnes sauf "Outcome"
y = df["Outcome"]               # y = la colonne "Outcome" (0 ou 1)

# 3) On définit la pipeline en trois étapes : 
#    - SimpleImputer : remplace les valeurs manquantes par la médiane
#    - StandardScaler : centre/réduit les données (moyenne=0, écart-type=1)
#    - RandomForestClassifier : crée et entraîne le modèle
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # imputation médiane
    ("scaler", StandardScaler()),                   # normalisation
    ("classifier", RandomForestClassifier(random_state=42))  # modèle
])

# 4) On découpe les données en train et test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% pour tester
    random_state=42,  # garantit la reproductibilité
    stratify=y        # garde la même proportion de 0/1 qu’à l’origine
)
print(f"🔍 Train: {X_train.shape}, Test: {X_test.shape}")

# 5) On entraîne la pipeline sur les données d’entraînement
print("⏳ Entraînement en cours...")
pipe.fit(X_train, y_train)

# 6) On prédit sur le jeu de test et on évalue
y_pred = pipe.predict(X_test)                 # étiquette prédite
y_prob = pipe.predict_proba(X_test)[:, 1]     # probabilité d’être diabétique (classe=1)

print("📝 Rapport de classification :")
print(classification_report(y_test, y_pred))  # precision, recall, f1-score
print("🏅 AUC-ROC :", roc_auc_score(y_test, y_prob))  # score ROC-AUC

# 7) On sauvegarde la pipeline entraînée pour la réutiliser plus tard
joblib.dump(pipe, "diabetes_pipeline.pkl")
print("💾 Pipeline enregistrée dans diabetes_pipeline.pkl")


