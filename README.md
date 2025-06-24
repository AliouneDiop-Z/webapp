## Diabetes Risk Predictor v1

Le **Diabetes Risk Predictor** est une WebApp Streamlit interactive conçue pour estimer la probabilité qu'un patient développe un diabète de type 2 à partir de mesures cliniques basiques (nombre de grossesses, taux de glucose, tension artérielle, etc.). Réalisée dans un but pédagogique, elle met en œuvre un pipeline complet de Machine Learning (prétraitement, entraînement, sauvegarde du modèle) et son déploiement avec Streamlit.

### Dépôt GitHub Python

---

### 1️⃣ ✨ Objectif de la WebApp

* Prédire en temps réel la **probabilité** qu’un patient soit diabétique (en pourcentage).
* Fournir un **diagnostic binaire** (diabétique / non-diabétique).
* Illustrer un **pipeline ML** complet : prétraitement, entraînement, sérialisation du modèle.

### 2️⃣ 🚀 Démo rapide

* **Cloner le dépôt et installer les dépendances** (cf. §6).
* Lancer l’application :

  ```bash
  streamlit run app.py
  ```
* **Dans l’interface :**

  1. Ajuster les sliders et inputs pour les 8 variables cliniques.
  2. Vérifier le DataFrame récapitulatif.
  3. Cliquer sur **Prédire** pour obtenir :

     * La probabilité (pipeline.predict\_proba)
     * Le diagnostic (pipeline.predict)
  4. Résultat affiché en **vert** (non-diabétique) ou **rouge** (diabétique).

### 3️⃣ 📊 Choix du jeu de données

Nous utilisons `diabetes.csv` issu des **Pima Indians Women**, disponible sur Kaggle et la UCI ML Repository.

| Points forts                          | Limites                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| 768 échantillons (taille raisonnable) | Valeurs manquantes codées en 0 (insuline, épaisseur cutanée) |
| Variables numériques standardisées    | Jeu assez petit pour un usage clinique réel                  |
| Classification binaire claire         |                                                              |

**Source** : Kaggle / UCI Machine Learning Repository

### 4️⃣ 🧠 Choix du modèle & pipeline

**Modèle :** Random Forest Classifier de scikit‑learn
**Avantages :** robustesse aux données après imputation, interprétabilité via `feature_importances_`, bonnes performances sans optimisation poussée.

**Pipeline :**

1. **Imputation** par la médiane (`SimpleImputer`) pour remplacer les zéros aberrants
2. **Standardisation** (`StandardScaler`)
3. **Entraînement** du `RandomForestClassifier(random_state=42)`

### 5️⃣ ⚙️ Fonctionnement global

```
./
├── train_model.py      # Chargement des données, construction & entraînement du pipeline, évaluation, sauvegarde dans diabetes_pipeline.pkl
├── app.py              # WebApp Streamlit : chargement du pipeline, interface utilisateur, prédiction
├── diabetes_pipeline.pkl
├── diabetes.csv
└── requirements.txt
```

**Étapes d’entraînement (train\_model.py) :**

* Chargement de `diabetes.csv`
* Construction et entraînement du pipeline
* Évaluation train/test (classification report + ROC‑AUC)
* Sérialisation dans `diabetes_pipeline.pkl`

**Déploiement (app.py) :**

* Chargement de la pipeline sauvegardée
* Interface Streamlit pour saisir les 8 variables
* Affichage dynamique de la probabilité et du diagnostic

### 6️⃣ 📥 Installation & lancement

```bash
# 1. Cloner le repo
git clone https://github.com/votre-utilisateur/diabetes-risk-predictor.git
cd diabetes-risk-predictor

# 2. Créer et activer l’environnement
python -m venv .venv && source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate                         # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer l’application
streamlit run app.py
```

> **Note :** l’application requiert que `diabetes_pipeline.pkl` et `diabetes.csv` se trouvent à la racine du projet.

### 7️⃣ 📎 Ressources

* **Dataset & vidéo explicative** : [Drive](https://drive.google.com/file/d/1lUld_SDHN1H29bADNZaTkn9N1FqZTX2e/view?usp=sharing)

---

*Fin du document.*

Interagir avec les sliders pour tester différents profils patients

liens vers le drive présentant le dataset ainsi que la vidéo explicative
https://drive.google.com/file/d/1lUld_SDHN1H29bADNZaTkn9N1FqZTX2e/view?usp=sharing

