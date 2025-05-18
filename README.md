Objectif de la WebApp

Cette application interactive a pour but de prédire la probabilité qu'un patient soit atteint de diabète de type 2 à partir de mesures cliniques basiques (nombre de grossesses, taux de glucose, tension artérielle, etc.). L’utilisateur peut saisir ces paramètres via une interface web et obtenir en temps réel :

Une probabilité d’être diabétique (en pourcentage)

Un diagnostic binaire (diabétique / non-diabétique)

L’objectif pédagogique est de mettre en place un pipeline complet de Machine Learning (prétraitement, entraînement, sauvegarde du modèle) et de le déployer simplement avec Streamlit.

Choix du dataset

Nous utilisons le jeu de données diabetes.csv issu des Pima Indians Women disponible sur Kaggle et la UCI Machine Learning Repository. Ce dataset est couramment utilisé pour des projets de détection du diabète :

Avantages :

Taille raisonnable (768 échantillons), idéal pour des démonstrations locales

Variables numériques standardisées (Glucose, BMI, etc.)

Classification binaire claire (Outcome = 0 ou 1)

Inconvénients :

Présence de valeurs manquantes codées en 0 (ex. Insulin, SkinThickness)

Jeu assez petit, donc modèle à prendre avec précaution pour un usage clinique réel

Choix du modèle

Nous avons opté pour un Random Forest Classifier de scikit-learn pour les raisons suivantes :

Robustesse aux valeurs manquantes après imputation, et capacité à gérer des variables de nature différente sans trop de réglages

Interprétabilité relative : on peut extraire l’importance des variables par feature_importances_

Performance généralement bonne même sans optimisation poussée

Simplicité de mise en œuvre avec la Pipeline de scikit-learn

La pipeline se compose de :

Imputation par la médiane (SimpleImputer) pour remplacer les zéros aberrants

Standardisation (StandardScaler) pour centrer/réduire chaque variable

Entraînement du RandomForestClassifier avec random_state=42 pour la reproductibilité

Fonctionnement global de l'application

Entraînement (train_model.py) :

Chargement de diabetes.csv

Construction et entraînement de la pipeline (imputation → standardisation → modèle)

Évaluation sur un split train/test (classification report + ROC-AUC)

Sauvegarde de la pipeline dans diabetes_pipeline.pkl

Déploiement (app.py) :

Chargement de la pipeline sauvegardée

Interface Streamlit :

Sliders et number_input pour saisir les 8 variables cliniques

Affichage du DataFrame saisi pour confirmation

Bouton « Prédire » qui déclenche :

pipeline.predict_proba pour obtenir la probabilité

pipeline.predict pour le diagnostic binaire

Affichage coloré du résultat (vert pour non diabétique, rouge pour diabétique)

Utilisation :

Dans un terminal, exécuter streamlit run app.py

Ouvrir l’URL locale fournie (généralement http://localhost:8501)

Interagir avec les sliders pour tester différents profils patients
