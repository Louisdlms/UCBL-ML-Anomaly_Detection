# ==========================================================
# utils.py
# Fonctions utilitaires pour la partie 1 du TP Détection d'anomalies
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.set_printoptions(threshold=10000, suppress = True)
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# Fonction 1 : Chargement du jeu de données
# ----------------------------------------------------------
def load_mouse_data(path):
    """
    Charge le fichier mouse.txt et renvoie un DataFrame.
    """
    data = pd.read_csv(path, sep=r"\s+", header=None, names=["x1", "x2"])
    return data


# ----------------------------------------------------------
# Fonction 2 : Analyse statistique de base
# ----------------------------------------------------------
def describe_data(df):
    """
    Affiche les dimensions, les premières lignes,
    les statistiques descriptives et la présence de valeurs manquantes.
    """
    print("Nombre d'observations :", df.shape[0])
    print("Nombre de variables :", df.shape[1])
    print("\nAperçu du jeu de données :")
    print(df.head())

    print("\nStatistiques descriptives :")
    print(df.describe())

    print("\nValeurs manquantes :")
    print(df.isna().sum())


# ----------------------------------------------------------
# Fonction 3 : Visualisation des données
# ----------------------------------------------------------
def plot_mouse_data(df):
    """
    Affiche le nuage de points x1/x2 et la matrice de corrélation.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(df["x1"], df["x2"], s=20, color="steelblue")
    plt.title("Visualisation du jeu de données Mouse")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()



# ----------------------------------------------------------
# Fonction 4 : Visualisation des distributions individuelles
# ----------------------------------------------------------
def plot_distributions(df):
    """
    Affiche les histogrammes (avec courbe de densité) de chaque variable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    sns.histplot(df["x1"], bins=20, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Distribution de x1")

    sns.histplot(df["x2"], bins=20, kde=True, ax=axes[1], color="lightcoral")
    axes[1].set_title("Distribution de x2")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Fonction 5 : Détection d'anomalies avec Isolation Forest
# ----------------------------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def run_isolation_forest(df, contamination=0.02, random_state=42, show_results=True):
    """
    Applique Isolation Forest sur les données pour détecter les outliers.
    
    Paramètres :
    ------------
    df : DataFrame
        Données d'entrée (colonnes x1 et x2)
    contamination : float
        Proportion estimée d'anomalies (ici 10/500 = 0.02)
    random_state : int
        Graine aléatoire pour la reproductibilité
    show_results : bool
        Si True, affiche les résultats de la détection

    Retour :
    --------
    preds : ndarray
        Tableau de 0 (normal) et 1 (anomalie)
    scores : ndarray
        Score d'anomalie (plus grand = plus anormal)
    """

    # Création et entraînement du modèle
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iso.fit(df)

    # Scores et prédictions
    scores = -iso.decision_function(df)  # plus grand = plus anormal
    threshold = np.percentile(scores, 100 * (1 - contamination))
    preds = (scores > threshold).astype(int)

    if show_results:
        print(f"Seuil utilisé : {threshold}")
        print(f"Nombre d'anomalies détectées : {preds.sum()} sur {len(preds)} observations")

    return preds, scores



# ----------------------------------------------------------
# Fonction 6 : Détection d'anomalies avec Local Outlier Factor
# ----------------------------------------------------------
from sklearn.neighbors import LocalOutlierFactor

def run_lof(df, contamination=0.02, n_neighbors=20, show_results=True):
    """
    Applique le modèle Local Outlier Factor (LOF) pour détecter les outliers.

    Paramètres :
    ------------
    df : DataFrame
        Données d'entrée (colonnes x1 et x2)
    contamination : float
        Proportion estimée d'anomalies (ex: 10/500 = 0.02)
    n_neighbors : int
        Nombre de voisins pour le calcul du facteur local
    show_results : bool
        Si True, affiche le nombre d'anomalies détectées

    Retour :
    --------
    preds : ndarray
        Tableau de 0 (normal) et 1 (anomalie)
    scores : ndarray
        Score d'anomalie (plus grand = plus anormal)
    """

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof.fit_predict(df)
    scores = -lof.negative_outlier_factor_          # inversion du score

    threshold = np.percentile(scores, 100 * (1 - contamination))
    preds = (scores > threshold).astype(int)

    if show_results:
        print(f"Seuil utilisé : {threshold}")
        print(f"Nombre d'anomalies détectées : {preds.sum()} sur {len(preds)} observations")

    return preds, scores


# ----------------------------------------------------------
# Fonction 7 : Visualisation des anomalies détectées
# ----------------------------------------------------------
import matplotlib.pyplot as plt

def plot_anomalies(df, preds, title="Anomalies détectées"):
    """
    Affiche les points normaux et anormaux dans un plan 2D.
    
    Paramètres :
    ------------
    df : DataFrame
        Données (colonnes x1, x2)
    preds : array-like
        Tableau de 0 (normal) et 1 (anomalie)
    title : str
        Titre du graphique
    """

    plt.figure(figsize=(6,6))
    plt.scatter(df["x1"], df["x2"], s=20, color="lightgray", label="Normaux")
    plt.scatter(df.loc[preds==1, "x1"], df.loc[preds==1, "x2"],
                s=70, color="red", marker="x", label="Anomalies")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()



# ----------------------------------------------------------
# Fonction 8 : Histogrammes des scores d'anomalie
# ----------------------------------------------------------
import seaborn as sns

def plot_anomaly_scores(scores_if, scores_lof):
    """
    Affiche les histogrammes des scores d'anomalie
    pour Isolation Forest et LOF.
    
    Paramètres :
    ------------
    scores_if : array-like
        Scores d'anomalie de l'Isolation Forest
    scores_lof : array-like
        Scores d'anomalie du Local Outlier Factor
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # Histogramme Isolation Forest
    sns.histplot(scores_if, bins=30, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Scores d'anomalie - Isolation Forest")
    axes[0].set_xlabel("Score (plus élevé = plus anormal)")
    axes[0].set_ylabel("Fréquence")

    # Histogramme LOF
    sns.histplot(scores_lof, bins=30, kde=True, ax=axes[1], color="orange")
    axes[1].set_title("Scores d'anomalie - Local Outlier Factor")
    axes[1].set_xlabel("Score (plus élevé = plus anormal)")
    axes[1].set_ylabel("Fréquence")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Fonction : Seuil IQR avec ajustement automatique
# ----------------------------------------------------------
import numpy as np

def threshold_iqr_auto(scores, target_rate=0.02, tol=0.005):
    """
    Méthode IQR améliorée :
    ajuste automatiquement le coefficient pour obtenir un nombre
    d'anomalies proche de target_rate (ex: 0.02 = 2%).
    
    Paramètres :
    ------------
    scores : array-like
        Scores d'anomalie (plus grand = plus anormal)
    target_rate : float
        Proportion cible d'anomalies (~2%)
    tol : float
        Tolérance autour de target_rate
    
    Retour :
    --------
    threshold : float
        Seuil final choisi
    preds : ndarray
        0 = normal, 1 = anomalie
    coef : float
        Coefficient final utilisé
    """

    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1

    coef = 1.5  # valeur initiale classique
    n = len(scores)
    preds = np.zeros(n)
    nb_anomalies = 0

    # Ajustement automatique du coefficient IQR
    while True:
        threshold = q3 + coef * iqr
        preds = (scores > threshold).astype(int)
        nb_anomalies = preds.sum()
        rate = nb_anomalies / n

        if abs(rate - target_rate) <= tol or coef < 0.1 or coef > 5:
            break
        elif rate < target_rate:
            coef *= 0.9  # abaisse le seuil pour détecter plus
        else:
            coef *= 1.1  # augmente le seuil pour détecter moins

    return threshold, preds, coef


# ----------------------------------------------------------
# Fonction : Clustering (KMeans) avec ajustement automatique
# ----------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def threshold_clustering_auto(scores, target_rate=0.02):
    """
    Méthode de clustering améliorée :
    utilise KMeans sur les scores standardisés puis ajuste le seuil
    pour obtenir un taux d’anomalies proche de target_rate.
    """
    X = np.array(scores).reshape(-1, 1)
    X_scaled = StandardScaler().fit_transform(X)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Identifier le cluster le plus anormal (moyenne la plus élevée)
    cluster_means = [scores[np.where(labels == i)].mean() for i in range(2)]
    anomaly_cluster = np.argmax(cluster_means)

    preds = (labels == anomaly_cluster).astype(int)
    
    # Ajustement automatique : si le taux est trop loin de la cible
    rate = preds.sum() / len(preds)
    if rate < target_rate * 0.5 or rate > target_rate * 1.5:
        # Ajuste un seuil intermédiaire basé sur la moyenne des scores du cluster anormal
        sorted_scores = np.sort(scores)
        k = int(len(scores) * target_rate)
        threshold = sorted_scores[-k] if k > 0 else sorted_scores[-1]
        preds = (scores >= threshold).astype(int)
    else:
        threshold = np.mean(scores[labels == anomaly_cluster])

    return threshold, preds, rate


# ----------------------------------------------------------
# Fonction 11 : Évaluation des résultats
# ----------------------------------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_results(y_true, y_pred, name="Méthode"):
    """
    Calcule et affiche les métriques de performance pour une méthode donnée.
    """
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"=== {name} ===")
    print(f"Précision : {p:.3f} | Rappel : {r:.3f} | F1-score : {f1:.3f}")
    print("Matrice de confusion :")
    print(cm)
    print()

    return {"precision": p, "recall": r, "f1": f1, "cm": cm}


# ----------------------------------------------------------
# Fonction 12 : Comparaison des méthodes
# ----------------------------------------------------------
import pandas as pd

def compare_methods(results_dict):
    """
    Crée un tableau de comparaison à partir des métriques calculées.
    results_dict : dict {nom_méthode: dict_métriques}
    """
    summary = pd.DataFrame.from_dict(results_dict, orient="index")
    display(summary[["precision", "recall", "f1"]])


# ----------------------------------------------------------
# Fonction 13 : Détection de nouveautés avec LOF (novelty=True)
# ----------------------------------------------------------
from sklearn.neighbors import LocalOutlierFactor

def run_lof_novelty(train_df, test_df, contamination=0.02, n_neighbors=20, show_results=True):
    """
    Entraîne un LOF en mode novelty detection sur des données normales,
    puis prédit les anomalies sur un jeu de test.

    Paramètres :
    ------------
    train_df : DataFrame
        Jeu d'entraînement (données normales uniquement)
    test_df : DataFrame
        Jeu de test (peut contenir des anomalies)
    contamination : float
        Proportion estimée d'anomalies dans le test
    n_neighbors : int
        Nombre de voisins LOF
    show_results : bool
        Si True, affiche le nombre d'anomalies détectées

    Retour :
    --------
    preds : ndarray
        0 = normal, 1 = anomalie
    scores : ndarray
        Score d'anomalie (plus grand = plus anormal)
    """

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_df)

    scores = -lof.decision_function(test_df)  # plus grand = plus anormal
    threshold = np.percentile(scores, 100 * (1 - contamination))
    preds = (scores > threshold).astype(int)

    if show_results:
        print(f"Seuil utilisé : {threshold:.4f}")
        print(f"Nombre d'anomalies détectées : {preds.sum()} sur {len(preds)} observations")

    return preds, scores


