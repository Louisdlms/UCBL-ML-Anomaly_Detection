import numpy as np
np.set_printoptions(threshold=10000, suppress = True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, balanced_accuracy_score
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV

##-------- Exercice 1 --------

def load_data(file_path):
    """Charge les données depuis un fichier."""
    if file_path=="data/mouse.txt":
        data=pd.read_csv(file_path, sep=' ', header=None, names=['x1', 'x2'])
        data['true_outlier']=[1 for k in range (490)]+[-1 for k in range (10)]
    else : # pour les 2 autres fichiers
        data=pd.read_csv(file_path)
    return data

def plot_data(data, title="Nuage de points"):
    """Affiche un nuage de points des données."""
    # Créer un graphique de dispersion
    plt.scatter(data['x1'], data['x2'],c=-data['true_outlier'],cmap='coolwarm')
    # Ajouter des étiquettes aux axes
    plt.xlabel('Axe X1')
    plt.ylabel('Axe X2')
    # Donner un titre au graphique
    plt.title(title)
    # Afficher le graphique
    plt.show()

def detect_outliers_isolation_forest(data, contamination=0.02, random_state=42):
    """Détecte les outliers avec Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=random_state)
    scores = model.fit_predict(data[['x1', 'x2']])  
    data['iso_outlier'] = scores
    return data, model.decision_function(data[['x1', 'x2']])


def detect_outliers_lof(data, n_neighbors=20, contamination=0.02):
    """Détecte les outliers avec LOF."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    scores = model.fit_predict(data[['x1', 'x2']])
    data['lof_outlier'] = scores
    return data, model.negative_outlier_factor_

def plot_outliers(data, method='iso'):
    """Affiche les outliers détectés par une méthode donnée."""
    if method == 'iso':
        plt.scatter(data['x1'], data['x2'], c=-data['iso_outlier'],cmap='coolwarm')
        plt.title("Outliers détectés par Isolation Forest")
    elif method == 'lof':
        plt.scatter(data['x1'], data['x2'], c=-data['lof_outlier'],cmap='coolwarm')
        plt.title("Outliers détectés par LOF")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def plot_anomaly_scores(iso_scores, lof_scores):
    """Affiche les histogrammes des scores d'anomalie."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(iso_scores, bins=50, color='blue')
    plt.title("Histogramme des scores d'anomalie (Isolation Forest)")
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Fréquence")

    plt.subplot(1, 2, 2)
    plt.hist(lof_scores, bins=50, color='red')
    plt.title("Histogramme des scores d'anomalie (LOF)")
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Fréquence")

    plt.tight_layout()
    plt.show()

def adjust_threshold_with_kmeans(data, scores, k=2):
    """Ajuste le seuil avec K-Means."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scores.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_
    threshold = np.mean(cluster_centers)
    outliers = scores < threshold
    return outliers, threshold

def adjust_threshold_with_iqr(data, scores):
    """Ajuste le seuil avec IQR."""
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    outliers = scores < lower_bound
    return outliers, lower_bound


def plot_unsupervised_results(data, iso_kmeans_outliers, iso_kmeans_threshold, iso_iqr_outliers, iso_iqr_threshold,
                 lof_kmeans_outliers, lof_kmeans_threshold, lof_iqr_outliers, lof_iqr_threshold):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(data['x1'], data['x2'], c=iso_kmeans_outliers, cmap='coolwarm')
    plt.title(f"Isolation Forest + K-Means\nSeuil: {iso_kmeans_threshold:.2f}")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(2, 2, 2)
    plt.scatter(data['x1'], data['x2'], c=iso_iqr_outliers, cmap='coolwarm')
    plt.title(f"Isolation Forest + IQR\nSeuil: {iso_iqr_threshold:.2f}")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(2, 2, 3)
    plt.scatter(data['x1'], data['x2'], c=lof_kmeans_outliers, cmap='coolwarm')
    plt.title(f"LOF + K-Means\nSeuil: {lof_kmeans_threshold:.2f}")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(2, 2, 4)
    plt.scatter(data['x1'], data['x2'], c=lof_iqr_outliers, cmap='coolwarm')
    plt.title(f"LOF + IQR\nSeuil: {lof_iqr_threshold:.2f}")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.tight_layout()
    plt.show()

def create_new_data(quantity=10):
    np.random.seed(42)
    new_data = pd.DataFrame({
        'x1': np.random.uniform(low=-0, high=1, size=quantity),
        'x2': np.random.uniform(low=0, high=1, size=quantity)
    })
    return(new_data)

def detect_novelty_lof(train_data, new_data, n_neighbors=20):
    """Détecte les nouveautés avec LOF."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    model.fit(train_data[['x1', 'x2']])
    novelty_scores = model.predict(new_data)
    return novelty_scores

def plot_novelty_detection(data, new_data, novelty_scores):
    plt.scatter(data['x1'], data['x2'], color='grey', label='Données originales', s=10)
    plt.scatter(new_data['x1'], new_data['x2'], c=-novelty_scores, cmap='coolwarm', s=50, edgecolors='black', label='Nouveautés')
    plt.title("Détection de nouveautés avec LOF")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()



##-------- Exercice 2 --------

def preprocess_data(data):
    """Prétraite les données en supprimant la colonne Time et en normalisant les features."""
    if "Time" in data.columns : # Pour CreditCard
        data = data.drop(columns=["Time"])
        X = data.drop(columns=['Class'])
        y = data['Class']
    else : # Pour KDDCup99
        selected_classes = ['normal', 'buffer_overflow'] # On choisit uniquement un des types d'outliers à détecter : ici 'buffer_overflow'
        data = data[data['label'].isin(selected_classes)]
        X = data.drop(columns=['label'])
        X = pd.get_dummies(X, drop_first=True)
        y = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Sépare les données en ensembles d'entraînement et de test de manière stratifiée."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def train_easy_ensemble(X_train, y_train):
    """Entraîne un modèle EasyEnsemble."""
    model = EasyEnsembleClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_isolation_forest(X_train, n_estimators=100, contamination='auto'):
    """Entraîne un modèle Isolation Forest."""
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

def train_lof(X_train, n_neighbors=20, contamination='auto'):
    """Entraîne un modèle LOF."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    model.fit(X_train)
    return model

def train_xgboost(X_train, y_train):
    """Entraîne un modèle XGBoost."""
    model = XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]), random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Entraîne un modèle Random Forest."""
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_with_tomek_links(X_train, y_train):
    """Entraîne un modèle avec Tomek Links pour l'undersampling."""
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_train, y_train)
    model = XGBClassifier(random_state=42)
    model.fit(X_res, y_res)
    return model

def train_with_smote(X_train, y_train):
    """Entraîne un modèle avec SMOTE pour l'oversampling."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(random_state=42)
    model.fit(X_res, y_res)
    return model

def optimize_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def find_optimal_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


def evaluate_model(model, X_test, y_test, optimize_threshold=False):
    """
    Évalue le modèle sur les données de test et ajuste le seuil si demandé.
    """
    # Calcul des scores et prédictions
    if hasattr(model, 'predict_proba'):
        # Modèles supervisés : scores = probabilités de la classe positive
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        optimal_threshold = 0.5  # Seuil par défaut pour les modèles supervisés

    elif hasattr(model, 'decision_function'):
        # Modèles comme Isolation Forest : scores = -decision_function
        y_scores = -model.decision_function(X_test)
        y_pred = model.predict(X_test)
        y_pred = [1 if pred == -1 else 0 for pred in y_pred]
        optimal_threshold = None

    else:
        # Modèles comme LOF : scores = -negative_outlier_factor_
        y_scores = -model.negative_outlier_factor_
        y_pred = model.predict(X_test)
        y_pred = [1 if pred == -1 else 0 for pred in y_pred]
        optimal_threshold = None

    # Optimisation du seuil pour les modèles non supervisés
    if optimize_threshold:
        optimal_threshold = find_optimal_threshold(y_test, y_scores)
        y_pred_adjusted = (y_scores >= optimal_threshold).astype(int)
    else:
        y_pred_adjusted = y_pred

    # Calcul des métriques
    conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
    f1 = f1_score(y_test, y_pred_adjusted, average='binary')
    roc_auc = roc_auc_score(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_adjusted)

    return conf_matrix, f1, roc_auc, avg_precision, balanced_acc, y_scores

def plot_metrics(y_true, y_scores):
    """Affiche les courbes ROC et Precision-Recall."""
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def result_evaluation(models, X_test, y_test, optimize_threshold=False):
    # Évaluer chaque modèle
    results = {}
    for name, model in models.items():
        try:
            conf_matrix, f1, roc_auc, avg_precision, balanced_acc, y_scores = evaluate_model(model, X_test, y_test, optimize_threshold=optimize_threshold)
            results[name] = {
                'Confusion Matrix': conf_matrix,
                'F1-score': f1,
                'ROC AUC': roc_auc,
                'Average Precision': avg_precision,
                'Balanced Accuracy': balanced_acc
            }
            print(f"{name} Results:")
            print(f"Confusion Matrix:\n{conf_matrix}\n")
            print(f"F1-score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}\n")
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"Average Precision: {avg_precision:.4f}\n")

            # Afficher les courbes ROC et Precision-Recall
            plot_metrics(y_test, y_scores)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

def cross_validate_model(model, X, y, cv=StratifiedKFold(n_splits=5), optimize_threshold=False):
    """Effectue une validation croisée stratifiée."""
    f1_scores = []
    roc_auc_scores = []
    avg_precision_scores = []
    balanced_acc_scores = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if hasattr(model, 'fit_resample'):
            X_res, y_res = model.fit_resample(X_train, y_train)
            model = XGBClassifier(random_state=42)
            model.fit(X_res, y_res)
        else:
            if hasattr(model, 'fit_predict'):
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

        _, f1, roc_auc, avg_precision, balanced_acc, _ = evaluate_model(model, X_test, y_test, optimize_threshold=optimize_threshold)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)
        avg_precision_scores.append(avg_precision)
        balanced_acc_scores.append(balanced_acc)

    return np.mean(f1_scores), np.mean(roc_auc_scores), np.mean(avg_precision_scores), np.mean(balanced_acc_scores)


def result_cross_val(models, X_scaled, y, optimize_threshold=False):
    # Validation croisée pour chaque modèle
    cv_results = {}
    for name, model in models.items():
        if name in ['IsolationForest', 'LOF']:
            continue  # Ces modèles ne sont pas adaptés pour la validation croisée supervisée
        try:
            f1, roc_auc, avg_precision, balanced_acc = cross_validate_model(model, X_scaled, y, optimize_threshold=optimize_threshold)
            cv_results[name] = {
                'F1-score': f1,
                'ROC AUC': roc_auc,
                'Average Precision': avg_precision,
                'Balanced Accuracy': balanced_acc
            }
            print(f"{name} Cross-Validation Results:")
            print(f"F1-score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        except Exception as e:
            print(f"Error in cross-validation for {name}: {e}")



