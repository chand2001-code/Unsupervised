from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans, DBSCAN, Birch, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import shap
import scipy.cluster.hierarchy as sch
import seaborn as sns
import logging
from typing import Optional, Dict, Any
import uvicorn



logging.basicConfig(level=logging.INFO)

ALGORITHMS = [
    "K-Means clustering",
    "K-Modes clustering",
    "K-Prototypes clustering",
    "Hierarchical clustering",
    "DBSCAN clustering",
    "BIRCH clustering",
    "Spectral clustering"
]

ANOMALY_DETECTION_ALGORITHMS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "Minimum Covariance Determinant"
]

def read_and_preprocess_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

def safe_label_encoder(series):
    le = LabelEncoder()
    return pd.Series(le.fit_transform(series.astype(str)), index=series.index)

def calculate_metrics(df, max_clusters=12):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow_graph = {}
    calinski_graph = {}
    davies_graph = {}
    silhouette_graph = {}

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        elbow_graph[k] = kmeans.inertia_
        calinski_graph[k] = calinski_harabasz_score(X_scaled, labels)
        davies_graph[k] = davies_bouldin_score(X_scaled, labels)
        silhouette_graph[k] = silhouette_score(X_scaled, labels)

    optimal_k = max(silhouette_graph, key=silhouette_graph.get)

    return optimal_k, {
        "elbow_graph": list(elbow_graph.items()),
        "calinski_graph": list(calinski_graph.items()),
        "davies_graph": list(davies_graph.items()),
        "silhouette_graph": list(silhouette_graph.items())
    }

def run_clustering_algorithm(algorithm, X_scaled, n_clusters, X_cat=None):
    try:
        if algorithm == "K-Means clustering":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = model.fit_predict(X_scaled)
            return labels, model.inertia_, None
        elif algorithm == "K-Modes clustering":
            if X_cat is None:
                return None, None, None
            X_cat_encoded = X_cat.apply(safe_label_encoder)
            model = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            labels = model.fit_predict(X_cat_encoded)
            return labels, model.cost_, None
        elif algorithm == "K-Prototypes clustering":
            if X_cat is None:
                return None, None, None
            X_cat_encoded = X_cat.apply(safe_label_encoder)
            X_mixed = np.column_stack((X_scaled, X_cat_encoded))
            categorical_indices = list(range(X_scaled.shape[1], X_mixed.shape[1]))
            model = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            labels = model.fit_predict(X_mixed, categorical=categorical_indices)
            return labels, model.cost_, None
        elif algorithm == "Hierarchical clustering":
            # Create a linkage matrix and plot the dendrogram
            linkage_matrix = sch.linkage(X_scaled, method='ward')
            plt.figure(figsize=(10, 7))
            sch.dendrogram(linkage_matrix)
            plt.title('Dendrogram for Hierarchical Clustering')
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            plt.tight_layout()

            # Save the dendrogram plot to a byte stream
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            dendrogram_img = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            # Fit the model to get cluster labels
            model = AgglomerativeClustering(n_clusters=n_clusters)  # Specify n_clusters
            cluster_labels = model.fit_predict(X_scaled)  # Clustering for label purposes
            
            # Calculate silhouette score for Hierarchical clustering
            if len(set(cluster_labels)) > 1:
                silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
                avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
            else:
                avg_silhouette_score = -1  # Indicate insufficient clusters for silhouette score

            return cluster_labels, None, dendrogram_img  # Return dendrogram image

        elif algorithm == "DBSCAN clustering":
            model = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = model.fit_predict(X_scaled)

            # Create a dendrogram for DBSCAN using hierarchical clustering on the same data
            linkage_matrix = sch.linkage(X_scaled, method='ward')
            plt.figure(figsize=(10, 7))
            sch.dendrogram(linkage_matrix)
            plt.title('Dendrogram for DBSCAN Clustering')
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            plt.tight_layout()

            # Save the dendrogram plot to a byte stream
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            dendrogram_img = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            # Calculate silhouette score if more than one cluster is found
            if len(set(cluster_labels)) > 1:
                silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
                avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
            else:
                avg_silhouette_score = -1  # Indicate insufficient clusters for silhouette score

            return cluster_labels, None, dendrogram_img  # Return dendrogram image

        elif algorithm == "BIRCH clustering":
            model = Birch()
            cluster_labels = model.fit_predict(X_scaled)

            # Create a dendrogram for BIRCH using hierarchical clustering on the same data
            linkage_matrix = sch.linkage(X_scaled, method='ward')
            plt.figure(figsize=(10, 7))
            sch.dendrogram(linkage_matrix)
            plt.title('Dendrogram for BIRCH Clustering')
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            plt.tight_layout()

            # Save the dendrogram plot to a byte stream
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            dendrogram_img = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            # Calculate silhouette score if more than one cluster is found
            if len(set(cluster_labels)) > 1:
                silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
                avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
            else:
                avg_silhouette_score = -1  # Indicate insufficient clusters for silhouette score

            return cluster_labels, None, dendrogram_img  # Return dendrogram image

        elif algorithm == "Spectral clustering":
            model = SpectralClustering(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = model.fit_predict(X_scaled)
            return labels, None, None
    except Exception as e:
        logging.error(f"Error in {algorithm}: {str(e)}")
    return None, None, None

def calculate_cluster_metrics(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        logging.warning(f"Only one cluster found. Returning zero values for metrics.")
        return {
            "silhouette": 0,
            "calinski": 0,
            "davies": 0
        }
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels),
        "davies": davies_bouldin_score(X, labels)
    }

def shap_analysis(df: pd.DataFrame, algorithm: str, n_clusters: int, X_scaled: np.ndarray) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the CSV file")

    df_sampled = df.sample(frac=0.1, random_state=42) if len(df) > 1000 else df
    X_numeric = df_sampled[numeric_cols]
    X_categorical = df_sampled[categorical_cols]

    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    model = None
    cluster_labels = None

    try:
        if algorithm == "K-Modes clustering":
            if len(categorical_cols) == 0:
                raise ValueError("K-Modes clustering requires categorical data")
            X_cat_encoded = X_categorical.apply(LabelEncoder().fit_transform)
            model = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            cluster_labels = model.fit_predict(X_cat_encoded)
            X_scaled = X_cat_encoded
        elif algorithm == "K-Prototypes clustering":
            if len(categorical_cols) == 0:
                raise ValueError("K-Prototypes clustering requires both numeric and categorical data")
            X_cat_encoded = X_categorical.apply(LabelEncoder().fit_transform)
            X_combined = np.hstack((X_numeric_scaled, X_cat_encoded))
            categorical_indices = list(range(X_numeric_scaled.shape[1], X_combined.shape[1]))
            model = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            cluster_labels = model.fit_predict(X_combined, categorical=categorical_indices)
            X_scaled = X_combined
        elif algorithm == "K-Means clustering":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "Hierarchical clustering":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(X_scaled)
        elif algorithm == "DBSCAN clustering":
            model = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = model.fit_predict(X_scaled)
        elif algorithm == "BIRCH clustering":
            model = Birch(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(X_scaled)
        elif algorithm == "Spectral clustering":
            model = SpectralClustering(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = model.fit_predict(X_numeric_scaled)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if cluster_labels is None:
            return {"error": "Cluster labels are not defined."}

        surrogate_model = LogisticRegression(random_state=42)
        surrogate_model.fit(X_numeric_scaled, cluster_labels)

        def surrogate_predict(X):
            return surrogate_model.predict(X)

        explainer = shap.KernelExplainer(surrogate_predict, X_numeric_scaled[:100])
        shap_values = explainer.shap_values(X_numeric_scaled, nsamples=50)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_numeric_scaled, plot_type="dot",
                          feature_names=list(numeric_cols), show=False, color_bar=True, max_display=10)
        plt.title(f"SHAP Summary Plot for {algorithm}")
        plt.tight_layout()

        img_stream = BytesIO()
        plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img_stream.seek(0)
        beeswarm_plot = base64.b64encode(img_stream.getvalue()).decode('utf-8')

        feature_importance = []
        for i, col in enumerate(numeric_cols):
            importance = np.abs(shap_values[i]).mean()
            feature_importance.append({"feature": col, "importance": importance})

        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        response = {
            "beeswarm_plot": f"data:image/png;base64,{beeswarm_plot}",
            "feature_importance": feature_importance,
            "algorithm": algorithm
        }

        return response

    except Exception as e:
        logging.error(f"Error in SHAP analysis: {str(e)}")
        return {"error": f"SHAP analysis failed: {str(e)}"}

def generate_plot(X_scaled, labels, title, plot_type='cluster'):
    if plot_type == 'cluster':
        fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=labels, title=f"Cluster Plot: {title}")
    elif plot_type == 'silhouette':
        silhouette_vals = silhouette_samples(X_scaled, labels)
        y_lower, y_upper = 0, 0
        yticks = []
        for i in range(len(np.unique(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            yticks.append((y_lower + y_upper) / 2)
            y_lower += len(cluster_silhouette_vals)
        
        fig = px.bar(x=silhouette_vals, y=range(len(silhouette_vals)), orientation='h',
                     title=f"Silhouette Plot: {title}", labels={'x': 'Silhouette coefficient', 'y': 'Cluster'})
        fig.update_yaxes(tickvals=yticks, ticktext=list(range(len(np.unique(labels)))))

    fig.update_layout(hovermode='closest')
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_anomalies(X_scaled, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='coolwarm', marker='o', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly Status (-1: Anomaly, 1: Normal)')
    plt.grid()
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return base64.b64encode(img_stream.getvalue()).decode('utf-8')

def anomaly_detection(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"error": "Input DataFrame is None or empty; cannot perform anomaly detection."}

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the DataFrame.")

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = {}

        for algorithm in ANOMALY_DETECTION_ALGORITHMS:
            try:
                if algorithm == "Isolation Forest":
                    model = IsolationForest(contamination=0.1, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    results[algorithm] = {
                        "anomalies": np.sum(labels == -1),
                        "plot": plot_anomalies(X_scaled, labels, f"{algorithm} Anomaly Detection")
                    }

                elif algorithm == "Local Outlier Factor":
                    model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                    labels = model.fit_predict(X_scaled)
                    results[algorithm] = {
                        "anomalies": np.sum(labels == -1),
                        "plot": plot_anomalies(X_scaled, labels, f"{algorithm} Anomaly Detection")
                    }

                elif algorithm == "Minimum Covariance Determinant":
                    model = EllipticEnvelope(contamination=0.1, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    results[algorithm] = {
                        "anomalies": np.sum(labels == -1),
                        "plot": plot_anomalies(X_scaled, labels, f"{algorithm} Anomaly Detection")
                    }

            except Exception as e:
                logging.error(f"Error in {algorithm}: {str(e)}")
                results[algorithm] = {"error": str(e)}

        return results

    except Exception as e:
        logging.error(f"Error during anomaly detection: {str(e)}")
        return {"error": str(e)}

def run_clustering_analysis(df):
    if df is None or df.empty:
        return {"error": "Input DataFrame is None or empty; cannot perform clustering analysis."}

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the CSV file")

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        optimal_k, metric_graphs = calculate_metrics(df)

        results = {algo: {} for algo in ALGORITHMS}
        shap_results = {}

        for algorithm in ALGORITHMS:
            if algorithm in ["K-Modes clustering", "K-Prototypes clustering"] and len(categorical_cols) == 0:
                logging.info(f"Skipping {algorithm} due to lack of categorical data.")
                continue

            labels, inertia, dendrogram_img = run_clustering_algorithm(
                algorithm,
                X_scaled,
                optimal_k,
                df[categorical_cols] if len(categorical_cols) > 0 else None
            )

            if labels is not None and len(np.unique(labels)) > 1:
                metrics = calculate_cluster_metrics(X_scaled, labels)
                results[algorithm] = {
                    "inertia": inertia if inertia is not None else 0,
                    "calinski": metrics["calinski"],
                    "davies": metrics["davies"],
                    "silhouette": metrics["silhouette"],
                    "cluster_plot": generate_plot(X_scaled, labels, f"{algorithm} (k={optimal_k})", 'cluster'),
                    "silhouette_plot": generate_plot(X_scaled, labels, f"{algorithm} (k={optimal_k})", 'silhouette'),
                    "dendrogram": dendrogram_img  # Include dendrogram image
                }

                shap_result = shap_analysis(df, algorithm, optimal_k, X_scaled)  # Pass X_scaled
                shap_results[algorithm] = shap_result
            else:
                logging.warning(f"Skipping {algorithm} due to insufficient unique labels.")

        # Run anomaly detection
        anomaly_results = anomaly_detection(df)

        return {
            "optimal_k": optimal_k,
            # "elbow_graph": metric_graphs["elbow_graph"],
            # "calinski_graph": metric_graphs["calinski_graph"],
            # "davies_graph": metric_graphs["davies_graph"],
            # "silhouette_graph": metric_graphs["silhouette_graph"],
            "cluster_metrics": results,
            "shap_results": shap_results,
            "anomaly_results": anomaly_results  # Include anomaly results in the return
        }

    except Exception as e:
        logging.error(f"Error during clustering analysis: {str(e)}")
        return {"error": str(e)}
