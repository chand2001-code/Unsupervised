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
from typing import Optional
import uvicorn


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



class ClusteringRequest(BaseModel):
    algorithm: str

class FinalClusteringRequest(BaseModel):
    algorithm: str
    n_clusters: int

def read_and_preprocess_csv(file: UploadFile) -> pd.DataFrame:
    try:
        df = pd.read_csv(file.file)
        # Handling NaN values
        if df.isnull().any().any():
            # Option 1: Fill NaN with the mean of each column (numeric columns)
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
            # Option 2: Fill NaN with the mode of each column (categorical columns)
            for col in df.select_dtypes(include=['object', 'category']).columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or preprocess CSV file: {str(e)}")



ALGORITHMS = [
    "K-Means clustering",
    "K-Modes clustering",
    "K-Prototypes clustering",
    "Hierarchical clustering",
    "DBSCAN clustering",
    "BIRCH clustering",
    "Spectral clustering",
    "Isolation Forest",
    "Local Outlier Factor",
    "Minimum Covariance Determinant"
]



@app.post("/cluster_analysis")
async def cluster_analysis(
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    n_clusters: Optional[int] = Form(None)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail="Invalid algorithm selected")

    try:
        df = read_and_preprocess_csv(file)

        if df.isnull().any().any():
            raise HTTPException(status_code=400, detail="Data still contains NaN values after preprocessing")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) == 0 and algorithm not in ["K-Modes clustering"]:
            raise HTTPException(status_code=400, detail="No numeric columns found in the CSV file")
        
        if len(numeric_cols) > 0:
            X = df[numeric_cols]
            if X.shape[1] == 1:  # If only one column, reshape it
                X = X.values.reshape(-1, 1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = np.array([]).reshape(-1, 1)  # Empty array reshaped as 2D

        # X = df[numeric_cols]
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        # Validate n_clusters for algorithms that require it
        if algorithm in ["K-Means clustering", "K-Prototypes clustering", "K-Modes clustering", "Spectral clustering"]:
            if n_clusters is None or n_clusters <= 0:
                raise HTTPException(status_code=400, detail="Number of clusters must be a positive integer")
            cluster_counts = list(range(1, n_clusters + 1))  # Start from 1 to n_clusters
        else:
            cluster_counts = list(range(2, min(12, len(X) - 1) + 1))  # For DBSCAN and BIRCH, no n_clusters needed

        metrics_results = {
            "elbow_data": [],
            "silhouette_data": [],
            "calinski_data": [],
            "davies_data": [],
            "max_clusters": min(12, len(X) - 1),
            "algorithm": algorithm,
            "n_clusters": n_clusters
        }

        # Collect metrics for each cluster count
        for k in cluster_counts:
            model = None
            cluster_labels = None
            if algorithm == "K-Means clustering":
                model = KMeans(n_clusters=k, n_init=10, random_state=42)
                cluster_labels = model.fit_predict(X_scaled)
                metrics_results["elbow_data"].append(model.inertia_)
            elif algorithm == "K-Modes clustering":
                if len(categorical_cols) > 0:
                    X_cat = df[categorical_cols].apply(LabelEncoder().fit_transform)
                    if X_cat.shape[1] == 1:  # If only one column, reshape it
                        X_cat = X_cat.values.reshape(-1, 1)
                    model = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
                    cluster_labels = model.fit_predict(X_cat)
                    metrics_results["elbow_data"].append(model.cost_)
                else:
                    raise HTTPException(status_code=400, detail="No categorical columns found for K-Modes clustering")
            elif algorithm == "K-Prototypes clustering":
                categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
                X_mixed = df.values
                model = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=0)
                cluster_labels = model.fit_predict(X_mixed, categorical=categorical_indices)
                metrics_results["elbow_data"].append(model.cost_)
            elif algorithm in ["Hierarchical clustering", "BIRCH clustering", "Spectral clustering"]:
                if algorithm == "Hierarchical clustering":
                    model = AgglomerativeClustering(n_clusters=k)
                elif algorithm == "BIRCH clustering":
                    model = Birch(n_clusters=k)
                else:  # Spectral clustering
                    model = SpectralClustering(n_clusters=k, n_init=10, random_state=42)
                cluster_labels = model.fit_predict(X_scaled)
                metrics_results["elbow_data"].append(None)

            elif algorithm == "DBSCAN clustering":
                if k == 2:  # DBSCAN typically only has one relevant case for cluster count
                    model = DBSCAN(eps=0.5, min_samples=5)
                    cluster_labels = model.fit_predict(X_scaled)
                    valid_labels = cluster_labels != -1
                    if np.any(valid_labels):
                        X_valid = X_scaled[valid_labels]
                        cluster_labels_valid = cluster_labels[valid_labels]

                        if len(set(cluster_labels_valid)) > 1:
                            try:
                                metrics_results["silhouette_data"].append(silhouette_score(X_valid, cluster_labels_valid))
                                metrics_results["calinski_data"].append(calinski_harabasz_score(X_valid, cluster_labels_valid))
                                metrics_results["davies_data"].append(davies_bouldin_score(X_valid, cluster_labels_valid))
                            except ValueError as e:
                                logging.error(f"Error calculating metrics: {e}")
                                metrics_results["silhouette_data"].append(-1)  # Default value
                                metrics_results["calinski_data"].append(-1)   # Default value
                                metrics_results["davies_data"].append(-1)     # Default value
                        else:
                            metrics_results["silhouette_data"].append(0)  # Default value
                            metrics_results["calinski_data"].append(0)    # Default value
                            metrics_results["davies_data"].append(0)      # Default value
                    else:
                        metrics_results["silhouette_data"].append(0)  # Default value
                        metrics_results["calinski_data"].append(0)    # Default value
                        metrics_results["davies_data"].append(0)      # Default value
                    
                    metrics_results["elbow_data"].append(None)
                else:
                    metrics_results["silhouette_data"].append(0)
                    metrics_results["calinski_data"].append(0)
                    metrics_results["davies_data"].append(0)
                    metrics_results["elbow_data"].append(None)

            # Calculate metrics for the current cluster labels
            if cluster_labels is not None:
                unique_clusters = len(set(cluster_labels))
                if unique_clusters > 1:
                    try:
                        metrics_results["silhouette_data"].append(silhouette_score(X_scaled, cluster_labels))
                        metrics_results["calinski_data"].append(calinski_harabasz_score(X_scaled, cluster_labels))
                        metrics_results["davies_data"].append(davies_bouldin_score(X_scaled, cluster_labels))
                    except ValueError as e:
                        logging.error(f"Error calculating metrics: {e}")
                        metrics_results["silhouette_data"].append(-1)  # Default value
                        metrics_results["calinski_data"].append(-1)   # Default value
                        metrics_results["davies_data"].append(-1)     # Default value
                else:
                    metrics_results["silhouette_data"].append(0)  # Default value
                    metrics_results["calinski_data"].append(0)    # Default value
                    metrics_results["davies_data"].append(0)      # Default value

        # Ensure metrics lists are trimmed to exactly n_clusters
        if n_clusters:
            metrics_results["elbow_data"] = metrics_results["elbow_data"][:n_clusters]
            metrics_results["silhouette_data"] = metrics_results["silhouette_data"][:n_clusters]
            metrics_results["calinski_data"] = metrics_results["calinski_data"][:n_clusters]
            metrics_results["davies_data"] = metrics_results["davies_data"][:n_clusters]

        # Determine if the results are valid
        flag = 1 if any(metrics_results[key] for key in ["elbow_data", "silhouette_data", "calinski_data", "davies_data"]) else 0
        metrics_results["flag"] = flag

        return JSONResponse(metrics_results)

    except Exception as e:
        logging.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/final_clustering")
async def final_clustering(
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    n_clusters: Optional[int] = Form(None)  # Make n_clusters optional
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail="Invalid algorithm selected")

    # Validate n_clusters for algorithms that require it
    if algorithm in ["K-Means clustering", "K-Prototypes clustering", "K-Modes clustering", "Spectral clustering"]:
        if n_clusters is None or n_clusters <= 0:
            raise HTTPException(status_code=400, detail="Number of clusters must be a positive integer")
    elif algorithm in ["Hierarchical clustering", "DBSCAN clustering", "BIRCH clustering"]:
        if n_clusters is not None:
            raise HTTPException(status_code=400, detail="Number of clusters should not be provided for this algorithm")

    try:
        df = read_and_preprocess_csv(file)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 0 and algorithm not in ["K-Modes clustering"]:
            raise HTTPException(status_code=400, detail="No numeric columns found in the CSV file")
        
        X = df[numeric_cols]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = None
        cluster_labels = None
        silhouette_vals = np.zeros(len(X_scaled))
        avg_silhouette_score = 0
        dendrogram_img = None  # Initialize the variable

        if algorithm == "K-Means clustering":
            model = KMeans(n_clusters=n_clusters, n_init=10)
            cluster_labels = model.fit_predict(X_scaled)
        elif algorithm == "K-Modes clustering":
            if len(categorical_cols) == 0:
                raise HTTPException(status_code=400, detail="No categorical columns found for K-Modes clustering")
            X_cat = df[categorical_cols]
            model = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            cluster_labels = model.fit_predict(X_cat)
        elif algorithm == "K-Prototypes clustering":
            if len(categorical_cols) == 0:
                raise HTTPException(status_code=400, detail="No categorical columns found for K-Prototypes clustering")
            categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
            X_mixed = df.values
            model = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            cluster_labels = model.fit_predict(X_mixed, categorical=categorical_indices)
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
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            dendrogram_img = img_stream.read()

            # Fit the model to get cluster labels
            model = AgglomerativeClustering()  # No n_clusters specified
            cluster_labels = model.fit_predict(X_scaled)  # Clustering for label purposes
            
            # Calculate silhouette score for Hierarchical clustering
            if len(set(cluster_labels)) > 1:
                silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
                avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
            else:
                avg_silhouette_score = -1  # Indicate insufficient clusters for silhouette score

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
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            dendrogram_img = img_stream.read()

            # Calculate silhouette score if more than one cluster is found
            if len(set(cluster_labels)) > 1:
                silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
                avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
            else:
                avg_silhouette_score = -1  # Indicate insufficient clusters for silhouette score

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
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            dendrogram_img = img_stream.read()

            # Calculate silhouette score if more than one cluster is found
            if len(set(cluster_labels)) > 1:
                silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
                avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
            else:
                avg_silhouette_score = -1  # Indicate insufficient clusters for silhouette score

        elif algorithm == "Spectral clustering":
            model = SpectralClustering(n_clusters=n_clusters, n_init=10)
            cluster_labels = model.fit_predict(X_scaled)

        if model is None:
            raise HTTPException(status_code=400, detail="Failed to create model")

        # Prepare response
        response = {
            "algorithm": algorithm,
            "cluster_labels": cluster_labels.tolist(),
        }

        if algorithm in ["Hierarchical clustering", "DBSCAN clustering", "BIRCH clustering"]:
            if dendrogram_img is None:
                raise HTTPException(status_code=500, detail="Dendrogram image generation failed.")
            response["dendrogram_img"] = "data:image/png;base64," + base64.b64encode(dendrogram_img).decode('utf-8')

        # Add silhouette score to response for all algorithms that support it
        if avg_silhouette_score != -1:
            response["avg_silhouette_score"] = avg_silhouette_score
            response["silhouette_vals"] = silhouette_vals.tolist()
        else:
            response["avg_silhouette_score"] = "Insufficient clusters for silhouette score"
            response["silhouette_vals"] = "Insufficient clusters for silhouette score"

        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model fitting failed: {str(e)}")

 
@app.post("/shap_beeswarm")
async def shap_beeswarm(
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    n_clusters: Optional[int] = Form(None)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail="Invalid algorithm selected")

    try:
        df = read_and_preprocess_csv(file)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) == 0:
            raise HTTPException(status_code=400, detail="No numeric columns found in the CSV file")

        X_numeric = df[numeric_cols]
        X_categorical = df[categorical_cols]

        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)

        model = None
        cluster_labels = None

        surrogate_model = RandomForestClassifier(n_estimators=50, random_state=42)

        if algorithm == "Local Outlier Factor":
            model = LocalOutlierFactor(n_neighbors=2, contamination=0.1, novelty=False)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            cluster_labels = np.where(cluster_labels == -1, 1, 0)
        elif algorithm == "K-Modes clustering":
            if len(categorical_cols) == 0:
                raise HTTPException(status_code=400, detail="K-Modes clustering requires categorical data")
            X_cat_encoded = X_categorical.apply(LabelEncoder().fit_transform)
            model = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            cluster_labels = model.fit_predict(X_cat_encoded)
            X_scaled = X_cat_encoded
        elif algorithm == "K-Prototypes clustering":
            if len(categorical_cols) == 0:
                raise HTTPException(status_code=400, detail="K-Prototypes clustering requires both numeric and categorical data")
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
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "DBSCAN clustering":
            model = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "BIRCH clustering":
            model = Birch(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "Spectral clustering":
            model = SpectralClustering(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "Isolation Forest":
            model = IsolationForest(contamination=0.1, random_state=42)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "Minimum Covariance Determinant":
            model = EllipticEnvelope(contamination=0.1, random_state=42)
            cluster_labels = model.fit_predict(X_numeric_scaled)
            X_scaled = X_numeric_scaled
        elif algorithm == "Minimum Covariance Determinant":
                model = EllipticEnvelope(contamination=0.1, random_state=42)
                X_scaled = X_numeric_scaled
                cluster_labels = model.fit_predict(X_scaled)
        else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
 
        # Add other algorithms here similarly

        if cluster_labels is None:
            raise HTTPException(status_code=400, detail="Cluster labels are not defined.")

        logging.info(f"Cluster labels: {cluster_labels}")

        surrogate_model.fit(X_numeric_scaled, cluster_labels)

        def surrogate_predict(X):
            return surrogate_model.predict(X)

        explainer = shap.KernelExplainer(surrogate_predict, X_numeric_scaled)
        shap_values = explainer.shap_values(X_numeric_scaled)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_numeric_scaled, plot_type="dot",
                          feature_names=list(numeric_cols), show=False, color_bar=True, max_display=10)
        plt.title(f"SHAP Summary Plot for {algorithm}")
        plt.tight_layout()

        img_stream = io.BytesIO()
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

        return JSONResponse(response)

    except Exception as e:
        logging.error(f"Error in SHAP beeswarm calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SHAP beeswarm calculation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)