import numpy as np
import pandas as pd
import json
from pandarallel import pandarallel
import holoviews as hv
from holoviews import opts
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

# Load necessary extensions and libraries
hv.extension('bokeh')

# Load the dataset
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    return data

# Clean the data
def clean_data(data):
    # Perform data cleaning operations here
    return cleaned_data

# Extract selected columns
def extract_selected_columns(data):
    certain_cols = ["title", "rating", "calories", ...]  # List the selected columns
    data_selected = data[certain_cols]
    return data_selected

# Get recipe information
def get_recipe_info(recipe_title, recipe_full):
    # Implement recipe information extraction here
    return {"len_directions": len_directions, "len_ingredients": len_ingredients}

# Exploratory Data Analysis (EDA) functions

def perform_eda(data_selected):
    # Perform EDA operations here
    return eda_results

# Clustering functions

def perform_kmeans_clustering(data_selected, n_clusters):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
    clusters = kmeans.fit_predict(data_selected)
    return clusters

def perform_gaussian_mixture_clustering(data_selected, n_components):
    # Perform Gaussian Mixture clustering
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    clusters = gmm.fit_predict(data_selected)
    return clusters

def perform_birch_clustering(data_selected, n_clusters):
    # Perform Birch clustering
    birch = Birch(n_clusters=n_clusters)
    clusters = birch.fit_predict(data_selected)
    return clusters

# Main function
def main():
    # Load the data
    csv_path = '../Data/Raw/epi_r.csv'
    data = load_data(csv_path)

    # Clean the data
    cleaned_data = clean_data(data)

    # Extract selected columns
    data_selected = extract_selected_columns(cleaned_data)

    # Load recipe information
    with open("../Data/Raw/full_format_recipes.json", "r") as f:
        recipe_full = json.load(f)

    # Get additional recipe information
    data_selected["len_directions"], data_selected["len_ingredients"] = zip(
        *data_selected["title"].apply(lambda x: get_recipe_info(x, recipe_full))
    )

    # Perform EDA
    eda_results = perform_eda(data_selected)

    # Perform clustering
    n_clusters = 16
    kmeans_clusters = perform_kmeans_clustering(data_selected, n_clusters)
    gmm_clusters = perform_gaussian_mixture_clustering(data_selected, n_clusters)
    birch_clusters = perform_birch_clustering(data_selected, n_clusters)

    # Add cluster labels to the DataFrame
    data_selected["kmeans_cluster"] = kmeans_clusters
    data_selected["gmm_cluster"] = gmm_clusters
    data_selected["birch_cluster"] = birch_clusters

    # Save the results to CSV
    data_selected.to_csv("clustered_recipes.csv", index=False)

if __name__ == "__main__":
    main()
