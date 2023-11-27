# Unsupervised Methods
##Overview

This project explores various clustering techniques and dimensionality reduction methods applied to a given dataset. The main script, main.py, provides a modularized approach to analyzing and visualizing data. The script includes functionalities for data preprocessing, distance matrix calculation, clustering, and embedding techniques.

##Requirements

Python 3.x
Virtual environment (venv)
Packages listed in requirements.txt

##Usage

To run the main script and generate clusters:

```
python main.py

```
Parameters
path: Path to the dataset file.
plotting: Boolean flag to enable/disable plotting (default is True).
metric: Distance metric for clustering (default is 'euclidean').
clusters: Number of clusters for certain methods (default is 3).
categorical: List of categorical variables for one-hot-encoding (default is None).


The script performs the following steps:

Data Preprocessing: Reads data from the specified path and performs normalization. If categorical variables are present, it uses one-hot-encoding.
Visualization: Plots the normalized data in a hypercube and generates a heatmap and surface plot of the distance matrix.
Clustering Techniques: Applies various clustering techniques (Mountain Clustering, Subtractive Clustering, K-Means, Fuzzy C-Means, Spectral Clustering) on different embeddings (Autoencoder, UMAP) and distance metrics.
Virtual Environment Setup: The script automatically creates a virtual environment, installs necessary dependencies from requirements.txt, and displays the location of the created environment.
Additional Information

For more details on the methods used and custom modules, refer to the respective files in the src directory.
