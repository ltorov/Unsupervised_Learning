def main(path, plotting = True, metric = 'euclidean', clusters = 3, categorical = None):

    import umap

    # Import my own methods
    from distances import distance, sort_distances
    from plotting import immersive_scatter, scatter_clusters, scatter, heatmap, surface, scatter_to_surface
    from clusters import grid, boxes, box_clusters, neighbors, similarity_index, intra_cluster_indices, extra_cluster_indices
    from auxiliary import normalize, read, one_hot_encoding, Autoencoder, clusters_structure
    from main import main
    from mountain_clustering import MountainClustering
    from subtractive_clustering import SubtractiveClustering
    from kmeans_clustering import KMeansClustering
    from fuzzycmeans_clustering import FuzzyCMeansClustering
    from spectral_clustering import SpectralClustering

    # Read the data as a np.array from the path specified
    data = read(path)

    # If data contains categorical variables, use one-hot-encoding. Normalize data
    if categorical != None:
        categorical_data = one_hot_encoding(data, categorical)
        norm_data = normalize(categorical_data)
    else: norm_data = normalize(data)

    # Plot the normalized data in the hypercube [0,1]x[0,1]x[0,1] for all sets of dimensions.
    if plotting: scatter(norm_data)

    # Calculate the distance matrix from all points to all points and plot the heatmap.
    D = distance(norm_data, metric = metric)
    if plotting: 
        heatmap(D, "Heatmap of "+metric+" Distance Matrix (Points x Points)") 
        surface(D, "Surface plot of "+metric+" Distance Matrix (Points x Points)")

    # Sort the distance matrix using the boxes cluster criteria and plot the heatmap and surface map.
    D, _, _ = sort_distances(distance(norm_data, metric = metric), div=clusters)
    if plotting: 
        heatmap(D, "Heatmap of sorted "+metric+" Distance Matrix (Points x Points)")
        surface(D, "Surface plot of sorted "+metric+" Distance Matrix (Points x Points)")

    vertices = grid(norm_data.shape[1], 1/2)

    # Calculate the distance matrix from all vertices to all points and plot the heatmap.
    D = distance(norm_data, vertices, metric = metric)
    # if plotting: 
    #     heatmap(D, "Heatmap of "+metric+" Distance Matrix (Vertices x Points)") 
    #     surface(D, "Surface plot of "+metric+" Distance Matrix (Vertices x Points)")

    # Cluster with our artisinal methods
    Clusters = []
    Metrics = ['euclidean', 'manhattan', 'cosine', 'lp']
    for met in Metrics:

        D = distance(norm_data, metric = met)

        b = boxes(D, div = clusters)
        
        M = box_clusters(b, D)
        Clusters.append(M)

        labels = neighbors(D, clusters, 0.1)
        print(labels)

        Clusters.append(labels)

    # Embed data with autoencoder

    # Create the autoencoder for 2 dimensions
    autoencoder_2d = Autoencoder(encoded_layers = 2)
    autoencoder_2d.compile(optimizer='adam', loss='mse')
    autoencoder_2d = autoencoder_2d.predict(norm_data)

    # Create the autoencoder for 3 dimensions
    autoencoder_3d = Autoencoder(encoded_layers = 3)
    autoencoder_3d.compile(optimizer='adam', loss='mse')
    autoencoder_3d = autoencoder_3d.predict(norm_data)

    # Create the autoencoder for 3 dimensions
    autoencoder_5d = Autoencoder(encoded_layers = 5)
    autoencoder_5d.compile(optimizer='adam', loss='mse')
    autoencoder_5d = autoencoder_5d.predict(norm_data)

    # Embed data with umap
    mapper = umap.UMAP(min_dist=0.1, metric="euclidean", n_components =2).fit(norm_data)
    embedding_2d = mapper.embedding_
    mapper = umap.UMAP(min_dist=0.1, metric="euclidean", n_components =3).fit(norm_data)
    embedding_3d = mapper.embedding_

    umap_2d = normalize(embedding_2d)
    umap_3d = normalize(embedding_3d)

    experiment_data = [norm_data, autoencoder_2d, autoencoder_3d, autoencoder_5d, umap_2d, umap_3d]

    for experiment in experiment_data:
        if experiment.shape[1] == 2: vertices = grid(2, 1/11)
        elif experiment.shape[1] == 3: vertices = grid(3, 1/4)
        else: vertices = grid(experiment.shape[1], 1/2)

        model = MountainClustering()
        clusters = model.cluster(experiment, vertices, show = False) 
        Clusters.append(clusters_structure(clusters))

        model = SubtractiveClustering()
        clusters = model.cluster(experiment, show = False) 
        Clusters.append(clusters_structure(clusters))

        k = len(clusters_structure(clusters))

        model = KMeansClustering(k = k)
        clusters = model.cluster(experiment)
        Clusters.append(clusters_structure(clusters))

        model = FuzzyCMeansClustering(k = k)
        clusters = model.cluster(experiment)
        Clusters.append(clusters_structure(clusters))

        model = SpectralClustering(k = k)
        clusters = model.cluster(experiment)
        Clusters.append(clusters_structure(clusters))
        


    return Clusters

if __name__ == "__main__":
    # System imports
    from venv import create
    from os.path import join, expanduser, abspath
    from subprocess import run

    # Create virtual environment
    try:
        dir = join(expanduser("."), "venv")
        create(dir, with_pip=True)
        print("Virtual environment created on: ", dir)
    except Exception as e:
        raise Exception("Failed to create the virtual environment: " + str(e))

    # Install packages in 'requirements.txt'.
    try:
        run(["python3", "-m", "pip3", "install", "--upgrade", "pip3"])
        run(["bin/pip3", "install", "-r", abspath("requirements.txt")], cwd=dir)
    except:
        run(["python", "-m", "pip", "install", "--upgrade", "pip"])
        run(["bin/pip", "install", "-r", abspath("requirements.txt")], cwd=dir)
    finally:
        print("Completed installation of requirements.")

    import umap

    # Import my own methods
    from distances import distance, sort_distances
    from plotting import immersive_scatter, scatter_clusters, scatter, heatmap, surface, scatter_to_surface
    from clusters import grid, boxes, box_clusters, neighbors, similarity_index, intra_cluster_indices, extra_cluster_indices
    from auxiliary import normalize, read, one_hot_encoding, Autoencoder, clusters_structure
    from main import main
    from mountain_clustering import MountainClustering
    from subtractive_clustering import SubtractiveClustering
    from kmeans_clustering import KMeansClustering
    from fuzzycmeans_clustering import FuzzyCMeansClustering
    from spectral_clustering import SpectralClustering

    path = 'data/iris.csv'
    Clusters = main(path, plotting = True, metric = 'euclidean')
    print(Clusters)