def main(path, plotting = True, metric = 'euclidean', clusters = 3, categorical = None):

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

    # Import my own methods
    from distances import distance, sort_distances
    from plotting import scatter, heatmap, surface
    from clusters import grid, boxes, box_clusters, neighbors
    from auxiliary import normalize, read, one_hot_encoding

    path = 'data/iris.csv'
    Clusters = main(path, plotting = True, metric = 'euclidean')
    print(Clusters)