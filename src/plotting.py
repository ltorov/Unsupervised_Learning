import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.interpolate import griddata
from babyplots import Babyplot

def immersive_scatter(data, clusters, axis= [0,1,2], colors = "YlGnBu", xlabel = "X", ylabel = "Y", zlabel = "Z"):
    points = [clusters[cluster]['points'] for cluster in clusters.keys()]
    targets = [i for n in range(len(data)) for i in range(len(points)) if n in points[i]]
    data = data[:, axis]
    
    # create the babyplots visualization
    bp = Babyplot()
    bp.add_plot(data, "shapeCloud", "categories", targets, {"shape": "sphere",
                                                                    "colorScale": colors,
                                                                    "showAxes": [True, True, True],
                                                                    "axisLabels": [xlabel, ylabel, zlabel]})
    # show the visualization
    return bp

# Function to save the plot to the "results" folder
def save_plot(plt, title, folder="results", file_format="png"):
    # Create the "results" folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate the filename from the title
    filename = os.path.join(folder, title.replace(" ", "_") + "." + file_format)

    # Save the plot with the generated filename
    plt.savefig(filename)

def scatter(data, title = "Scatter plot for data"):
    # Create a 3D scatter plot for each combination of dimensions
    fig = plt.figure(figsize=(12, 8))

    # Plot 1: Dimensions 0, 1, and 2
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax1.set_xlabel('Dimension 0')
    ax1.set_ylabel('Dimension 1')
    ax1.set_zlabel('Dimension 2')
    ax1.set_title('3D Scatter Plot: Dimensions 0, 1, and 2')

    # Plot 2: Dimensions 0, 1, and 3
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(data[:, 0], data[:, 1], data[:, 3])
    ax2.set_xlabel('Dimension 0')
    ax2.set_ylabel('Dimension 1')
    ax2.set_zlabel('Dimension 3')
    ax2.set_title('3D Scatter Plot: Dimensions 0, 1, and 3')

    # Plot 3: Dimensions 0, 2, and 3
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(data[:, 0], data[:, 2], data[:, 3])
    ax3.set_xlabel('Dimension 0')
    ax3.set_ylabel('Dimension 2')
    ax3.set_zlabel('Dimension 3')
    ax3.set_title('3D Scatter Plot: Dimensions 0, 2, and 3')

    # Plot 4: Dimensions 1, 2, and 3
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(data[:, 1], data[:, 2], data[:, 3])
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.set_zlabel('Dimension 3')
    ax4.set_title('3D Scatter Plot: Dimensions 1, 2, and 3')

    # Add title
    plt.title(title)

    plt.tight_layout()
    
    # Save plot to results folder
    save_plot(plt, title)

def heatmap(D, title = "Heatmap of Distance Matrix", sorted = False):
    if sorted:
        # Flatten the matrix into a 1D array, preserving the original indices
        flattened_matrix = D.flatten()
        indices = np.arange(len(flattened_matrix))

        # Sort the flattened matrix and corresponding indices
        sorted_indices = np.argsort(flattened_matrix)
        sorted_matrix = flattened_matrix[sorted_indices]
        sorted_original_indices = indices[sorted_indices]

        # Reshape the sorted 1D array back into a 2D matrix
        D = sorted_matrix.reshape(D.shape)
    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(D, cmap="YlGnBu")

    # Add labels and title
    plt.title(title)
    plt.xlabel("D")
    plt.ylabel("D")

    # Save plot to results folder
    save_plot(plt, title)

def surface(D, title= 'Surface plot of Distance Matrix'):
    # Create x, y coordinates based on the shape of D
    x = np.arange(D.shape[0])
    y = np.arange(D.shape[1])
    x, y = np.meshgrid(x, y)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the matrix D for the z-values
    z = D.flatten()

    # Plot the 3D surface
    surf = ax.plot_surface(x, y, D, cmap="YlGnBu")

    # Set labels for the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Add title
    plt.title(title)

    # Save plot to results folder
    save_plot(plt, title)




def scatter_clusters(data, clusters, title = "Clusters", palette="YlGnBu", x_label='X', y_label='Y',
                             z_label='Z', axis = [0,1,2], return_figure=False):
    """
    Create a 3D scatter plot of data points with different clusters using Matplotlib.

    Parameters:
        data (numpy.ndarray): Data points with three columns (x, y, z).
        clusters (dict): A dictionary where keys are cluster labels and values are lists of indices in each cluster.
        palette (str): Color palette for clusters.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        z_label (str): Label for the z-axis.
        figsize (tuple): Figure size (width, height) for the plot.
        return_figure (bool): Whether to return the Matplotlib figure.

    Returns:
        matplotlib.figure.Figure (optional): The Matplotlib figure if return_figure is True, else None.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    figsize=(10, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette(palette, len(clusters))
    points = [clusters[cluster]['points'] for cluster in clusters.keys()]

    for id_color, point in enumerate(points):

        ax.scatter(data[point][:, axis[0]], data[point][:, axis[1]], data[point][:, axis[2]], color=colors[id_color], marker='o', label=id_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if return_figure:
        return fig

    plt.title(title)
    plt.legend()
    plt.show()

    # Restore warnings to their default behavior
    warnings.resetwarnings()



def scatter_to_surface(x, y, z, title = "Surface plot", xlabel = 'X Axis',  ylabel = 'Y Axis', zlabel = 'Z Axis', colors = "YlGnBu"):
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the grid for interpolation
    xi, yi = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

    # Interpolate Z-values to create a surface
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Create the surface plot
    ax.plot_surface(xi, yi, zi, cmap=colors)

    # Set labels for the axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Set title
    plt.title(title)

    # Show the plot
    plt.show()