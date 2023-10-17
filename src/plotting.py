import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
