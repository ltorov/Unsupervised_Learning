import numpy as np
import pandas as pd
import keras

def read(path: str) -> np.ndarray:
    """
    Read data from a file into a NumPy array.

    Parameters:
        path (str): The path to the input file.

    Returns:
        numpy.ndarray or None: The data as a NumPy array, or None if an error occurs.
    """
    try:
        # Determine the file extension
        file_extension = path.split('.')[-1].lower()

        if file_extension == 'csv':
            data = pd.read_csv(path).to_numpy()
        elif file_extension in ('txt', 'json'):
            data = pd.read_json(path).to_numpy()
        elif file_extension in ('xlsx', 'xls'):
            data = pd.read_excel(path, engine='openpyxl').to_numpy()
        else:
            print(f"Unsupported file format: {file_extension}. Supported formats are: csv, txt, json, xlsx, xls")
            return None
        return data

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize a NumPy array by scaling values between 0 and 1.

    Parameters:
        data (numpy.ndarray): The input data as a NumPy array.

    Returns:
        numpy.ndarray: The normalized data.
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)



def one_hot_encoding(data: np.ndarray, categorical: list) -> np.ndarray:
    """
    Use hot encoding for categorical columns in our data.

    Parameters:
        data (numpy.ndarray): The input data as a NumPy array.
        categorical (list): The columns that we will one-hot-encode.

    Returns:
        numpy.ndarray: The one-hot-encoded data.
    """
    # Create an array 'non_categorical' with indices for non-categorical columns
    non_categorical = np.arange(data.shape[1])

    # Filter out elements in 'non_categorical' which are in 'categorical'
    non_categorical = non_categorical[~np.isin(non_categorical, categorical)]

    # Initialize 'data_categorical' with the non-categorical columns from 'data'
    data_categorical = data[:, list(non_categorical)]

    # Loop through each categorical column
    for i in range(len(categorical)):
        col = data[:, categorical[i]]

        # Find and loop throught unique classes in the column
        classes = np.unique(col)
        for c in range(len(classes)):
            # Create and concatenate binary column for the current class
            class_c = np.array([[1 if j == classes[c] else 0 for j in col]]).T
            data_categorical = np.concatenate((data_categorical, class_c), axis=1)

    return data_categorical.astype(float)

class Autoencoder(keras.Model):
    def __init__(self, input_shape=(4,), encoded_layers = 3):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Dense(encoded_layers, activation='relu', input_shape=input_shape),
        ])

        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.Dense(input_shape[0], activation='linear'),
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        return encoded
    
def clusters_structure(clusters):
    # Convert the dictionary to an array of arrays
    result = []
    for key, value in clusters.items():
        cluster_array = np.zeros(150)  # Assuming there are 150 points in total
        cluster_array[value['points']] = 1
        result.append(cluster_array)

    return np.array(result)