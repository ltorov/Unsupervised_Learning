# Unsupervised Methods

**Project Description**: 
I just used a template for ChatGPT haven't really worked on it yet

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
  - [read(path)](#readpath)
  - [normalize(data)](#normalizedata)
  - [create_virtual_environment(venv_name)](#create_virtual_environmentvenv_name)
  - [activate_virtual_environment(venv_name)](#activate_virtual_environmentvenv_name)
  - [install_requirements()](#install_requirements)
  - [deactivate_virtual_environment()](#deactivate_virtual_environment)
  - [main(path, plotting=True, metric='euclidean')](#mainpath-plotting-truemetric-euclidean)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Before using this project, make sure you have the following prerequisites:

- Python (version 3.6 or higher)
- pip (Python package manager)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   ```

2. Navigate to the project directory:

   ```bash
   cd yourproject
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment (skip if not using a virtual environment):

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

5. Install the required packages:

   ```bash
   pip install -r requirements/requirements.txt
   ```

## Usage

To use the project's methods and utilities, you can import them into your Python code. Here's how to use some of the key methods:

```python
from yourproject import read, normalize, main

# Example usage of the 'read' method
data = read('data/yourdata.csv')

# Example usage of the 'normalize' method
normalized_data = normalize(data)

# Example usage of the 'main' method
path = 'data/iris.csv'
main(path, plotting=False)
```

For more details on each method, please refer to the [Methods](#methods) section below.

## Methods

### `read(path: str) -> numpy.ndarray`

Reads data from a file into a NumPy array.

- Parameters:
  - `path` (str): The path to the input file.

- Returns:
  - numpy.ndarray or None: The data as a NumPy array, or None if an error occurs.

### `normalize(data: numpy.ndarray) -> numpy.ndarray`

Normalizes a NumPy array by scaling values between 0 and 1.

- Parameters:
  - `data` (numpy.ndarray): The input data as a NumPy array.

- Returns:
  - numpy.ndarray: The normalized data.

### `create_virtual_environment(venv_name: str)`

Creates a virtual environment.

- Parameters:
  - `venv_name` (str): The name of the virtual environment.

### `activate_virtual_environment(venv_name: str)`

Activates a virtual environment.

- Parameters:
  - `venv_name` (str): The name of the virtual environment.

### `install_requirements()`

Installs required Python packages specified in the `requirements/requirements.txt` file.

### `deactivate_virtual_environment()`

Deactivates the virtual environment.

### `main(path: str, plotting: bool = True, metric: str = 'euclidean')`

The main function of the project, which performs various data analysis and visualization tasks. It reads data, normalizes it, calculates distances, and plots the results.

- Parameters:
  - `path` (str): The path to the data file.
  - `plotting` (bool, optional): Whether to enable plotting (default is True).
  - `metric` (str, optional): The distance metric to use (default is 'euclidean').

## Examples

You can find examples of how to use the methods in the code and comments within the project files. For detailed examples, please refer to the code in the project directory.

## Contributing

If you would like to contribute to this project, please follow our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
