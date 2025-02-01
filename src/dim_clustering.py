import os
import pandas as pd
import timeit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import umap.umap_ as umap  
from sklearn.cluster import DBSCAN



def load_data(filepath: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the data from the CSV or XLSX file and prepare it for clustering.
    --------------------------------------------------------------------------
    Parameters: 
    - filepath: str, path to the CSV file to load.
    --------------------------------------------------------------------------
    Returns:
    - data: pd.DataFrame, the loaded data.
    - filenames: pd.Series, the filenames in the data.
    - data_to_reduce: pd.DataFrame, the data without the filenames and paths.
    - filepath: pd.Series, the input filepath
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Could not find file {filepath}.')
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            data = pd.read_excel(filepath)
        filenames = pd.Series(data['filename'])
        paths = pd.Series(data['path'])
        data_to_reduce = data.drop(['filename', 'path'], axis=1)
    except Exception as e:
        raise ValueError(f'Could not load data from {filepath}: {e}')
    
    return data, filenames, data_to_reduce, paths


def scale_data(data_to_scale: pd.DataFrame = None, scaler: str = 'MinMax') -> pd.DataFrame:  
    '''Scale data using MinMaxScaler and return it as a pandas DataFrame.
    --------------------------------------------------------------------------
    Parameters:
    - data_to_scale: pd.DataFrame with only the features to scale. Can be extracted from the original DataFrame using load_data().
    '''

    if data_to_scale is None:
        raise ValueError('No DataFrame to scale data from.')
    if scaler.lower().strip() == 'minmax':
        scaler_object = MinMaxScaler()
    elif scaler.lower().strip() == 'standard':
        scaler_object = StandardScaler()
    else:
        raise ValueError(f'Could not recognize scaler {scaler}. Please use MinMax or Standard.')
    scaled_data = scaler_object.fit_transform(data_to_scale)
    return scaled_data


def compute_clustering(dimensions: int = 2, data_to_reduce = None, min_dist: float=0.1, n_neighbors: int=100, eps: float=0.5, min_samples: int=5):
    """Compute clustering using UMAP and DBSCAN.
    --------------------------------------------------------------------------
    Parameters:
    - dimension: int, the number of dimensions for the UMAP embedding (2 or 3).
    - data_to_reduce: pd.DataFrame, the data to reduce and cluster.
    - min_dist: float, the minimum distance between points in the UMAP embedding.
    - n_neighbors: int, the number of neighbors to consider in the UMAP embedding.
    - eps: float, the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: int, the number of samples in a neighborhood for a point to be considered as a core point.
    --------------------------------------------------------------------------
    Returns:
    - clustered_data: pd.DataFrame, the data with the UMAP coordinates.
    - labels_series: pd.Series, the labels assigned by DBSCAN.
    """
    if data_to_reduce is None:
        raise ValueError('No data to reduce and cluster.')
    if dimensions not in [2, 3]:
        raise ValueError('Dimensions must be 2 or 3.')
    
    print('\nTransforming Data to lower Dimensions...')
    start = timeit.default_timer()
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=dimensions)
    transformed_data = umap_model.fit_transform(data_to_reduce)
    print(f'Transformed Data into {dimensions} dimensions in', round(timeit.default_timer() - start, 2), 'seconds\n')
    
    print('\nClustering Data...')
    start = timeit.default_timer()
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(transformed_data)
    labels = clustering.labels_
    print('Data clustered in', round(timeit.default_timer() - start, 2), 'seconds\n')

    if dimensions == 2:
        clustered_data = pd.DataFrame(transformed_data, columns=['x', 'y'])
    elif dimensions == 3:
        clustered_data = pd.DataFrame(transformed_data, columns=['x', 'y', 'z'])
        print('3D Array shape:', clustered_data.shape)

    labels_series = pd.Series(labels, name='label')

    return clustered_data, labels_series


def create_clustered_data(data: pd.DataFrame, labels: pd.Series, filenames: pd.Series, filepaths: pd.Series) -> pd.DataFrame:
    """Combine the data, labels, filenames and filepaths into a single DataFrame for plotting.
    --------------------------------------------------------------------------
    Parameters:
    - data: pd.DataFrame, the original data.
    - labels: pd.Series, the labels for the data.
    - filenames: pd.Series, the filenames for the data.
    - filepaths: pd.Series, the filepaths for the data.
    --------------------------------------------------------------------------
    Returns:
    - data_plotready: pd.DataFrame, the combined data.
    """

    data_plotready = pd.concat([data, labels, filenames, filepaths], axis=1)
    return data_plotready

def go(filepath: str, dimensions: int = 2, min_dist: float=0.1, n_neighbors: int=100, eps: float=0.5, min_samples: int=5, scaler: str='MinMax') -> pd.DataFrame:
    start = timeit.default_timer()
    print(f'Loading data from {filepath}...')
    _, filenames, data_to_reduce, paths = load_data(filepath)
    print('Data loaded successfully in', round(timeit.default_timer() - start, 2), 'seconds\n')

    start = timeit.default_timer()
    print('Scaling data...')
    scaled_data = scale_data(data_to_reduce, scaler)
    print('Data scaled successfully in', round(timeit.default_timer() - start, 2), 'seconds\n')

    start = timeit.default_timer()
    clustered_data, labels = compute_clustering(dimensions=dimensions, data_to_reduce=scaled_data, min_dist=min_dist, n_neighbors=n_neighbors, eps=eps, min_samples=min_samples)
    
    print('Finalizing DataFrame...')
    data_plotready = create_clustered_data(clustered_data, labels, filenames, paths)
    print('DataFrame ready for plotting\n\n')
    return data_plotready
