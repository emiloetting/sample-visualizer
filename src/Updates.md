# Updates and Changes

## Files and Folders
- `src/Updates.md` - This file
- removed notebooks
- split pipeline into separate files:
    - data_prep.py
    - dim_clustering.py
    - visual.py

### data_prep.py
Create all of the new required paths and directories for the pipeline. This includes the following:
- folder: cwd/audio_data
- folder: cwd/audio_data/all -> includes all audio files used. 
- folder: cwd/audio_data/csv_to_cluster -> for CSV with audio features
- folder: cwd/audio_data/loops -> for audio files with "loop" in the name, later to train classifier
- fodler: cwd/audio_data/to_label -> for audio files to label, also for classifier

Constants used over the pipeline (e.g. silence threshold, sample root folder, .csv - output name, etc.) are defined here.
All the for feature extraction required functions are defined and used to create a .csv file with all the features extracted from the audio files.
Currently, the features extracted are:
    - path
    - filename
    - sample rate
    - duration 
    - rms (mean and std)
    - spectral flatness (mean and std)
    - spectral centroid (mean and std)
    - spectral bandwidth (mean and std)
    - zero crossing rate (mean and std)

Possible other features to be included later:
    - Decay Time for One-Shots
    - Attack Time for One-Shots

## dim_clustering.py
This file contains the functions to cluster the features extracted from the audio files in data_prep.py. 
The first step is scaling the data. The user can choose between MinMaxScaler or StandardScaler via the GUI.
The second step is to reduce the dimensionality of the data using UMAP. Users have the choice to bring down the data to 2 or 3 dimensions. Depending on the number of dimensions, a 3D or 2D plot is later created.
The third step is to cluster the data using DBSCAN. The user can choose the epsilon and minimum samples for the clustering. The clusters are then plotted in the 2D or 3D plot. The color code is yet to be implemented.
After clustering the data, a Pandas DataFrame is created with the following columns:
    - path
    - filename
    - cluster
    - x
    - y
    - z (if 3D)

## visual.py
This file contains the functions to visualize the audio files. The user can choose between a 2D or 3D plot. The color code is yet to be implemented.
The plot itself is created using VisPy. This plot than is implemented as a widgetin the acutual GUI using PySide6. 
This ensured an interactive plot in the GUI to better understand the transformed data.
Over here, the user can currently select the following:
    - 2D or 3D plot
    - UMAP Hyperparameters: n_neighbors, min_dist
    - DBSCAN Hyperparameters: epsilon, min_samples
    - Scaling: MinMaxScaler or StandardScaler

Planned to Add:
    - Selector for only certain sounds (one-shots, loops, etc.) -> classfier needed in advance
    - Shell integration for drag and drop of audio files
    - Sample wave form visualization
    - Sample name visualization
    - Update for color code in plots







