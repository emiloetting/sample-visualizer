import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QPushButton, QSlider, QLabel, QComboBox
)
from PySide6.QtCore import Qt
from vispy import scene
import vispy
import data_prep as dp
import dim_clustering as dc
import os

# Constants
FILE_PATH = os.path.join(dp.CSV_OUTPUT_FOLDER, dp.OUTPUT_CSV_NAME)
COLORMAP = 'viridis'

# Vispy-Widget, das als Stub für die 3D-Visualisierung dient
class VispyWidget(QWidget):
    def __init__(self, parent=None, dimensions=2):
        super().__init__(parent)
        #safe dimensions to plot
        self.dims_to_plot = dimensions

        # Create Vispy Canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='#0C0714', parent=self)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        
        # Canvas Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        #initialize view and scatter to be updated
        self.view = self.canvas.central_widget.add_view()
        self.scatter = scene.visuals.Markers()
        self.view.add(self.scatter)

    def set_camera(self, dimensions):
        """Adapts Plot-Camera to the given dimensions"""
        if dimensions == 2:
            self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        elif dimensions == 3:
            self.view.camera = scene.cameras.TurntableCamera(fov=45, elevation=30, azimuth=30)
        self.dims_to_plot = dimensions
        self.canvas.update()
    
    def update_visualization(self, new_data, labels, dimensions):
        # Convert new_data to numpy array, only use Positional Data
        if self.dims_to_plot != dimensions:
            self.set_camera(dimensions)  # Switch Camera if Dimensions change

        if self.dims_to_plot == 2:
            new_clustering = new_data.to_numpy()[:, :2]
        elif self.dims_to_plot == 3:
            new_clustering = new_data.to_numpy()[:, :3]

        if labels is None:
            raise ValueError('Labels of None Value not supported')
        if len(labels) != len(new_clustering):
            raise ValueError('Labels and data length do not match.')
        
        # Normalizing Labels to [0,1] -> for Color Mapping
        norm_labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-10)
        color_map = vispy.color.get_colormap(COLORMAP)
        plot_colors = color_map.map(norm_labels)#[:, :3]
        self.scatter.set_data(new_clustering, edge_color='black', face_color=plot_colors, size=5)
        self.canvas.update()

# Hauptfenster mit Steuerungselementen
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Clustering Visualisierung")
        
        # Zentrales Widget und Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Vispy-Widget hinzufügen
        self.vispy_widget = VispyWidget()
        main_layout.addWidget(self.vispy_widget, stretch=1)
        
        # Steuerungspanel für Parameter
        control_widget = QWidget()
        control_layout = QGridLayout(control_widget)
        
        # UMAP Min Distance Slider
        self.umap_min_distance_slider = QSlider(Qt.Horizontal)
        self.umap_min_distance_slider.setMinimum(0)
        self.umap_min_distance_slider.setMaximum(100)
        self.umap_min_distance_slider.setValue(10)  # Starting Value
        control_layout.addWidget(QLabel("UMAP Min Distance"), 0, 0)
        control_layout.addWidget(self.umap_min_distance_slider, 0, 1)
        
        # UMAP Neighbors Slider
        self.umap_neighbors_slider = QSlider(Qt.Horizontal)
        self.umap_neighbors_slider.setMinimum(1)
        self.umap_neighbors_slider.setMaximum(100)
        self.umap_neighbors_slider.setValue(15)  # Starting Value
        control_layout.addWidget(QLabel("UMAP Neighbors"), 1, 0)
        control_layout.addWidget(self.umap_neighbors_slider, 1, 1)
        
        # DBSCAN Epsilon Slider
        self.dbscan_epsilon_slider = QSlider(Qt.Horizontal)
        self.dbscan_epsilon_slider.setMinimum(1)
        self.dbscan_epsilon_slider.setMaximum(100)
        self.dbscan_epsilon_slider.setValue(5)  # Starting Value
        control_layout.addWidget(QLabel("DBSCAN Epsilon"), 2, 0)
        control_layout.addWidget(self.dbscan_epsilon_slider, 2, 1)
        
        # DBSCAN Min Samples Slider
        self.dbscan_min_samples_slider = QSlider(Qt.Horizontal)
        self.dbscan_min_samples_slider.setMinimum(1)
        self.dbscan_min_samples_slider.setMaximum(100)
        self.dbscan_min_samples_slider.setValue(30) # Starting Value
        control_layout.addWidget(QLabel("DBSCAN Min Samples"), 3, 0)
        control_layout.addWidget(self.dbscan_min_samples_slider, 3, 1)

        # Input-Box for Dimensions
        self.dimensions = QComboBox()
        self.dimensions.addItem("2")
        self.dimensions.addItem("3")
        self.dims_to_plot = int(self.dimensions.currentText())
        control_layout.addWidget(QLabel("Dimensions"), 4, 0)
        control_layout.addWidget(self.dimensions, 4, 1)
        
        # Input-Box for Scaler
        self.scaler = QComboBox()
        self.scaler.addItem("MinMax")
        self.scaler.addItem("Standard")
        control_layout.addWidget(QLabel("Scaler"), 5, 0)
        control_layout.addWidget(self.scaler, 5, 1)

        # Button zum Aktualisieren des Clustering
        self.update_button = QPushButton("Update Clustering")
        self.update_button.clicked.connect(self.update_clustering)
        control_layout.addWidget(self.update_button, 6, 0, 1, 2)
        main_layout.addWidget(control_widget)
    
    def update_clustering(self):
    # Werte aus der GUI holen
        dimensions = int(self.dimensions.currentText())
        umap_min_distance = self.umap_min_distance_slider.value() / 100.0
        umap_neighbors = self.umap_neighbors_slider.value()
        dbscan_epsilon = self.dbscan_epsilon_slider.value() / 10.0
        dbscan_min_samples = self.dbscan_min_samples_slider.value()
        scaler = self.scaler.currentText()

        print("Übergebene Parameter:\nDimensions:", dimensions,
            "\nUMAP Min Distance:", umap_min_distance,
            "\nUMAP Neighbors:", umap_neighbors,
            "\nDBSCAN Epsilon:", dbscan_epsilon,
            "\nDBSCAN Min Samples:", dbscan_min_samples,
            "\nScaler:", scaler)

        # Daten für Clustering abrufen
        new_data = dc.go(filepath=FILE_PATH, dimensions=dimensions,
                        min_dist=umap_min_distance, n_neighbors=umap_neighbors,
                        eps=dbscan_epsilon, min_samples=dbscan_min_samples, scaler=scaler)
        labels = np.array(new_data['label'])
        if len(labels) != len(new_data):
            raise ValueError('Labels and data length do not match.')

        # Direktes Update ohne das Widget neu zu erstellen
        self.vispy_widget.update_visualization(new_data=new_data, labels=labels, dimensions=dimensions)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())





