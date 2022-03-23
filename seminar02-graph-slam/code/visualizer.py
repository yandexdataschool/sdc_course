import numpy as np

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import utils
import graph_elements as ge

class Visualizer(object):
    """Visualizes poses and features in global coordinate system"""
    MIN_SIZE = 10
    

    def __init__(self, margin=40.0, grid=False):
        self._plot_margin = margin
        self._grid = grid
        self._positions = None
        self._points = None
        self._lines = None


    def update_poses(self, poses):
        self._positions = poses[:, :2]
 

    def update_features(self, features):
        self._points = []
        self._lines = []
        for feature in features:
            if feature.ftype == ge.Feature.POINT:
                self._points.append(feature.visualization_data)
            elif feature.ftype == ge.Feature.LINE:
                self._lines.append(feature.visualization_data)
        if len(self._points) == 0:
            self._points = None
        else:
            self._points = np.array(self._points)
        if len(self._lines) == 0:
            self._lines = None
        else:
            self._lines = np.array(self._lines)
 

    def show(self, title, out_path):
        bounds = self._determine_bounds()
        bounds_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        if bounds_size[0] < 1E-5 or bounds_size[1] < 1E-5:
            return
        aspect = bounds_size[0] / bounds_size[1]
        if 1.0 < aspect: 
            plt.figure(figsize=(aspect * Visualizer.MIN_SIZE, Visualizer.MIN_SIZE))
        else:
            plt.figure(figsize=(Visualizer.MIN_SIZE, Visualizer.MIN_SIZE / aspect))
        plt.axis([bounds[0] - self._plot_margin,
                  bounds[2] + self._plot_margin,
                  bounds[1] - self._plot_margin,
                  bounds[3] + self._plot_margin])
        if self._grid:
            plt.grid()
        if title is not None:
            plt.title(title)
        if self._positions is not None:
            plt.plot(self._positions[:, 0], self._positions[:, 1], 'b^')
        if self._points is not None:
            plt.plot(self._points[:, 0], self._points[:, 1], 'r.')
        if self._lines is not None:
            for line in self._lines:
                plt.plot([line[0], line[2]], [line[1], line[3]], 'g')

        plt.savefig(out_path)
        plt.close()
        print('Figure {} was saved in {}'.format(title, out_path))


    def _determine_bounds(self):
        all_points = []
        if self._positions is not None:
            all_points.append(self._positions)
        if self._points is not None:
            all_points.append(self._points)
        if self._lines is not None:
            all_points.append(self._lines[:, :2])
            all_points.append(self._lines[:, 2:])
        all_points = np.concatenate(all_points, axis=0)
        return np.array(all_points.min(axis=0).tolist() + all_points.max(axis=0).tolist())
