from basic_shape import BasicShape
from abc import ABC
from matplotlib.patches import Polygon as pltPolygon
import matplotlib.pyplot as plt
import numpy as np
import mpl_colors as mcolors


class Polygon(BasicShape, ABC):
    def __init__(self, line_color: mcolors.Color, fill_color: mcolors.Color):
        super().__init__(line_color, fill_color)
        self.points = np.array([])

    def draw(self, axis: plt.Axes):
        polygon = pltPolygon(self.points, edgecolor=self.line_color, facecolor=self.fill_color)
        axis.add_artist(polygon)

    def _shape_center(self):
        x = np.mean(self.points[:, 0])
        y = np.mean(self.points[:, 1])
        return x, y

    def rotate(self, degrees: float, x: float = None, y: float = None):
        if x is None or y is None:
            x, y = self._shape_center()

        temp_x = self.points[:, 0] - x
        temp_y = self.points[:, 1] - y

        radians = degrees * (np.pi / 180)
        cos = np.cos(radians)
        sin = np.sin(radians)

        self.points[:, 0] = temp_x * cos - temp_y * sin + x
        self.points[:, 1] = temp_x * sin + temp_y * cos + y

    def translate(self, x: float, y: float):
        self.points[:, 0] += x
        self.points[:, 1] += y

    def scale(self, factor: float):
        cx, cy = self._shape_center()

        temp_x = self.points[:, 0] - cx
        temp_y = self.points[:, 1] - cy

        self.points[:, 0] = temp_x * factor + cx * factor
        self.points[:, 1] = temp_y * factor + cy * factor
