import numpy as np
from basic_shape import BasicShape
import matplotlib.pyplot as plt
import mpl_colors as mcolors


class Point(BasicShape):
    def __init__(self, x: float, y: float, line_color: mcolors.Color, fill_color: mcolors.Color):
        super().__init__(line_color, fill_color)
        self.x = x
        self.y = y

    def _shape_center(self):
        return self.x, self.y

    def draw(self, axis: plt.Axes):
        axis.plot(self.x, self.y, 'o', color=self.line_color)

    def rotate(self, degrees: float, x: float = None, y: float = None):
        temp_x = self.x - x
        temp_y = self.y - y

        radians = degrees * (np.pi / 180)
        cos = np.cos(radians)
        sin = np.sin(radians)

        self.x = temp_x * cos - temp_y * sin + x
        self.y = temp_x * sin + temp_y * cos + y

    def translate(self, x: float, y: float):
        self.x += x
        self.y += y

    def scale(self, factor: float):
        pass
