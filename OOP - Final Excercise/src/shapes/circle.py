import numpy as np
import mpl_colors
import matplotlib.pyplot as plt

from basic_shape import BasicShape
from matplotlib.patches import Circle as pltCircle


class Circle(BasicShape):
    def __init__(self, x: float = 0, y: float = 0, radius: float = 0.1, line_color=mpl_colors.Color.BLACK,
                 fill_color=mpl_colors.Color.WHITE):
        super().__init__(line_color, fill_color)
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, axis: plt.Axes):
        circle = pltCircle((self.x, self.y), self.radius, edgecolor=self.line_color, facecolor=self.fill_color)
        axis.add_artist(circle)

    def _shape_center(self):
        return self.x, self.y

    def rotate(self, degrees: float, x: float = None, y: float = None):
        if x is None or y is None:
            return  # Circle does not rotate

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
        self.radius *= factor
