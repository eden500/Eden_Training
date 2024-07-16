import numpy as np

from basic_shape import BasicShape
from matplotlib.patches import Circle as pltCircle


class Circle(BasicShape):
    def __init__(self, x=0, y=0, radius=1, line_color="black", fill_color="white"):
        super().__init__(line_color, fill_color)
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, axis):
        circle = pltCircle((self.x, self.y), self.radius, edgecolor=self.line_color, facecolor=self.fill_color)
        axis.add_artist(circle)

    def _shape_center(self):
        return self.x, self.y

    def rotate(self, degrees, x=None, y=None):
        if x is None or y is None:
            return  # Circle does not rotate

        temp_x = self.x - x
        temp_y = self.y - y

        radians = degrees * (np.pi / 180)
        cos = np.cos(radians)
        sin = np.sin(radians)

        self.x = temp_x * cos - temp_y * sin + x
        self.y = temp_x * sin + temp_y * cos + y

    def translate(self, x, y):
        self.x += x
        self.y += y

    def scale(self, factor):
        self.radius *= factor
