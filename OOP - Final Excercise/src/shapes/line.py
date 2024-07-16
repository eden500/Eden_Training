from polygon import Polygon
import numpy as np
import mpl_colors as mcolors
import matplotlib.pyplot as plt


class Line(Polygon):
    def __init__(self, x1: float, y1: float, x2: float, y2: float, line_color: mcolors.Color,
                 fill_color: mcolors.Color):
        super().__init__(line_color, fill_color)
        self.points = np.array([[x1, y1], [x2, y2]])

    def draw(self, axis: plt.Axes):
        axis.plot([self.x1, self.x2], [self.y1, self.y2], color=self.line_color)
