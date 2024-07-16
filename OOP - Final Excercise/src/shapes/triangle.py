import numpy as np
from polygon import Polygon
import mpl_colors


class Triangle(Polygon):
    def __init__(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
                 line_color=mpl_colors.Color.BLACK, fill_color=mpl_colors.Color.WHITE):
        super().__init__(line_color, fill_color)
        self.points = np.array([[x1, y1], [x2, y2], [x3, y3]])
