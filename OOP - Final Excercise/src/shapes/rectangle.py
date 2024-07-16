import numpy as np
from polygon import Polygon
import mpl_colors


class Rectangle(Polygon):
    def __init__(self, x: float = 0, y: float = 0, width: float = 0.1, height: float = 0.1,
                 line_color=mpl_colors.Color.BLACK, fill_color=mpl_colors.Color.WHITE):
        super().__init__(line_color, fill_color)
        self.points = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
