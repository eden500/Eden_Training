import numpy as np
from polygon import Polygon


class Triangle(Polygon):
    def __init__(self, x1, y1, x2, y2, x3, y3, line_color="black", fill_color="white"):
        super().__init__(line_color, fill_color)
        self.points = np.array([[x1, y1], [x2, y2], [x3, y3]])
