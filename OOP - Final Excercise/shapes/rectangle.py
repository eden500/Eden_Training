import numpy as np
from polygon import Polygon


class Rectangle(Polygon):
    def __init__(self, x=0, y=0, width=10, height=10, line_color="black", fill_color="white"):
        super().__init__(line_color, fill_color)
        self.points = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
