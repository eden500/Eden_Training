from polygon import Polygon
import numpy as np


class Line(Polygon):
    def __init__(self, x1, y1, x2, y2, line_color, fill_color):
        super().__init__(line_color, fill_color)
        self.points = np.array([[x1, y1], [x2, y2]])

    def draw(self, axis):
        axis.plot([self.x1, self.x2], [self.y1, self.y2], color=self.line_color)
