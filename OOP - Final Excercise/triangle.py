from shape import Shape
from matplotlib.patches import Polygon

class Triangle(Shape):
    def __init__(self, x1, y1, x2, y2, x3, y3, line_color, fill_color):
        super().__init__(line_color, fill_color)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def draw(self, axis):
        triangle = Polygon([[self.x1, self.y1], [self.x2, self.y2], [self.x3, self.y3]], edgecolor=self.line_color, facecolor=self.fill_color)
        axis.add_artist(triangle)