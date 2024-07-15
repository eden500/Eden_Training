from basic_shape import BasicShape
from matplotlib.patches import Rectangle as pltRectangle


class Rectangle(BasicShape):
    def __init__(self, x=0, y=0, width=10, height=10, line_color="black", fill_color="white"):
        super().__init__(line_color, fill_color)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, axis):
        rectangle = pltRectangle((self.x, self.y), self.width, self.height, edgecolor=self.line_color, facecolor=self.fill_color)
        axis.add_artist(rectangle)
