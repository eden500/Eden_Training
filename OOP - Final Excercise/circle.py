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