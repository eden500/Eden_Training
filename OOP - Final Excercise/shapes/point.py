import numpy as np
from shape import Shape


class Point(Shape):
    def __init__(self, x, y, line_color, fill_color):
        super().__init__(line_color, fill_color)
        self.x = x
        self.y = y

    def _shape_center(self):
        return self.x, self.y

    def draw(self, axis):
        axis.plot(self.x, self.y, 'o', color=self.line_color)

    def rotate(self, degrees, x=None, y=None):
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
        pass
