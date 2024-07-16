import matplotlib.pyplot as plt
from typing import Iterable
from shapes.shape import Shape

class Drawer:
    def __init__(self):
        self.figure, self.axes = plt.subplots()
        self.axes.set_aspect(1)

    def draw_shapes(self, shapes: Iterable[Shape]):
        for shape in shapes:
            shape.draw(self.axes)
        plt.show()
