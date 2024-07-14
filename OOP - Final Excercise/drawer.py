import matplotlib.pyplot as plt


class Drawer:
    def __init__(self):
        self.figure, self.axes = plt.subplots()
        self.axes.set_aspect(1)

    def draw_shapes(self, shapes):
        for shape in shapes:
            shape.draw(self.axes)
        plt.show()
