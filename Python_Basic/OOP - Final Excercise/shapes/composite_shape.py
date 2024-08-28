from typing import Iterable

from shape import Shape


class CompositeShape(Shape):
    def __init__(self, shapes: Iterable[Shape]):
        for shape in shapes:
            if not isinstance(shape, Shape):
                raise TypeError("All elements must be of type Shape")
        self.shapes = shapes

    def draw(self, axis):
        for shape in self.shapes:
            shape.draw(axis)

    def _shape_center(self):
        # I decided to calculate the center of the bounding box of the shape created by the centers of the shapes,
        # as I think this represent best how it may be in other applications.
        max_x, max_y, min_x, min_y = -float("inf"), -float("inf"), float("inf"), float("inf")
        for shape in self.shapes:
            cx, cy = shape._shape_center()
            max_x, max_y = max(max_x, cx), max(max_y, cy)
            min_x, min_y = min(min_x, cx), min(min_y, cy)
        return (max_x + min_x) / 2, (max_y + min_y) / 2

        # other implementation for mean of centers, looked less smooth.
        # cxs, cys = [], []
        # for shape in self.shapes:
        #     cx, cy = shape._shape_center()
        #     cxs.append(cx)
        #     cys.append(cy)
        # return sum(cxs) / len(cxs), sum(cys) / len(cys)

    def rotate(self, degrees, x=None, y=None):
        if x is None or y is None:
            x, y = self._shape_center()
        for shape in self.shapes:
            shape.rotate(degrees, x, y)

    def translate(self, x, y):
        for shape in self.shapes:
            shape.translate(x, y)

    def scale(self, factor):
        center_x, center_y = self._shape_center()
        for shape in self.shapes:
            shape.translate(-center_x, -center_y)
            shape.scale(factor)
            shape.translate(center_x, center_y)
