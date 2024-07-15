from typing import Iterable

from shape import Shape
from point import Point
from line import Line
from rectangle import Rectangle
from circle import Circle
from triangle import Triangle

SHAPES_NAMES = {
        "point": Point,
        "line": Line,
        "rectangle": Rectangle,
        "circle": Circle,
        "triangle": Triangle
    }


class CompositeShape(Shape):
    def __init__(self, shapes: Iterable[Shape]):
        for shape in shapes:
            if not isinstance(shape, Shape):
                raise TypeError("All elements must be of type Shape")
        self.shapes = shapes

    def draw(self, axis):
        for shape in self.shapes:
            shape.draw(axis)
