from shape import Shape
from abc import ABC
import numpy as np


class BasicShape(Shape, ABC):
    def __init__(self, line_color="black", fill_color="white"):
        self.line_color = line_color
        self.fill_color = fill_color
