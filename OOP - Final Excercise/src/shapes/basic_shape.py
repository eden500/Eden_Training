from shape import Shape
from abc import ABC
import mpl_colors


class BasicShape(Shape, ABC):
    def __init__(self, line_color: mpl_colors.Color = mpl_colors.Color.BLACK,
                 fill_color: mpl_colors.Color = mpl_colors.Color.WHITE):
        self.line_color = line_color
        self.fill_color = fill_color
