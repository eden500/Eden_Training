from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, line_color, fill_color):
        self.line_color = line_color
        self.fill_color = fill_color

    @abstractmethod
    def draw(self, axis):
        pass

