from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from enum import Enum


class Operations(Enum):
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALE = "scale"


class Shape(ABC):
    @abstractmethod
    def draw(self, axis: plt.Axes):
        pass

    @abstractmethod
    def _shape_center(self):
        pass

    def apply_operations(self, operations):
        for operation, values in operations.items():
            if operation == Operations.ROTATION.value:
                self.rotate(values)
            elif operation == Operations.TRANSLATION.value:
                self.translate(*values)
            elif operation == Operations.SCALE.value:
                self.scale(values)
            else:
                print(f"Operation {operation} is not supported.")

    @abstractmethod
    def rotate(self, degrees: float, x: float = None, y: float = None):
        pass

    @abstractmethod
    def translate(self, x: float, y: float):
        pass

    @abstractmethod
    def scale(self, factor: float):
        pass
