from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def draw(self, axis):
        pass

    @abstractmethod
    def _shape_center(self):
        pass

    def apply_operations(self, operations):
        for operation, values in operations.items():
            if operation == "rotation":
                self.rotate(values)
            elif operation == "translation":
                self.translate(*values)
            elif operation == "scale":
                self.scale(values)
            else:
                print(f"Operation {operation} is not supported.")

    @abstractmethod
    def rotate(self, degrees, x=None, y=None):
        pass

    @abstractmethod
    def translate(self, x, y):
        pass

    def scale(self, factor):
        pass
