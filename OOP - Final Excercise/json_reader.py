import json

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


class JsonReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.shapes = []
        self.__read()
        self.__load_definitions()
        self.__load_shapes()
        self.get_shapes()

    def __read(self):
        f = open(self.file_path, "r")
        self.data = json.load(f)
        f.close()

    def __load_definitions(self):
        pass

    def __load_shapes(self):
        shapes = self.data["Draw"]
        for shape_name, shape_descriptions in shapes.items():
            shape_descriptions = [shape_descriptions] if not isinstance(shape_descriptions, list) else shape_descriptions
            try:
                for shape_description in shape_descriptions:
                    shape = SHAPES_NAMES[shape_name](**shape_description)
                    self.shapes.append(shape)
            except KeyError:
                print(f"Shape {shape_name} is not supported."
                      f"Check Json file definitions.")

    def get_shapes(self):
        return self.shapes
