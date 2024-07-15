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

class ShapeFactory:
    def __init__(self, additional_shapes={}):
        self.additional_shapes = additional_shapes

    def set_additional_shapes(self, additional_shapes):
        self.additional_shapes = additional_shapes

    def create_shapes(self, shapes_dict):
        created_shapes = []
        for shape_name, shape_descriptions in shapes_dict.items():
            shape_descriptions = [shape_descriptions] if not isinstance(shape_descriptions, list) else shape_descriptions
            for shape_description in shape_descriptions:
                try:
                    if shape_name in SHAPES_NAMES:
                        shape = SHAPES_NAMES[shape_name](**shape_description)
                        created_shapes.append(shape)
                    elif shape_name in self.additional_shapes:
                        composite_shapes = self.additional_shapes[shape_name]
                        created_shapes.extend(self.create_shapes(composite_shapes))
                    else:
                        print(f"Shape {shape_name} is not supported. Check Json file definitions.")
                except TypeError:
                    print(f"Values {shape_description} are not valid for shape {shape_name}.")

        return created_shapes
