from shape import Shape


class CompositeShapeFactory():
    def __init__(self, shapes):
        for shape in shapes:
            if not isinstance(shape, Shape):
                raise TypeError("All elements must be of type Shape")
        self.shapes = shapes

    def create_instance(self):
        pass