import json
from shape_factory import ShapeFactory


class JsonReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.__read()
        definitions = self.__load_definitions()
        self.shape_factory = ShapeFactory(definitions)
        self.shapes = self.__load_shapes()

    def __read(self):
        with open(self.file_path, "r") as f:
            return json.load(f)

    def __load_definitions(self):
        return self.data["Definitions"] if "Definitions" in self.data else {}

    def __load_shapes(self):
        return self.shape_factory.create_shapes(self.data["Draw"])

    def get_shapes(self):
        return self.shapes
