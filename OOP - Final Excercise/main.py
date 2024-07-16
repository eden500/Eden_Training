from json_reader import JsonReader
from drawer import Drawer

if __name__ == '__main__':
    # json_reader = JsonReader("simple.json")
    # json_reader = JsonReader("composites.json")
    # json_reader = JsonReader("operations1.json")
    # json_reader = JsonReader("operations2.json")
    # json_reader = JsonReader("A street where one house got knocked down because of a hurricane.json")
    json_reader = JsonReader("A city where all the left houses got knocked down because of a hurricane.json")
    drawer = Drawer()

    shapes = json_reader.get_shapes()
    drawer.draw_shapes(shapes)

