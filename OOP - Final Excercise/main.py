from json_reader import JsonReader
from drawer import Drawer

if __name__ == '__main__':
    json_reader = JsonReader("simple.json")
    drawer = Drawer()

    shapes = json_reader.get_shapes()
    drawer.draw_shapes(shapes)

