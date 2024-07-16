from shape import Shape


class Point(Shape):
    def __init__(self, x, y, line_color, fill_color):
        super().__init__(line_color, fill_color)
        self.x = x
        self.y = y

    def draw(self):
        print(f"Drawing point at ({self.x}, {self.y}) with line color {self.line_color} and fill color {self.fill_color}")

