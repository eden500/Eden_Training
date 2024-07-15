from basic_shape import BasicShape


class Line(BasicShape):
    def __init__(self, x1, y1, x2, y2, line_color, fill_color):
        super().__init__(line_color, fill_color)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, axis):
        print(f"Drawing line from ({self.x1}, {self.y1}) to ({self.x2}, {self.y2}) with line color {self.line_color} and fill color {self.fill_color}")