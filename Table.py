import numpy as np


class Table:
    def __init__(self, width, length, index):
        self.width = int((width+0.4)/0.2)
        self.length = int((length+0.4)/0.2)
        self.curr_x_position = -1
        self.curr_y_position = 0
        self.table_number = index
        self.mass_center = (0, 0)

    def set_curr_position(self, new_x_position, new_y_position):
        self.curr_x_position = new_x_position
        self.curr_y_position = new_y_position

    def calculate_width(self, width):
        np.fromnumeric

