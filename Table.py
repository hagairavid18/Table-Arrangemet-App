import numpy as np


class Table:
    def __init__(self, width, length, index):
        self.width = int(width*5)
        self.length = int(length*5)
        self.location = (-1,-1)

        self.table_number = index
        self.mass_center = (0, 0)
        self.min_distance = np.inf

    def shuffle(self):
        n = np.random.randint(0, 2)
        if n == 1:
            temp = self.length
            self.length = self.width
            self.width = temp



