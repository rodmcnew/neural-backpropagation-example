import numpy

class Linear:
    @staticmethod
    def get_y(x):
        return x

    @staticmethod
    def get_slope(y):
        return 1

class LeakyRelu:
    def __init__(self, slope_below_zero=0.01):
        self.slope_below_zero = slope_below_zero

    def get_y(self, x):
        if x > 0:
            return x
        else:
            return self.slope_below_zero

    def get_slope(self, y):
        if y > 0:
            return 1
        else:
            return self.slope_below_zero


class Relu:
    @staticmethod
    def get_y(x):
        return max(0, x)

    @staticmethod
    def get_slope(y):
        if y > 0:
            return 1
        else:
            return 0


class Sigmoid:
    @staticmethod
    def get_y(x):
        return 1 / (1 + numpy.exp(-x))

    @staticmethod
    def get_slope(y):
        return y * (1 - y)
