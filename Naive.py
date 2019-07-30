import numpy as np


class Naive():
    def predict(self, data, predict_len):
        return [data[-1]] * predict_len
