import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SES import SES

class Theta():

    def theta_model(self, data, theta):
        prediction = [data[0], data[1]]
        for i in range(2, len(data)):
            pred_cur = prediction[0] + i * (prediction[1] - prediction[0]) + theta * (data[i] + (i - 1) * data[0] - i * data[1])
            prediction.append(pred_cur)
        return prediction

    def predict(self, data, predict_len):
        pred_data_1 = self.theta_model(data, 0)
        pred_data_2 = self.theta_model(data, 2)
        pred_1 = [0] * predict_len
        for i in range(predict_len):
            pred_1[i] = (pred_data_1[-1] - pred_data_1[0]) / (len(data) - 1) * (i + len(data)) + pred_data_1[0]
        ses_model = SES()
        pred_2 = ses_model.predict(pred_data_2, predict_len)
        prediction = [1/2 * (pred_1[i] + pred_2[i]) for i in range(predict_len)]
        return prediction


if __name__ == '__main__':
    data = pd.read_csv('sample.csv')
    data = list(data['Sales'])
    print(data[-10:-1])

    predict_len = 20
    model = Theta()
    prediction = model.predict(data, predict_len)
    print(prediction)

    plt.plot(data)
    plt.plot(range(len(data), len(data) + predict_len), prediction)
    plt.show()

