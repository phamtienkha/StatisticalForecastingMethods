import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SES():

    def fit(self, data):
        alpha_list = [i/100 for i in range(1, 100)]
        error_best = 1e100
        alpha_best = 0
        for alpha in alpha_list:
            pred_data = [data[0]]
            for i in range(1, len(data)):
                pred_data_cur = alpha * data[i-1] + (1-alpha) * pred_data[i-1]
                pred_data.append(pred_data_cur)
            error_cur = np.mean([(data[i] - pred_data[i])**2 for i in range(len(data))])
            if error_cur < error_best:
                error_best = error_cur
                alpha_best = alpha
        return alpha_best

    def predict(self, data, predict_len):
        alpha_best = self.fit(data)
        pred_data = [data[0]]
        for i in range(1, len(data)):
            pred_data_cur = alpha_best * data[i - 1] + (1 - alpha_best) * pred_data[i - 1]
            pred_data.append(pred_data_cur)
        prediction = [alpha_best*data[-1] + (1-alpha_best)*pred_data[-1]]*predict_len
        return prediction


if __name__ == '__main__':
    data = pd.read_csv('sample.csv')
    data = list(data['Sales'])
    print(data[-10:-1])

    predict_len = 5
    model = SES()
    prediction = model.predict(data, predict_len)
    print(model.fit())
    print(prediction)

    plt.plot(data)
    plt.plot(range(len(data), len(data) + predict_len), prediction)
    plt.show()
