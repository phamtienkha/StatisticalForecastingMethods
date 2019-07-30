import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DampMultiplicative():

    def fit(self, data):
        alpha_list = [i/10 for i in range(1, 10)]
        betaAst_list = [i/10 for i in range(1, 10)]
        phi_list = [i/10 for i in range(1, 10)]
        alpha_best = 0
        betaAst_best = 0
        phi_best = 0
        error_best = 1e100
        for alpha in alpha_list:
            for betaAst in betaAst_list:
                for phi in phi_list:
                    pred_data = [data[0]]
                    level_list = [data[0]]
                    trend_list = [0]
                    for i in range(1, len(data)):
                        level_cur = alpha * data[i] + (1 - alpha) * (level_list[i - 1] * trend_list[i - 1] ** phi)
                        level_list.append(level_cur)
                        trend_cur = betaAst * (level_list[i] / level_list[i - 1]) + (1 - betaAst) * trend_list[i - 1] ** phi
                        trend_list.append(trend_cur)
                        pred_data_cur = level_list[i] * trend_list[i]**phi
                        pred_data.append(pred_data_cur)
                    error_cur = np.mean([(data[i] - pred_data[i]) ** 2 for i in range(len(data))])
                    if error_cur < error_best:
                        error_best = error_cur
                        alpha_best = alpha
                        betaAst_best = betaAst
                        phi_best = phi
        return [alpha_best, betaAst_best, phi_best]

    def predict(self, data, predict_len):
        [alpha_best, betaAst_best, phi_best] = self.fit(data)
        prediction = [0]*predict_len
        pred_data = [data[0]]
        level_list = [data[0]]
        trend_list = [0]
        for i in range(1, len(data)):
            level_cur = alpha_best * data[i] + (1 - alpha_best) * (level_list[i - 1] * trend_list[i - 1]**phi_best)
            level_list.append(level_cur)
            trend_cur = betaAst_best * (level_list[i] / level_list[i - 1]) + (1 - betaAst_best) * trend_list[i - 1]**phi_best
            trend_list.append(trend_cur)
            pred_data_cur = level_list[i] * trend_list[i]**phi_best
            pred_data.append(pred_data_cur)
        for i in range(predict_len):
            prediction[i] = level_list[-1] * trend_list[-1]**np.sum([phi_best**(j+1) for j in range(i)])
        return prediction


if __name__ == '__main__':
    data = pd.read_csv('sample.csv')
    data = list(data['Sales'])
    print(data[-10:-1])

    predict_len = 5
    model = DampMultiplicative()
    prediction = model.predict(data, predict_len)
    print(model.fit(data))
    print(prediction)

    plt.plot(data)
    plt.plot(range(len(data), len(data) + predict_len), prediction)
    plt.show()

