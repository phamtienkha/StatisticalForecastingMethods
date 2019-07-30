import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import math


def sMAPE(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    loss = 0
    for i in range(len(y_true)):
        loss += 200 * abs(y_true[i] - y_pred[i]) / (abs(y_true[i]) + abs(y_pred[i]))
    return loss / len(y_true)


def MAPE(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    loss = 0
    for i in range(len(y_true)):
        loss += 100 * abs(y_true[i] - y_pred[i]) / (abs(y_true[i])+1e-6)
    return loss / len(y_true)


class HoltWintersMultiplicative():

    def fit(self, data, seasonal):
        # Find the best alpha, betaAst and gamma.
        point_num = 20
        alpha_list = [i / point_num for i in range(1, point_num)]
        betaAst_list = [i / point_num for i in range(1, point_num)]
        season_init_list = [0.1] * seasonal
        error_best = 1e100
        alpha_best = 0.5
        betaAst_best = 0.5
        gamma_best = 0.5
        for alpha in alpha_list:
            for betaAst in betaAst_list:
                gamma_list = [i / point_num for i in range(1, int((1 - alpha) * point_num))]
                for gamma in gamma_list:
                    pred_data = [data[0]]
                    level_list = [data[0]]
                    trend_list = [0]
                    season_list = [0.1]
                    for i in range(1, len(data)):
                        if i < seasonal:
                            level_cur = alpha * (data[i] / season_init_list[i]) + (1 - alpha) * (level_list[i - 1] + trend_list[i - 1])
                            level_list.append(level_cur)
                            trend_cur = betaAst * (level_list[i] - level_list[i - 1]) + (1 - betaAst) * trend_list[i - 1]
                            trend_list.append(trend_cur)
                            season_cur = gamma * (data[i] / (level_list[i - 1] + trend_list[i - 1])) + (1 - gamma) * season_init_list[i]
                            season_list.append(season_cur)
                            pred_data_cur = (level_list[i] + trend_list[i]) * season_init_list[i]
                            pred_data.append(pred_data_cur)
                        else:
                            level_cur = alpha * (data[i] / season_list[i - seasonal]) + (1 - alpha) * (level_list[i - 1] + trend_list[i - 1])
                            level_list.append(level_cur)
                            trend_cur = betaAst * (level_list[i] - level_list[i - 1]) + (1 - betaAst) * trend_list[i - 1]
                            trend_list.append(trend_cur)
                            season_cur = gamma * (data[i] /(level_list[i - 1] + trend_list[i - 1])) + (1 - gamma) * season_list[i - seasonal]
                            season_list.append(season_cur)
                            pred_data_cur = (level_list[i] + trend_list[i]) * season_list[i - seasonal]
                            pred_data.append(pred_data_cur)
                    # error_cur = np.mean([(data[i] - pred_data[i]) ** 2 for i in range(len(data))])
                    error_cur = MAPE(data, pred_data)
                    if error_cur < error_best:
                        error_best = error_cur
                        alpha_best = alpha
                        betaAst_best = betaAst
                        gamma_best = gamma

        return [alpha_best, betaAst_best, gamma_best]

    def predict(self, data, seasonal, predict_len):
        prediction = [0] * predict_len
        [alpha_best, betaAst_best, gamma_best] = self.fit(data, seasonal)
        season_init_list = [0.1] * seasonal
        pred_data = [data[0]]
        level_list = [data[0]]
        trend_list = [0]
        season_list = [0.1]
        for i in range(1, len(data)):
            if i < seasonal:
                level_cur = alpha_best * (data[i] / season_init_list[i]) + (1 - alpha_best) * (
                            level_list[i - 1] + trend_list[i - 1])
                level_list.append(level_cur)
                trend_cur = betaAst_best * (level_list[i] - level_list[i - 1]) + (1 - betaAst_best) * trend_list[i - 1]
                trend_list.append(trend_cur)
                season_cur = gamma_best * (data[i] / (level_list[i - 1] + trend_list[i - 1])) + (1 - gamma_best) * \
                             season_init_list[i]
                season_list.append(season_cur)
                pred_data_cur = (level_list[i] + trend_list[i]) * season_init_list[i]
                pred_data.append(pred_data_cur)
            else:
                level_cur = alpha_best * (data[i] / season_list[i - seasonal]) + (1 - alpha_best) * (
                            level_list[i - 1] + trend_list[i - 1])
                level_list.append(level_cur)
                trend_cur = betaAst_best * (level_list[i] - level_list[i - 1]) + (1 - betaAst_best) * trend_list[i - 1]
                trend_list.append(trend_cur)
                season_cur = gamma_best * (data[i] / (level_list[i - 1] + trend_list[i - 1])) + (1 - gamma_best) * \
                             season_list[i - seasonal]
                season_list.append(season_cur)
                pred_data_cur = (level_list[i] + trend_list[i]) * season_list[i - seasonal]
                pred_data.append(pred_data_cur)
        for i in range(predict_len):
            remainder = (i - 1) % seasonal
            prediction[i] = (level_list[-1] + trend_list[-1]) * season_list[-1 - seasonal + remainder]
        return prediction, level_list, trend_list, season_list, [alpha_best, betaAst_best, gamma_best]


if __name__ == '__main__':
    data = pd.read_csv('train-bandwidth.csv')
    data_cur = data[data.SERVER_NAME == 'SERVER_ZONE01_002']
    for i in range(2, data_cur.shape[1]):
        if math.isnan(data_cur.iloc[0, i]):
            data_cur.iloc[0, i] = data_cur.iloc[0, i-1]

    # print(data_cur.shape[1])
    # print(data_cur.dropna(axis=1).shape[1])
    # print(data_cur)

    predict_len = 720
    seasonal = 24
    data_cur = data_cur.values.tolist()[0]
    train = data_cur[2:-predict_len]
    test = data_cur[-predict_len:]
    model = HoltWintersMultiplicative()
    prediction = model.predict(train, seasonal, predict_len)
    print(test)
    print(prediction)
    print(sMAPE(test, prediction))
    print(MAPE(test, prediction))

    num_points = 1440
    plt.plot(range(len(data_cur[-num_points:])), data_cur[-num_points:])
    plt.plot(range(len(data_cur[-num_points:]) - predict_len, len(data_cur[-num_points:])), prediction)
    # plt.title('DepID: ' + str(dep_id) + '\n sMAPE: ' + str(sMAPE(test, prediction)) + '%')
    plt.show()
