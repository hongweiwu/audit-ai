# -*- coding: utf-8 -*-

from keras.models import load_model
import csv
import numpy as np

# violation dictionary, and data used for normalization
violations = ['采购策略问题', '串标', '虚假业务', '收受回扣', '流程违规',
        '成本偏高']

minval = [1.0, 0, 3401001, -100000000, 101, 1000, -100000000]
denom = [1000052162559.0, 88717952, 1100000, 200000000, 300, 2780,
        200000000]


# normalize floating point data, based on a pre-set assumption of
# maximum and minimum values
def normalize(data):
    ret = np.empty(data.shape)
    for i in range(data.shape[1]):
        min_arr = np.full((data.shape[0],), minval[i])
        ret[:,i] = (data[:,i] - min_arr) / denom[i]
    return ret


# load the data to predict, in data/predict.csv
def load_data():
    r = csv.reader(open('data/predict.csv', 'r'),
            delimiter=',', quotechar='"')

    raw_data = np.array(list(r))
    str_data = np.delete(raw_data[1:, 1:], [6], axis=1)
    num_list = raw_data[1:, 0]

    return normalize(str_data.astype('float32')), num_list


# predict the most suspicious 50 cases based on the binary model,
# and then predict the kind for these cases
def predict(model_b, model_m, x_data, num_list):
    y_data = model_b.predict(x_data)[:, 0]
    tup = zip(y_data, num_list, np.arange(y_data.shape[0]))
    res = sorted(tup, key=lambda x: x[0])

    count = 0; idx = 0; sus_list = []; rows = []
    duplicates = set(); duplicates.add('0')

    while idx < len(res) and count < 50:
        if res[idx][1] in duplicates:
            idx += 1; continue
        sus_list.append(res[idx][1]); rows.append(res[idx][2])
        count += 1; duplicates.add(res[idx][1]); idx += 1

    kind = np.argmax(model_m.predict(x_data[rows]), axis=1)
    return sus_list, kind


# main routine for ML predicting (inference)
def main():
    x_data, num_list = load_data()
    try:
        model_b = load_model('model/binary.h5')
        model_m = load_model('model/multi.h5')
    except:
        print('model file requested is not available'); exit(1)

    f = open('suspicious.txt', 'w')
    sus, kind = predict(model_b, model_m, x_data, num_list)
    for n, k in zip(sus, kind):
        f.write('%s, %s\n' % (n, violations[k]))
    f.close()


# standard python 3 routine to invoke the main() function
if __name__ == '__main__': main()
