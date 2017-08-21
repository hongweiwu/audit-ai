# -*- coding: utf-8 -*-

import shared as s
import csv
import numpy as np

cur_mode="debug"
if cur_mode=="debug":
    predict_file="data/predict-few.csv"
    file2write="data/suspicious-test.txt"
else:
    predict_file="data/predict.csv"
    file2write="data/suspicious.txt"

violations = ['采购策略问题', '串标', '虚假业务', '收受回扣', '流程违规',
        '成本偏高']
# normalize floating point data, based on a pre-set assumption of
# maximum and minimum values

# load the data to predict, in data/predict.csv
def load_data():
    r = csv.reader(open(predict_file, 'r' ,encoding='utf-8'),
            delimiter=',', quotechar='"')

    raw_data = np.array(list(r))
    s.print_data("raw",raw_data)
    str_data = np.delete(raw_data[1:, 1:], [6], axis=1)
    po_nums = raw_data[1:, 0]

    return s.normalize(str_data.astype('float32')), po_nums

# predict the most suspicious 50 cases based on the binary model,
# and then predict the violation_typeids for these cases
def predict(model_b, model_m, x_data, po_nums):
    pdata_b=model_b.predict(x_data)#二维数组，每一个 member 包含舞弊概率，和非舞弊概率
    #s.print_data("predict Data",pdata_b)
    y_data = pdata_b[:, 0]
    #s.print_data("y_data",y_data)
    #s.print_data("y_data shape0", y_data.shape[0])#就是y_data 的count
    #s.print_data("po_nums" ,po_nums)
    tup = zip(y_data, po_nums, np.arange(y_data.shape[0]))#zip之前需要 组成一个三列的二维数组，分别为 y_data，po_nums,和 0到n
    res = sorted(tup, key=lambda spo: spo[0], reverse=True) #spo for suspicious purchase order
    s.print_data("res desc",res)
    #ret=list()
    count = 0; idx = 0; sus_pos = []; likelihoods=[]; rows = []
    duplicates = set()
    #duplicates.add('0')
    
    while idx < len(res) and count < 50:
        if res[idx][1] in duplicates:
            pass
            #print("%s is already in duplicates ..." %(res[idx][1]))
        else:
            #print("will add %s whose index is %d" %(res[idx][1],res[idx][2]))
            likelihoods.append(res[idx][0])
            sus_pos.append(res[idx][1])
            rows.append(res[idx][2])
            #ret.append(res[idx])
            count += 1
            duplicates.add(res[idx][1])
        idx += 1
    #s.print_data("sus_pos",sus_pos)
    #s.print_data("rows",rows)
    #s.print_data("x_data",x_data)
    #s.print_data("x_data[rows]",x_data[rows])
    pdata_m=model_m.predict(x_data[rows])# x_data[rows] 即为排行较高的怀疑对象，需要被输入的 multicast prediction 的 x_data
    #s.print_data("pdata_m",pdata_m)
    violation_typeids = np.argmax(pdata_m, axis=1)
    #s.print_data("violation_typeids",violation_typeids)
    return sus_pos,likelihoods,violation_typeids
    #return ret

# main routine for ML predicting (inference)
def main():
    x_data, po_nums = load_data()
    from keras.models import load_model
    mfile_b='model/binary.h5'
    try:
        model_b = load_model(mfile_b)
    except:
        print('model file for binary prediction is not available'); exit(1)

    mfile_m='model/multi.h5'
    try:
        model_m = load_model(mfile_m)
    except:
        print('model file for multi-cast prediction is not available'); exit(2)

        model_m = load_model('model/multi.h5')
    f = open(file2write, 'w', encoding='utf-8')
    pos,likelihoods,vids = predict(model_b, model_m, x_data, po_nums)
    s.print_data("pos",pos)
    s.print_data("likelihoods",likelihoods)
    s.print_data("vids",vids)
    for po,likelihood, vid in zip(pos,likelihoods,vids):
        print('%s, %f%%, %s\n' % (po, likelihood*100, violations[vid]))
        f.write('%s, %f%%, %s\n' % (po, likelihood*100,violations[vid]))
    f.close()
    import gc; gc.collect()

# standard python 3 routine to invoke the main() function
if __name__ == '__main__': main()
