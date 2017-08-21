# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K

import csv
import numpy as np
import argparse

# constants, violation dictionary, and data used for normalization
batch = 128; folds = 10
violations = {
    '采购策略问题': 0,
    '串标':1,
    '虚假业务':2,
    '收受回扣':3,
    '流程违规':4,
    '成本偏高':5,
}
minval = [1.0, 0, 3401001, -100000000, 101, 1000, -100000000]
denom = [1000052162559.0, 88717952, 1100000, 200000000, 300, 2780,
        200000000]


# start of the program, handles the parsing of arguments
def init_program():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', action='store', required=True,
            type=int, help='specify the epochs to train the model')

    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument('-b', '--binary', action='store_true',
            help='classify between suspicious or not suspicious')
    gp.add_argument('-m', '--multiclass', action='store_true',
            help='determine the category of suspicious actions')

    parser.add_argument('-s', '--save', action='store_true',
            help='to save the model instead of cross validation')

    global args; args = parser.parse_args()


# process data for the binary classification
def process_b(data):
    str_data = np.delete(data[1:, 1:], [7], axis=1)
    for i in range(str_data.shape[0]):
        str_data[i,5] = '0' if str_data[i,5] == 'normal' else '1'
    return str_data.astype('float32')


# process data for the multiclass classification
def process_m(data):
    str_data = np.delete(data[1:, 1:], [7], axis=1)
    for i in range(str_data.shape[0]):
        str_data[i, 5] = violations[str_data[i, 5].split('；')[0]]
    return str_data.astype('float32')


# normalize floating point data, based on a pre-set assumption of
# maximum and minimum values
def normalize(data):
    ret = np.empty(data.shape)
    for i in range(data.shape[1]):
        min_arr = np.full((data.shape[0],), minval[i])
        ret[:,i] = (data[:,i] - min_arr) / denom[i]
    return ret


# returns a list of indices, indicating the training/testing data
# for each iteration of the cross validation
def kfold(dim, k):
    assert(dim % k == 0)
    indices = np.arange(dim); np.random.shuffle(indices)

    ret_train = []; ret_test = []
    for i in range(k):
        ret_train.append(list(np.concatenate((
                indices[0:i*dim//10], indices[(i+1)*dim//10:]))))
        ret_test.append(list(indices[i*dim//10:(i+1)*dim//10]))
    return zip(ret_train, ret_test)


# load data for the binary classification
def load_data_binary():
    f1 = open('data/pos-200.csv', 'r')
    f2 = open('data/neg-200.csv', 'r')
    r1 = csv.reader(f1, delimiter=',', quotechar='"')
    r2 = csv.reader(f2, delimiter=',', quotechar='"')

    pos_data = process_b(np.array(list(r1)))
    neg_data = process_b(np.array(list(r2)))
    data = np.concatenate((pos_data, neg_data), axis=0)
    np.random.shuffle(data)

    x_data = normalize(np.delete(data, [5], axis=1))
    y_data = data[:, 5]
    len_data = data.shape[0] // 1280 * 1280

    f1.close(); f2.close()
    return x_data[:len_data], y_data[:len_data]


# load data for the multiclass classification
def load_data_multiclass():
    f1 = open('data/neg-200.csv', 'r')
    r1 = csv.reader(f1, delimiter=',', quotechar='"')

    data = process_m(np.array(list(r1)))
    x_data = normalize(np.delete(data, [5], axis=1))
    y_data = data[:, 5]
    len_data = data.shape[0] // 1280 * 1280

    f1.close()
    return x_data[:len_data], y_data[:len_data]


# one iteration of cross validation consists of training just one
# instance of our model, and immediately test it
def train_test(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(
            x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=batch,
            epochs=args.epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    return model.evaluate(x_test, y_test, verbose=0)


# train the model based on all the given data, and then save it
def train_save(x_train, y_train):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(
            x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=batch,
            epochs=args.epochs,
            verbose=1)

    fname = 'model/binary.h5' if args.binary else 'model/multi.h5'
    model.save(fname)
    print("\n----------------------------------------\n")
    print('successfully saved the model at', fname)


# conduct the cross validation, take average and print results
def cross_validation(x_data, y_data):
    print(x_data.shape[0] // 10 * 9, 'train samples')
    print(x_data.shape[0] // 10, 'test samples')

    loss = []; accuracy = []
    for train_idx, test_idx in kfold(x_data.shape[0], folds):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        score = train_test(x_train, y_train, x_test, y_test)
        loss.append(score[0]); accuracy.append(score[1])

    print("\n----------------------------------------\n")
    print("run #\ttest loss\ttest accuracy")

    total_loss = total_accuracy = 0.0
    for i in range(folds):
        print("%d\t%f\t%f" % (i, loss[i], accuracy[i]))
        total_loss += loss[i]; total_accuracy += accuracy[i]

    print("\naverage loss:", total_loss/10)
    print("average accuracy:", total_accuracy/10)


# main routine for ML model training
def main():
    init_program()

    x_data = None; y_data = None; global num_classes
    if args.binary:
        num_classes = 2
        x_data, y_data = load_data_binary()
    else:
        num_classes = len(violations)
        x_data, y_data = load_data_multiclass()

    y_data = to_categorical(y_data, num_classes)

    if args.save: train_save(x_data, y_data)
    else: cross_validation(x_data, y_data)

    K.clear_session()


# standard python 3 routine to invoke the main() function
if __name__ == '__main__': main()
