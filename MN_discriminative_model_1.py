# -*- coding: utf-8 -*-
from LLKLIEP import LLKLIEP
from sklearn import datasets,model_selection
import numpy as np

if __name__ == '__main__':
    #data
    ############################
    '''
    L = LLKLIEP()
    iris = datasets.load_iris()
    
    data = iris["data"]
    label = iris["target"]
    '''
    L = LLKLIEP(edge_type='grid', turning_point=28)
    mnist = datasets.fetch_mldata('MNIST original',data_home='dataset')

    data = mnist.data.astype('float16')
    label = mnist.target.astype('int32')
    #'''
    
    train_size = 10000
    test_size = 5000
    
    data_train, data_test, label_train, label_test = \
    model_selection.train_test_split(data, label, test_size=test_size, train_size=train_size)
    
    data_train /= 255
    data_test /= 255
    ############################
    
    #fit phase
    data_each_label = []
    param_each_label = []
    label_set = set(label_train)
    
    for label in label_set:
        print(label)
        label_bol = label_train == label
        param = L.opt(data_train[label_bol].T, data_train)
        param_each_label.append(param)
        #data_each_label.append((label, iris_data_train[label_bol].T))
        
    #predict phase
    predict = np.zeros([len(label_set),test_size])
    
    for num, param in enumerate(param_each_label):
        predict[num,:] =   L.likelihood(param, data_test, data_train)
        
    predict_prob = predict / np.sum(predict, axis=0)
    answer = np.argmax(predict, axis=0)
    
    acc = np.zeros([test_size, 1])
    acc[label_test == np.array(list(label_set))[answer]] = 1
    accuracy = np.sum(acc)/test_size
    
    print(accuracy)
    
    params = np.array(param_each_label)
    np.savetxt('parameters.csv', params, delimiter=",")
    
