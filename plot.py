# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns

from LLKLIEP import LLKLIEP

L = LLKLIEP(edge_type='grid', turning_point=28)
dim = 784
parameters = np.loadtxt('parameters.csv', delimiter=',')
#sns.heatmap(np.reshape(parameters[9,:784],(28,28)), xticklabels=False, yticklabels=False,\
#            square=True)

#sns.heatmap(np.reshape(parameters[0,:784],(28,28)), xticklabels=False, yticklabels=False,\
#            square=True)


mat = L.bool_nei(dim, 28)
res = np.zeros([dim,dim])
res[mat] = parameters[0,784:]
sns.heatmap(np.reshape(np.sum(res, axis=0),(28, 28)), xticklabels=False, yticklabels=False,\
            square=True)
