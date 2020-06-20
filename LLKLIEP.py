# -*- coding: utf-8 -*-
from scipy import optimize
import numpy as np

class LLKLIEP(object):
    def __init__(self, kernel='RBF', edge_type='clique', turning_point=None, \
                 sigma=0.5, lamda=0.1):
        self.kernel = kernel
        self.edge_type = edge_type
        self.turning_point = turning_point
        self.sigma = sigma
        self.lamda = lamda
        self.kernel_dic = {'clique':self.clique_karnel,
                           'grid':  self.grid_kernel}
        self.edge_num_dic = {'clique':lambda dim:int(((dim*(dim+1))/2)-dim),
                             'grid':  lambda dim:int(2*(self.turning_point*int(dim/self.turning_point))-\
                                      (self.turning_point+int(dim/self.turning_point)))}
        self.kernel_func = self.kernel_dic[edge_type]
        self.edge_num_func = self.edge_num_dic[edge_type]
        
    def likelihood(self, param, test_data, train_data):
        """
        Calculate likelihood of test data from estimated parameters
        """
        dim = test_data.shape[1]
        edge_num = self.edge_num_func(dim)
        '''
        if self.edge_type == 'clique':
            edge_num = int(((dim*(dim+1))/2)-dim)
        elif self.edge_type == 'grid':
            edge_num = int(2*(self.turning_point*int(dim/self.turning_point))-\
                           (self.turning_point+int(dim/self.turning_point)))
        else:
            raise ValueError("invalid edge type")
        '''
            
        kP = self.kernel_func(train_data.T)
        kQ = self.kernel_func(test_data.T)
        
        bias = param[:dim]
        theta = param[dim:dim+edge_num]
        te = np.exp(bias.T.dot(test_data.T)+theta.dot(kQ))
        ne = np.mean(np.exp(bias.T.dot(train_data.T)+theta.dot(kP)))
    
        return (te/ne)
    
    def opt(self, P, Q):
        
        def func(param,P,Q,kP,kQ,lamda):
            
            l = -np.mean(param[:dim].dot(P)+param[dim:dim+edge_num].dot(kP)) + \
                    np.log(np.mean(np.exp(param[:dim].dot(Q)+param[dim:dim+edge_num].dot(kQ)))) + \
                    np.array([lamda/2]).dot(param[dim:dim+edge_num][None].dot(param[dim:dim+edge_num][None].T))#L2
            return l
        
        def jac(param,P,Q,kP,kQ,lamda):     
            
            N_q = np.sum(np.exp(param[:dim].dot(Q)+param[dim:dim+edge_num].dot(kQ)))
            g_q = np.exp(param[:dim].dot(Q)+param[dim:dim+edge_num].dot(kQ)) / N_q
            g_theta = -np.mean(kP,1) + kQ.dot(g_q.T) + np.array([lamda]).dot(param[dim:dim+edge_num][None])
            
            g_bias = -np.mean(P,1) + Q.dot(g_q.T)
            
            return np.r_[g_bias,g_theta]
        
        dim = P.shape[0]
        edge_num = self.edge_num_func(dim)
        
        Q = Q.T
        '''
        if self.edge_type == 'clique':
            edge_num = int(((dim*(dim+1))/2)-dim)
        elif self.edge_type == 'grid':
            edge_num = int(2*(self.turning_point*int(dim/self.turning_point))-\
                           (self.turning_point+int(dim/self.turning_point)))
        else:
            raise ValueError("invalid edge type")
        '''
        kP = self.kernel_func(P)
        kQ = self.kernel_func(Q)
        
        arg = (P,Q,kP,kQ,self.lamda)
        result = optimize.minimize(func,np.zeros(dim+edge_num),jac=jac,args=arg,method='L-BFGS-B',tol=1e-2)
        
        return result["x"]
    
    def clique_karnel(self, data):
        """
        Basis function calculation for clique
        
        parameters:
        data:
        sigma:
            Bandwidth with RBF kernel.
        kernel:
            Basis function specification. Select 'RBF' or 'lin'.
        
        return:
        phi:
            Basis function calculation result.
        """
        (dim, num) = data.shape
        #Combination of each dimension that requires calculation of basis functions
        
        calc_area = np.tri(dim, dim, -1)
        calc_area = (calc_area==1)
        calc_area = calc_area[:,:,None]
        area = np.tile(calc_area,(1, 1, num))
    
        data = data[:,:,None]
        data = np.rollaxis(data, 2, 1)
        base = np.tile(data,(1, data.shape[0], 1))
    
        trans = np.rollaxis(base.T, 0, 3)        
        
        calc = np.empty([dim,dim,num])
        if self.kernel == 'RBF':
            calc[area] = np.exp(-((base[area]-trans[area])**2 / (2*self.sigma**2)))
        elif self.kernel == 'lin':
            calc[area] = base[area] * trans[area]
        else:
            calc[area] = base[area] * trans[area]
        
        phi = np.reshape(calc[area], (int(dim*(dim+1)/2)-dim, num))
        
        return phi
    
    def bool_nei(self, dim, l):
        base_mat = np.zeros([dim,dim])
        #bool_eye = (np.eye(dim) == 1)
        bool_left_eye = (np.tri(dim,dim,1).T == np.tri(dim, dim, -1))
        bool_left = (np.tri(dim, dim, l).T == np.tri(dim, dim, -l))
        
        #left_right = np.logical_or(bool_left,bool_left.T)
        #eye_l_r = np.logical_or(bool_eye,left_right)
        
        #base_mat[bool_eye] = 1
        base_mat[bool_left_eye] = 2
        base_mat[bool_left] = 1
        
        for i in [x*l for x in range(1, int(np.ceil(dim/l)))]:
            base_mat[i, int(np.where(base_mat[i,:] == 2)[0][0])] = 0
                
        base_mat[base_mat != 0] =1
        
        return base_mat == 1
        
    def grid_kernel(self, data):
        """
        Basis function calculation for grid graph
        
        parameters:
        data:
        turning point:
            number of wrap points, like width of the image.
        sigma:
            Bandwidth with RBF kernel.
        kernel:
            Basis function specification. Select 'RBF' or 'lin'.
        
        return:
        phi:
            Basis function calculation result.
        """
            
        (dim, num) = data.shape
        bool_map = self.bool_nei(dim)
        (ind_list_0, ind_list_1) = np.where(bool_map==True)
        
        #del bool_map
            
        phi = np.zeros([ind_list_0.shape[0], num])
        phi = np.exp(-((data[ind_list_0,:] - data[ind_list_1,:])**2 / (2*self.sigma**2)))
        
        return phi
