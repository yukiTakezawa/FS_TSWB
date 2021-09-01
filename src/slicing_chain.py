import numpy as np
from treelib import Tree
import matplotlib.pyplot as plt
import networkx as nx
import copy 
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import time
from tqdm import tqdm
from bisect import bisect_left, bisect_right
import math
import random
import itertools
#from memory_profiler import profile

# githubから最新のversonを持ってくる必要がある。
# 実験やり直しを回避するためにこの関数だけコピーした。
#from ot.sliced import get_random_projections
#from random_projections import get_random_projections
from .random_projections import get_random_projections 

class SlicingChain():
    def __init__(self, X, n_slice=1):
        """
        Parameters
        ----------
        X : 
            a set of supports
        n_slice : int
            the number of sampled chains
        """
        
        self.n_slice = n_slice
        self.tree_list = []
        self.B_list = []
        self.Bt_list = []
        self.wv_list = []

        self.argsort_x_list = []
        self.inv_argsort_x_list = []
        
        self.projs = get_random_projections(n_slice, X.shape[1])
        proj_X = np.dot(self.projs, X.T)
        
        for i in tqdm(range(n_slice)):

            x_list = proj_X[i].tolist()
            argsort_array = np.argsort(x_list)
            self.argsort_x_list.append(argsort_array)

            inv_argsort_x_list = np.zeros_like(argsort_array)
            for j in range(len(argsort_array)):
                inv_argsort_x_list[argsort_array[j]] = j
            self.inv_argsort_x_list.append(inv_argsort_x_list)
            
            # minimum value corresponds to the root node.
            D = np.zeros((len(x_list), len(x_list)))
    
            for j in range(1, len(x_list)):
                D[argsort_array[j-1], argsort_array[j]] = 1.0
            B = np.linalg.inv(np.eye(len(x_list)) - D)

            # Compute edge length
            sorted_proj_X = sorted(proj_X[i])
            tmp_wv = (np.array([0] + sorted_proj_X[1:]) - np.array([0] + sorted_proj_X[:-1]))
            wv = np.zeros_like(tmp_wv)
            for j in range(len(wv)):
                wv[argsort_array[j]] = tmp_wv[j]
            wv = wv.astype(np.float32)            

            self.wv_list.append(wv)
            
            B = wv[:, np.newaxis] * B
            self.Bt_list.append(csc_matrix(B.astype(np.float32).T))
            self.B_list.append(csr_matrix(B.astype(np.float32)))

    def dot_B(self, a, n_slice):
        """
        self.B_list[n_slice].dot(a)
        """
        sorted_a = a[self.argsort_x_list[n_slice]][::-1] 
        cum = np.cumsum(sorted_a)[::-1]
        Ba = cum[self.inv_argsort_x_list[n_slice]] 
        return self.wv_list[n_slice] * Ba

    
    def dot_Bt(self, a, n_slice):
        """
        self.Bt_list[n_slice].dot(a)
        """
        a = self.wv_list[n_slice] * a
        sorted_a = a[self.argsort_x_list[n_slice]]
        cum = np.cumsum(sorted_a)
        Ba = cum[self.inv_argsort_x_list[n_slice]] 
        return Ba

    
    def projection_simplex_sort(self, v, z=1):
        # https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w


    def barycenter_naive(self, a_list, max_iter=1500, a=0.05, c=1/4, init_x="l2", debug_mode=False):
        """
        Projected Subgradient Descent
        n_slice=1
        """
        n = len(a_list)

        if init_x=="random":
            while True:
                x = np.random.randn(a_list[0].shape[0]).astype(np.float32)
                x = np.where(x<0, 0, x)
                
                if x.sum() != 0:
                    break
            x = x / x.sum()
        elif init_x=="l2":
            x = sum(a_list) / n
        else:
            print("ERROR : random or l2")
            
        Ba_list = (np.concatenate([self.B_list[0].dot(a_list[i].astype(np.float32))[:, np.newaxis] for i in range(n)], axis=1))
        
        min_val = np.inf
        min_x = x
        loss_list = []
        
        for k in tqdm(range(max_iter)):
            Bx = self.B_list[0].dot(x)
            grad = self.Bt_list[0].dot(np.sign(Bx[:, np.newaxis] - Ba_list)).sum(1) / n


            gamma = a / math.pow(k+1.0, c) / np.linalg.norm(grad)
            x = x - gamma * grad
            x = self.projection_simplex_sort(x)
            
            # compute loss
            loss = np.abs(Ba_list - Bx[:, np.newaxis]).sum()
            
            if min_val > loss:
                min_val = loss
                min_x = x

            if debug_mode:
                loss_list.append(loss)
        #print("loss", self.get_obj_func(self.B, a_list)(x), min_val)
        if debug_mode:
            return min_x, loss_list
        return min_x

    
    def barycenter(self, a_list, max_iter=1500, a=0.05, c=1/4, init_x="l2", debug_mode=False):
        """
        FastPSD

        Parameters
        ----------
        a_list : list of numpy.ndarray (shape is (n_doc, n_words))
            probability measures
        max_iter : int
            maximum number of iterations
        a : float
            step size
        init_x : str
            if init_x is "random", initial value is random. If init_x is "l2", L2 barycenter is used as the initial value

        Return
        ----------
        min_x : numpy.ndarray (shape is (n_words))
            FS-TSWB
        """
        
        n = len(a_list)
        n_leaf = a_list[0].shape[0]

        if init_x=="random":
            while True:
                x = np.random.randn(a_list[0].shape[0]).astype(np.float32)
                x = np.where(x<0, 0, x)
                
                if x.sum() != 0:
                    break
            x = x / x.sum()
        elif init_x=="l2":
            x = sum(a_list) / n
        else:
            print("ERROR : random or l2")
            
        #Ba_list = []
        Ba_list_sum = []
        sorted_Ba = []
        cum_list = []

        for n_slice in range(len(self.B_list)):
            Ba_list = np.concatenate([self.dot_B(a_list[i].astype(np.float32), n_slice)[:, np.newaxis] for i in range(n)], axis=1)
            sorted_Ba.append(np.sort(Ba_list, axis=1))
            Ba_list_sum.append(Ba_list.sum())
            cum_list.append(np.concatenate([np.zeros(Ba_list.shape[0])[:, np.newaxis], np.cumsum(sorted_Ba[-1], axis=1)], axis=1))

        
        min_val = np.inf
        min_x = x
        loss_list = []
        
        for k in tqdm(range(max_iter)):

            grad = np.zeros(n_leaf)
            loss = 0.0
            sort_idx_list = []
            for i in range(len(self.B_list)):
                Bx = self.dot_B(x, i)
                sort_idx_list.append(np.array([bisect_right(sorted_Ba[i][j], Bx[j]) for j in range(self.B_list[i].shape[0])]))
                grad_b = 2 * sort_idx_list[-1] - n
                #grad += self.Bt_list[i].dot(grad_b) / n
                grad += self.dot_Bt(grad_b, i) / n

                # Compute loss
                loss += Ba_list_sum[i] + ((2*sort_idx_list[i] - n) * Bx).sum()
                for j in range(len(sort_idx_list[i])):
                    loss -= 2*cum_list[i][j][sort_idx_list[i][j]]
                
            grad /= self.n_slice
            loss /= n
            
            gamma = a / math.pow(k+1.0, c) / np.linalg.norm(grad)
            x = x - gamma * grad
            x = self.projection_simplex_sort(x)
            
            if min_val > loss:
                min_val = loss
                min_x = x
            if debug_mode:
                loss_list.append(loss)

        if debug_mode:
            return min_x, loss_list
        
        return min_x

    
