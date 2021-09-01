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
#from memory_profiler import profile

class ClusteringTree():
    def __init__(self, X, k=5, d=6, n_slice=1, edge=False, debug_mode=False):
        """
        Parameter
        ----------
        X : 
            a set of supports
        k : int
            a number of child nodes
        d : int
            depth of a tree
        n_slice : int
            the number of sampled trees
        """

        self.n_slice = n_slice
        self.B_list = []
        self.Bt_list = []
        self.wv_list = []
        
        for i in tqdm(range(n_slice)):
            tree = self.build_tree(X, k, d, debug_mode=debug_mode)
            
            D1, D2 = self.gen_matrix(tree, X)
            
            self.sparse_D = coo_matrix(np.concatenate([D1, D2], axis=1).T)

            if edge:
                wv = self.calc_wv(tree)
            else:
                wv = np.ones(D1.shape[1] + D2.shape[1])
                wv[0] = 0.0
                
            self.wv_list.append(wv)

            B = np.linalg.inv(np.eye(D1.shape[0]) - D1).dot(D2)
            B = np.concatenate([B, np.eye(D2.shape[1])], axis=0)
            B = wv[:, np.newaxis] * B
            self.Bt_list.append(csc_matrix(B.astype(np.float32).T))
            self.B_list.append(csr_matrix(B.astype(np.float32)))

            
    def incremental_farthest_search(self, points, remaining_set, k, debug_mode=False):
        n_points = len(remaining_set)
        remaining_set = copy.deepcopy(remaining_set)

        if not debug_mode:
            solution_set = [remaining_set[random.randint(0, n_points-1)]]
        else:
            solution_set = [remaining_set[0]]
        remaining_set.remove(solution_set[0])

        for i in range(k-1):

            distance_list = []

            for idx in remaining_set:

                in_distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
                distance_list.append(min(in_distance_list))

            sol_idx = remaining_set[np.argmax(distance_list)]
            remaining_set.remove(sol_idx)
            solution_set.append(sol_idx)
            
        return solution_set

    
    def distance(self, A, B):
        return np.linalg.norm(A - B)

    
    def grouping(self, points, remaining_set, solution_set):
        n_points = len(points)
        remaining_set = copy.deepcopy(remaining_set)

        group = []
        for _ in range(len(solution_set)):
            group.append([])

        for idx in remaining_set:
            distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
            group_idx = np.argmin(distance_list)
            group[group_idx].append(idx)

        return group

    
    def clustering(self, points, remaining_set, k, debug_mode=False):
        solution_set = self.incremental_farthest_search(points, remaining_set, k, debug_mode=debug_mode)
        return self.grouping(points, remaining_set, solution_set)

    
    def _build_tree(self, X, remaining_set, k, d, debug_mode=False):
        tree = Tree()
        tree.create_node(data=None)

        if len(remaining_set) <= k or d==1:
            for idx in remaining_set:
                tree.create_node(parent=tree.root, data=idx)
            return tree

        groups = self.clustering(X, remaining_set, k, debug_mode=debug_mode)
        #print(groups)
        for group in groups:
            if len(group)==1:
                tree.create_node(parent=tree.root, data=group[0])
            else:
                subtree = self._build_tree(X, group, k, d-1, debug_mode=debug_mode)
                tree.paste(tree.root, subtree)
        return tree

    
    def build_tree(self, X, k, d, debug_mode=False):
        """
        k : the number of child nodes
        d : the depth of the tree
        """
        remaining_set = [i for i in range(len(X))]
        return self._build_tree(X, remaining_set, k, d, debug_mode=debug_mode)

    
    def gen_matrix(self, tree, X):
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in = n_node - n_leaf
        D1 = np.zeros((n_in, n_in))
        D2 = np.zeros((n_in, n_leaf))
        
        in_node = [node.identifier for node in tree.all_nodes() if node.data==None]
        

        for node in tree.all_nodes():
            # check node is leaf or not
            if node.data is not None:
                parent_idx = in_node.index(tree.parent(node.identifier).identifier)
                D2[parent_idx, node.data] = 1.0
            elif node.identifier == tree.root:
                continue
            else:
                parent_idx = in_node.index(tree.parent(node.identifier).identifier)
                node_idx = in_node.index(node.identifier)
                D1[parent_idx, node_idx] = 1.0
        return D1, D2


    def calc_wv(self, tree):
        in_node = [node for node in tree.all_nodes() if node.data==None]
        leaf_node = [node for node in tree.all_nodes() if node.data is not None]

        wv = np.zeros(len(in_node) + len(leaf_node))

        for i in range(len(in_node)):
            depth = tree.depth(in_node[i].identifier)
            if depth==0:
                wv[i] = 0.0
            else:
                wv[i] = 2**(-depth)
    
        for node in leaf_node:
            depth = tree.depth(node.identifier)
            idx = node.data + len(in_node)
            wv[idx] = 2**(-depth)
            
        return wv

    def calc_wv2(self, tree):
        in_node = [node for node in tree.all_nodes() if node.data==None]
        leaf_node = [node for node in tree.all_nodes() if node.data is not None]

        wv_in = np.zeros(len(in_node))
        wv_leaf =  np.zeros(len(leaf_node))

        for i in range(len(in_node)):
            depth = tree.depth(in_node[i].identifier)
            if depth==0:
                wv_in[i] = 0.0
            else:
                wv_in[i] = 2**(-depth)
    
        for node in leaf_node:
            depth = tree.depth(node.identifier)
            idx = node.data
            wv_leaf[idx] = 2**(-depth)
            
        return wv_in, wv_leaf    

    
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
        PSD
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
            tmp = Bx[:, np.newaxis] - Ba_list
            #grad = self.Bt_list[0].dot(np.sign(Bx[:, np.newaxis] - Ba_list)).sum(1) / n
            grad = self.Bt_list[0].dot(np.sign(tmp)).sum(1) / n

            gamma = a / math.pow(k+1.0, c) / np.linalg.norm(grad)
            x = x - gamma * grad
            x = self.projection_simplex_sort(x)
            
            # compute loss
            #loss = np.abs(Ba_list - Bx[:, np.newaxis]).sum()
            loss = np.abs(tmp).sum()
            
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
            a maximum number of iterations
        a : float
            step size
        init_x : str
            if init_x is "random", initial value is random. If init_x is "l2", L2 barycenter is used as the initial value

        Return
        ----------
        min_x : numpy.ndarray (shape is (n_words))
            the FS-TSWB
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
        
        for B in self.B_list:
            Ba_list = np.concatenate([B.dot(a_list[i].astype(np.float32))[:, np.newaxis] for i in range(n)], axis=1)
            #Ba_list.append(np.concatenate([B.dot(a_list[i].astype(np.float32))[:, np.newaxis] for i in range(n)], axis=1))
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
                Bx = self.B_list[i].dot(x)
                sort_idx_list.append(np.array([bisect_right(sorted_Ba[i][j], Bx[j]) for j in range(self.B_list[i].shape[0])]))
                grad_b = 2 * sort_idx_list[-1] - n
                grad += self.Bt_list[i].dot(grad_b) / n

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
