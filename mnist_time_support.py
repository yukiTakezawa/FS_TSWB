from src.preprocess import *
from src.clustering_tree import *
import matplotlib.pyplot as plt
from ot.bregman import barycenter_sinkhorn
from multiprocessing import Pool
import os
import re
import argparse


if __name__ == "__main__":
    n_pixels = [28*math.sqrt(1/2), 28, 28*math.sqrt(2), 28*math.sqrt(3), 28*math.sqrt(4), 28*math.sqrt(5), 28*math.sqrt(6)]

    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, help="FastPSD, PSD, or IBP")
    args = parser.parse_args() 
    
    n_classes = [i for i in range(10)]
    time_list = {"1/2" : [], "1" : [], "2" : [], "3" : [], "4" : [], "5" : [], "6" : []}

    for n_class in n_classes:
        for i in range(len(n_pixels)):        

            if args.method == "IBP":
                mnist = MNIST(n_pixels=int(n_pixels[i]))        
                a_list = np.array(mnist.get_data(n_class)[:1000]).T
                M = mnist.compute_M()
                
                start_time = time.time()
                bar = barycenter_sinkhorn(a_list, M, 0.01)
                result = time.time() - start_time

            else:
                mnist = MNIST(n_pixels=int(n_pixels[i]))
                a_list = mnist.get_data(n_class)[:1000]
                support = mnist.get_support()
                tree = ClusteringTree(support)
                
                if args.method == "FastPSD":
                    start_time = time.time()
                    bar = tree.barycenter(a_list)
                    result = time.time() - start_time
                    
                elif args.method == "PSD":
                    start_time = time.time()
                    bar = tree.barycenter_naive(a_list)
                    result = time.time() - start_time
                    
            time_list[list(time_list.keys())[i]].append(result)        
            print(time_list)

    pickle.dump(time_list, open("exp/mnist_time_support_" + args.method + ".pk", "wb"))
