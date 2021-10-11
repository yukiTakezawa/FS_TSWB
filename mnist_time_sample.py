from src.preprocess import *
from src.clustering_tree import *
import matplotlib.pyplot as plt
from ot.bregman import barycenter_sinkhorn
from multiprocessing import Pool
import os
import re
import argparse


if __name__ == "__main__":
    n_samples =  [100, 1000, 2000, 3000, 4000, 5000]

    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, help="FastPSD, PSD, or IBP")
    args = parser.parse_args() 

    time_list = {"100" : [], "1000" : [], "2000" : [], "3000" : [], "4000" : [], "5000" : []}
    n_classes = [i for i in range(10)]

    for n_class in n_classes:
        for i in range(len(n_samples)):
            
            if args.method == "IBP":
                mnist = MNIST()
                all_a_list = mnist.get_data(n_class)
                a_list = np.array(random.sample(all_a_list, n_samples[i])).T
                #a_list = np.array(mnist.get_data(n_class)[:n_samples[i]]).T
                M = mnist.compute_M()

                start_time = time.time()
                bar = barycenter_sinkhorn(a_list, M, 0.01)
                result = time.time() - start_time

            else:
                mnist = MNIST()
                all_a_list = mnist.get_data(n_class)
                a_list = random.sample(all_a_list, n_samples[i])
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
    pickle.dump(time_list, open("exp/mnist_time_sample_" + args.method + ".pk", "wb"))
