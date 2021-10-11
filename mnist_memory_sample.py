from src.preprocess import *
from src.clustering_tree import *
import matplotlib.pyplot as plt
from ot.bregman import barycenter_sinkhorn
from multiprocessing import Pool
import os
import re
import argparse


def peak_memory():
    """
    Return the peak memory that the process used (MB).
    """
    
    pid = os.getpid()
    with open(f'/proc/{pid}/status') as f:
        # extract "VmHWM:..."
        for line in f:
            if not line.startswith('VmHWM:'):
                continue
            return int(re.search('[0-9]+', line)[0]) / 1000.
    raise ValueError('Not Found')


def calc_tree_bar(func_args):
    tree = func_args[0]
    a_list = pickle.load(open("exp/tmp/" + args.method + "_a_list.pk", "rb"))
    #a_list = args[1]

    barycenter = tree.barycenter(a_list)

    mem_size = peak_memory()
    print(f"Peak Memory : {mem_size} MB")
    return mem_size


def calc_tree_naive_bar(func_args):
    tree = func_args[0]
    #a_list = args[1]
    a_list = pickle.load(open("exp/tmp/" + args.method + "_a_list.pk", "rb"))

    barycenter = tree.barycenter_naive(a_list)

    mem_size = peak_memory()
    print(f"Peak Memory : {mem_size} MB")
    return mem_size


def calc_ibp_bar(func_args):
    #M = args[0]
    #a_list = args[1]
    M = pickle.load(open("exp/tmp/" + args.method + "_M.pk", "rb"))
    a_list = pickle.load(open("exp/tmp/" + args.method + "_A.pk", "rb"))

    barycenter = barycenter_sinkhorn(a_list, M, 0.01)

    mem_size = peak_memory()
    print(f"Peak Memory : {mem_size} MB")
    return mem_size


if __name__ == "__main__":
    n_samples =  [100, 1000, 2000, 3000, 4000, 5000]
    n_classes = [i for i in range(10)]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, help="FastPSD, PSD, or IBP")
    args = parser.parse_args() 

    process_list = [Pool(1) for _ in range(len(n_samples)*len(n_classes))]
    
    mem_list = {}
    for n_sample in n_samples:
        mem_list[str(n_sample)] = []

    n_process = 0
    for i in range(len(n_samples)):        
        for n_class in n_classes:
            if args.method == "IBP":
                mnist = MNIST()        
                a_list = np.array(mnist.get_data(n_class)[:n_samples[i]]).T
                M = mnist.compute_M()

                #pickle.dump(a_list, open("exp/tmp/" + args.method + "_a_list.pk", "wb"))
                pickle.dump(M, open("exp/tmp/" + args.method + "_M.pk", "wb"))
                pickle.dump(a_list, open("exp/tmp/" + args.method + "_A.pk", "wb"))
                
                result = process_list[n_process].map(calc_ibp_bar, [(None,),])

            else:
                mnist = MNIST()
                a_list = mnist.get_data(n_class)[:n_samples[i]]
                support = mnist.get_support()
                tree = ClusteringTree(support)

                pickle.dump(a_list, open("exp/tmp/" + args.method + "_a_list.pk", "wb"))
                #pickle.dump(M, open("exp/tmp/" + args.method + "_M.pk", "wb"))
                #pickle.dump(A, open("exp/tmp/" + args.method + "_A.pk", "wb"))
                
                if args.method == "FastPSD":
                    result = process_list[n_process].map(calc_tree_bar, [(tree,),])
                elif args.method == "PSD":
                    result = process_list[n_process].map(calc_tree_naive_bar, [(tree,),])
        
            process_list[n_process].close()
            mem_list[str(n_samples[i])].append(result[0])
            print(mem_list)
            n_process += 1
    pickle.dump(mem_list, open("exp/mnist_memory_sample_" + args.method + ".pk", "wb"))
