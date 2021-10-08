import sys
sys.path.append("../")
import unittest
from util import *
from numpy.testing import assert_array_equal, assert_almost_equal
import ot
import numpy as np

def calc_wb_loss2(barycenter, a_list, M):
    loss = 0.0
    for a in a_list:
        tmp = ot.sinkhorn2(barycenter, a, M, 0.01)
        #print("tur", tmp)
        loss += tmp
    return loss / len(a_list)


class Test(unittest.TestCase):

    def setUp(self):
        self.M = np.random.randn(10, 10).astype(np.float32)
        self.M = np.where(self.M<0, 1, self.M)
        self.M /= self.M.max()
        
        self.a = np.random.randn(10).astype(np.float32)
        self.a = np.where(self.a<0, 0.0, self.a)
        self.a /= self.a.sum()

        self.b_list = []
        for _ in range(20):
            tmp = np.random.randn(10).astype(np.float32)
            tmp = np.where(tmp<0, 0.001, tmp)
            self.b_list.append(tmp / tmp.sum())
        
    def test_calc_wb_loss(self):
        loss = calc_wb_loss(self.a, self.b_list, self.M)
        true_loss = calc_wb_loss2(self.a, self.b_list, self.M)
        print(loss, true_loss)
        assert_almost_equal(loss, true_loss, 5)


class Test2(unittest.TestCase):

    def setUp(self):
        self.M = np.random.randn(20, 20).astype(np.float32)
        self.M = np.where(self.M<0, 0.1, self.M)
        self.M /= self.M.max()
        
        self.a = np.random.randn(20).astype(np.float32)
        self.a = np.where(self.a<0, 0.0, self.a)
        self.a /= self.a.sum()

        self.b_list = []
        for _ in range(30):
            tmp = np.random.randn(20).astype(np.float32)
            tmp = np.where(tmp<0, 0.001, tmp)
            self.b_list.append(tmp / tmp.sum())
        
    def test_calc_wb_loss(self):
        loss = calc_wb_loss(self.a, self.b_list, self.M)
        true_loss = calc_wb_loss2(self.a, self.b_list, self.M)
        assert_almost_equal(loss, true_loss)

        
if __name__ == "__main__":
    unittest.main()
