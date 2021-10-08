import sys
sys.path.append("../")
import unittest
import sinkhorn
from numpy.testing import assert_array_equal, assert_almost_equal
import ot
import numpy as np
import torch
import ot.gpu

class Test(unittest.TestCase):
    def setUp(self):
        n = 10
        self.k = 20
        self.M = np.random.randn(n, n).astype(np.float32)
        self.M = np.where(self.M<0, 1, self.M)
        self.M /= self.M.max()
        
        self.a = np.random.randn(n).astype(np.float32)
        self.a = np.where(self.a<0, 0.1, self.a)
        self.a /= self.a.sum()

        self.b_list = []
        for i in range(self.k):
            tmp = np.random.randn(n).astype(np.float32)
            tmp = np.where(tmp<0, 0.1, tmp)
            tmp /= tmp.sum()
            self.b_list.append(tmp)
        
    def test_calc_wb_loss(self):
        #true_loss = np.array([ot.sinkhorn2(self.a, self.b_list[i], self.M, 0.01)[0] for i in range(self.k)])
        true_loss = ot.gpu.sinkhorn(self.a, np.array(self.b_list).T, self.M, 0.01)
        print(true_loss)
        sinkhorn_layer = sinkhorn.SinkhornLayer(0.01)
        loss = sinkhorn_layer(torch.tensor(self.M).cuda(0), torch.tensor(self.a).cuda(0), torch.tensor(torch.tensor(self.b_list)).cuda(0), 10000)
        
        assert_almost_equal(np.array(loss), true_loss, 4)
        
if __name__ == "__main__":
    unittest.main()
