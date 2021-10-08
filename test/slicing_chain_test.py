import sys
sys.path.append("../")
import unittest
from slicing_chain import *
from numpy.testing import assert_array_equal, assert_almost_equal
import ot

class SlicingChainTest1(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 0.0],
                                 [1.0, 0.5],
                                 [-1.0, 1.0],
                                 [-1.0, -0.5]])

        self.tree = SlicingChain(self.support, n_slice=1)
    
    def test_distance(self):
        a = np.random.randn(4)
        a = np.where(a<0, 0, a)
        a /= a.sum()
        
        b = np.random.randn(4)
        b = np.where(b<0, 0, b)
        b /= b.sum()
        
        distance = np.abs(self.tree.B_list[0].dot(a - b)).sum()
        
        proj_X = np.dot(self.tree.projs, self.support.T)
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                M[i, j] = abs(proj_X[0][i] - proj_X[0][j]) 
        true_distance = ot.emd2(a, b, M)
        
        assert_almost_equal(distance, true_distance, 4)

    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

            
class SlicingChainTest2(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[0.0, 0.0],
                                 [0.0, 0.5],
                                 [0.3, 0.0],
                                 [1.0, 1.0],
                                 [0.7, 1.0],
                                 [1.0, 0.5]])


        self.tree = SlicingChain(self.support, n_slice=1)
    
    def test_distance(self):
        
        a = np.random.randn(self.support.shape[0])
        a = np.where(a<0, 0, a)
        a /= a.sum()
        
        b = np.random.randn(self.support.shape[0])
        b = np.where(b<0, 0, b)
        b /= b.sum()
        
        distance = np.abs(self.tree.B_list[0].dot(a - b)).sum()
        
        proj_X = np.dot(self.tree.projs, self.support.T)
        M = np.zeros((self.support.shape[0], self.support.shape[0]))
        for i in range(self.support.shape[0]):
            for j in range(self.support.shape[0]):
                M[i, j] = abs(proj_X[0][i] - proj_X[0][j]) 
        true_distance = ot.emd2(a, b, M)
        
        assert_almost_equal(distance, true_distance, 4)

    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

        
class SlicingChainTest3(unittest.TestCase):
    def setUp(self):
        self.support = np.array([np.array([math.cos(-2*math.pi * i/12.0 + math.pi/2), math.sin(-2*math.pi * i/12.0 + math.pi/2)]) for i in range(12)])
        self.tree = SlicingChain(self.support, n_slice=1)
    
    def test_distance(self):
        
        a = np.random.randn(self.support.shape[0])
        a = np.where(a<0, 0, a)
        a /= a.sum()
        
        b = np.random.randn(self.support.shape[0])
        b = np.where(b<0, 0, b)
        b /= b.sum()
        
        distance = np.abs(self.tree.B_list[0].dot(a - b)).sum()
        
        proj_X = np.dot(self.tree.projs, self.support.T)
        M = np.zeros((self.support.shape[0], self.support.shape[0]))
        for i in range(self.support.shape[0]):
            for j in range(self.support.shape[0]):
                M[i, j] = abs(proj_X[0][i] - proj_X[0][j]) 
        true_distance = ot.emd2(a, b, M)
        
        assert_almost_equal(distance, true_distance, 4)


    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

        
class SlicingChainTest4(unittest.TestCase):
    def setUp(self):
        self.support = np.random.randn(101, 20)
        self.tree = SlicingChain(self.support, n_slice=1)
    
    def test_distance(self):
        
        a = np.random.randn(self.support.shape[0])
        a = np.where(a<0, 0, a)
        a /= a.sum()
        
        b = np.random.randn(self.support.shape[0])
        b = np.where(b<0, 0, b)
        b /= b.sum()
        
        distance = np.abs(self.tree.B_list[0].dot(a - b)).sum()
        
        proj_X = np.dot(self.tree.projs, self.support.T)
        M = np.zeros((self.support.shape[0], self.support.shape[0]))
        for i in range(self.support.shape[0]):
            for j in range(self.support.shape[0]):
                M[i, j] = abs(proj_X[0][i] - proj_X[0][j]) 
        true_distance = ot.emd2(a, b, M)
        
        assert_almost_equal(distance, true_distance, 4)


    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

        
class SlicingChainTest5(unittest.TestCase):
    def setUp(self):
        self.support = np.random.randn(100, 31)
        self.tree = SlicingChain(self.support, n_slice=1)
    
    def test_distance(self):
        
        a = np.random.randn(self.support.shape[0])
        a = np.where(a<0, 0, a)
        a /= a.sum()
        
        b = np.random.randn(self.support.shape[0])
        b = np.where(b<0, 0, b)
        b /= b.sum()
        
        distance = np.abs(self.tree.B_list[0].dot(a - b)).sum()
        
        proj_X = np.dot(self.tree.projs, self.support.T)
        M = np.zeros((self.support.shape[0], self.support.shape[0]))
        for i in range(self.support.shape[0]):
            for j in range(self.support.shape[0]):
                M[i, j] = abs(proj_X[0][i] - proj_X[0][j]) 
        true_distance = ot.emd2(a, b, M)
        
        assert_almost_equal(distance, true_distance, 4)

    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

class SlicingChainTest6(unittest.TestCase):
    def setUp(self):
        self.support = np.random.randn(100, 31)
        self.tree = SlicingChain(self.support, n_slice=10)
    
    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

        
class SlicingChainTest7(unittest.TestCase):
    def setUp(self):
        self.support = np.random.randn(100, 31)
        self.tree = SlicingChain(self.support, n_slice=20)
    
    def test_dot_B(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_B(x, i))
            label.append(self.tree.B_list[i].dot(x))
        assert_almost_equal(label, pred, 4)

    def test_dot_Bt(self):
        label = []
        pred = []
        x = np.random.randn(self.support.shape[0])
        for i in range(1):
            pred.append(self.tree.dot_Bt(x, i))
            label.append(self.tree.Bt_list[i].dot(x))
        assert_almost_equal(label, pred, 4)



if __name__ == "__main__":
    unittest.main()
