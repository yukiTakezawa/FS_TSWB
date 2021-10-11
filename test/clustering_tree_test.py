import sys
sys.path.append("../")
import unittest
from src.clustering_tree import *
from numpy.testing import assert_array_equal


class ClusteringTest1(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 0.0],
                                 [1.0, 0.5],
                                 [-1.0, 1.0],
                                 [-1.0, -0.5]])

        self.tree = ClusteringTree(self.support, 2, 10, debug_mode=True)

        self.B = np.array([[1.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

        self.wv = np.ones(self.B.shape[0])
        self.wv[0] = 0.0
        self.B = self.B * self.wv[:, np.newaxis]
        
    def test_B(self):
        assert_array_equal(self.tree.B_list[0].toarray(), self.B)

        
class ClusteringTest2(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[0.0, 0.0],
                                 [0.0, 0.5],
                                 [0.3, 0.0],
                                 [1.0, 1.0],
                                 [0.7, 1.0],
                                 [1.0, 0.5]])

        self.tree = ClusteringTree(self.support, 2, 10, debug_mode=True)

        self.B = np.concatenate([np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                           [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]), np.eye(6)])
        self.wv = np.ones(self.B.shape[0])
        self.wv[0] = 0.0
        self.B = self.B * self.wv[:, np.newaxis]

        
    def test_B(self):
        assert_array_equal(self.tree.B_list[0].toarray(), self.B)

        
class ClusteringTest3(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([np.array([math.cos(-2*math.pi * i/12.0 + math.pi/2), math.sin(-2*math.pi * i/12.0 + math.pi/2)]) for i in range(12)])

        self.tree = ClusteringTree(self.support, 4, 100, debug_mode=True)

        self.B = np.concatenate([np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]), np.eye(12)])
        self.wv = np.ones(self.B.shape[0])
        self.wv[0] = 0.0
        self.B = self.B * self.wv[:, np.newaxis]

    def test_B(self):
        assert_array_equal(self.tree.B_list[0].toarray(), self.B)


class ClusteringTest4(unittest.TestCase):
    def setUp(self):
        self.support = np.array([[5.0, 0.0],
                                 [5.0, 1.0],
                                 [5.0, 2.0],
                                 [5.0, 3.0],
                                 [-5.0, 0.0],
                                 [-5.0, 1.0],
                                 [-5.0, 2.0],
                                 [-5.0, 3.0]])

        
        self.tree = ClusteringTree(self.support, 2, 100, debug_mode=True)

        self.B = np.concatenate([np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                           [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]), np.eye(8)])
        self.wv = np.ones(self.B.shape[0])
        self.wv[0] = 0.0
        self.B = self.B * self.wv[:, np.newaxis]

        
    def test_B(self):
        assert_array_equal(self.tree.B_list[0].toarray(), self.B)


class ClusteringTest5(unittest.TestCase):
    def setUp(self):
        self.support = np.array([[5.0, 0.0],
                                 [5.0, 1.0],
                                 [5.0, 2.0],
                                 [5.0, 3.0],
                                 [-5.0, 0.0],
                                 [-5.0, 1.0],
                                 [-5.0, 2.0],
                                 [-5.0, 3.0]])

        
        self.tree = ClusteringTree(self.support, 4, 100, debug_mode=True)

        self.B = np.concatenate([np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                                           [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]), np.eye(8)])

        self.wv = np.ones(self.B.shape[0])
        self.wv[0] = 0.0
        self.B = self.B * self.wv[:, np.newaxis]

        
    def test_B(self):
        assert_array_equal(self.tree.B_list[0].toarray(), self.B)


class ClusteringTest6(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 0.0],
                                 [1.0, 0.5],
                                 [-1.0, 1.0],
                                 [-1.0, -0.5]])

        self.tree = ClusteringTree(self.support, 2, 10, edge=True, debug_mode=True)

        self.wv = np.array([0.0, 1/2, 1/2, 1/4, 1/4, 1/4, 1/4])
        
        self.B = self.wv[:, np.newaxis] * np.array([[1.0, 1.0, 1.0, 1.0],
                                                    [1.0, 1.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0, 1.0],
                                                    [1.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]])
        
        self.wv = np.ones(self.B.shape[0])
        self.wv[0] = 0.0
        self.B = self.B * self.wv[:, np.newaxis]

        
    def test_B(self):
        assert_array_equal(self.tree.B_list[0].toarray(), self.B)

        
if __name__ == "__main__":
    unittest.main()
