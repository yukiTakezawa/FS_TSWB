import sys
sys.path.append("../")
import unittest
from quadtree import *
from numpy.testing import assert_array_equal


class QuadtreeTest1(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 1.0],
                                 [1.0, -1.0],
                                 [-1.0, 1.0],
                                 [-1.0, -1.0]])

        self.quadtree = Quadtree(self.support, random_shift=False, width=2.0, origin=np.array([0.0, 0.0]))
        self.wv = np.array([0.0, 0.5, 0.5, 0.5, 0.5])

        self.B = self.wv[:, np.newaxis] *np.array([[1.0, 1.0, 1.0, 1.0],
                                                   [1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0], 
                                                   [0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0]])
        
    def test_wv(self):
        assert_array_equal(self.quadtree.wv, self.wv)
        
    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)


class QuadtreeTest2(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 1.0],
                                 [1.0, -1.0],
                                 [-1.0, 1.0],
                                 [-1.0, -1.0]])

        self.quadtree = Quadtree(self.support, random_shift=False, width=2.0, origin=np.array([-2.0, -2.0]))
        self.wv = np.array([0.0, 0.5, 0.25, 0.25, 0.25, 0.25])

        self.B = self.wv[:, np.newaxis] *np.array([[1.0, 1.0, 1.0, 1.0],
                                                   [1.0, 1.0, 1.0, 1.0],
                                                   [1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0], 
                                                   [0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0]])
        
    def test_wv(self):
        assert_array_equal(self.quadtree.wv, self.wv)
        

    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)

        
class QuadtreeTest3(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, -1.0],
                                 [1.0, 1.0],
                                 [1.0, 3.0],
                                 [-1.0, -1.0],
                                 [-1.0, 1.0],
                                 [-1.0, 3.0]])

        self.quadtree = Quadtree(self.support, random_shift=False, width=4.0, origin=np.array([0.0, 0.0]))

        self.wv = np.array([0.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25])
        # the array whose second and third is swapped is also OK. 
        self.B = self.wv[:, np.newaxis] *np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                                                            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],                  
                                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
    def test_wv(self):
        assert_array_equal(self.quadtree.wv, self.wv)        
        
    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)


class QuadtreeTest4(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 1.0, 1.0],
                                 [1.0, 1.0, -1.0],
                                 [-1.0, -1.0, 1.0],
                                 [-1.0, -1.0, -1.0]])

        self.quadtree = Quadtree(self.support, random_shift=False, width=2.0, origin=np.array([0.0, 0.0, 0.0]))
        self.wv = np.array([0.0, 0.5, 0.5, 0.5, 0.5])

        self.B = self.wv[:, np.newaxis]*np.array([[1.0, 1.0, 1.0, 1.0],
                                                  [1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0], 
                                                  [0.0, 0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 0.0, 1.0]])
        
    def test_wv(self):
        assert_array_equal(self.quadtree.wv, self.wv)
        
    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)


class QuadtreeTest5(unittest.TestCase):
    def setUp(self):        
        self.support = np.array([[1.0, 1.0, 1.0],
                                 [1.0, 1.0, -1.0],
                                 [1.0, -1.0, 1.0],
                                 [1.0, -1.0, -1.0],
                                 [-1.0, 1.0, 1.0],
                                 [-1.0, 1.0, -1.0],
                                 [-1.0, -1.0, 1.0],
                                 [-1.0, -1.0, -1.0]])
                                 
        self.quadtree = Quadtree(self.support, random_shift=False, width=2.0, origin=np.array([0.0, 0.0, 0.0]))
        self.wv = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        self.B = self.wv[:, np.newaxis]*np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
    def test_wv(self):
        assert_array_equal(self.quadtree.wv, self.wv)
        
    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)

        
class QuadtreeTest6(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[-7.0, 7.0],
                                 [1.0, 7.0],
                                 [-7.0, 5.0],
                                 [3.0, 5.0],
                                 [-7.0, 1.0],
                                 [-5.0, 1.0]])

        self.quadtree = Quadtree(self.support, random_shift=False, width=8.0, origin=np.array([0.0, 0.0]))
        #self.wv = np.array([0.0, 0.5, 0.5, 0.5, 0.5])

        # the array whose 2~5 row are swapped is also OK.
        self.B = self.quadtree.wv[:, np.newaxis] *np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                                                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                                                            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        
    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)


class QuadtreeTest7(unittest.TestCase):
    def setUp(self):
        
        self.support = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                 [1.0, 1.0, 1.0, 1.0, -1.0],
                                 [1.0, 1.0, 1.0, -1.0, 1.0],
                                 [1.0, 1.0, 1.0, -1.0, -1.0]])

        self.quadtree = Quadtree(self.support, random_shift=False, width=2.0, origin=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        self.wv = np.array([0.0, 0.5, 0.5, 0.5, 0.5])

        self.B = self.wv[:, np.newaxis] *np.array([[1.0, 1.0, 1.0, 1.0],
                                                   [1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0], 
                                                   [0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0]])
        
    def test_wv(self):
        assert_array_equal(self.quadtree.wv, self.wv)
        
    def test_B(self):
        assert_array_equal(self.quadtree.B.toarray(), self.B)

        
if __name__ == "__main__":
    unittest.main()
