from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction import stop_words as sklearn_stop_words # for python 3.8
from sklearn.feature_extraction import text as sklearn_stop_words
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim
import gensim.downloader
from wordcloud import *
import pickle
from mat4py import loadmat
import torch
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stop_words
import torchvision
import torchvision.transforms as transforms

    
class AGNews2():

    def __init__(self, glove_vectors=None):
        print("loading...")
        self.train_df = pd.read_csv("data/agnews/train.csv")
        print("complete")
        
        if glove_vectors is None: 
            self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
        else:
            self.glove_vectors = glove_vectors

        self.ps = PorterStemmer()
        self.stop_words = set(sklearn_stop_words.ENGLISH_STOP_WORDS) | set(nltk_stop_words.words('english'))
        
    def clean(self, sentence):
        meaningful_words = word_tokenize(re.sub("[^a-zA-Z]", " ", sentence).lower())
        meaningful_words = [self.ps.stem(word) for word in meaningful_words if word not in self.stop_words]
        return " ".join(meaningful_words)
    
    def get_agnews(self, class_n):

        if not (class_n in [1,2,3,4]):
            print("class_num must be 1, 2, 3, 4")
            return None
        
        vectorizer = CountVectorizer(analyzer='word', stop_words=self.stop_words)

        corpus = self.train_df[self.train_df["Class Index"] == class_n]["Description"].values.tolist()

        for i in tqdm(range(len(corpus))):
            corpus[i] = self.clean(corpus[i])

        X = vectorizer.fit_transform(corpus)

        # Above X contains the words that are not contained in the GloVe embeddings.
        # Then, remove such words.
        additional_stop_words = []    
        for w in tqdm(vectorizer.get_feature_names()):
            try:
                self.glove_vectors[w]
            except:
                additional_stop_words.append(w)

        vectorizer2 = CountVectorizer(analyzer='word', stop_words=self.stop_words | set(additional_stop_words))

        X = vectorizer2.fit_transform(corpus)
        words = vectorizer2.get_feature_names()

        # remove documents that have only stop words.
        # Then, normalize bag-of-words.
        a = (X.sum(1)).squeeze(1).tolist()[0]
        invalid_idx = []
        for i in range(len(a)):
            if a[i] <= 0.0:
                invalid_idx.append(i)

        a_list = [(X[i] / X[i].sum()).toarray()[0] for i in range(X.shape[0]) if i not in invalid_idx]

        return a_list, words

    
    def get_support(self, words):
        support = np.zeros((len(words), 50)) 

        for i in range(len(words)):
            support[i] = self.glove_vectors[words[i]]

        return support

    def compute_M(self, words, device=0):
        M_list = []

        for sub_words1 in [words[:5000], words[5000:10000], words[10000:15000], words[15000:20000], words[20000:]]:
            M_tmp_list = []
            for sub_words2 in [words[:5000], words[5000:10000], words[10000:15000], words[15000:20000], words[20000:]]:

                tmp1 = np.zeros((len(sub_words1), 50))

                for i in range(len(sub_words1)):
                    tmp1[i] = self.glove_vectors[sub_words1[i]]

                tmp2 = np.zeros((len(sub_words2), 50))

                for i in range(len(sub_words2)):
                    tmp2[i] = self.glove_vectors[sub_words2[i]]

                tmp1 = torch.tensor(tmp1).half().cuda(device)
                tmp2 = torch.tensor(tmp2).half().cuda(device)

                M_tmp = (torch.norm(tmp1.unsqueeze(1) - tmp2.unsqueeze(0), dim=2).permute(1,0).T)
                M_tmp_list.append(M_tmp.detach().cpu().numpy())
            M_list.append(M_tmp_list)
        M = np.concatenate([np.concatenate(M_list[i], axis=1) for i in range(len(M_list))]).astype(np.float32)
        return M / M.max()

    
class AMAZON2():
    def __init__(self, glove_vectors=None):
        print("reading mat file...")
        self.dataset = loadmat("./data/amazon-emd_tr_te_split.mat")
        print("done.")

        if glove_vectors is None: 
            self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
        else:
            self.glove_vectors = glove_vectors

            
    def get_amazon(self, class_num):
        words = []
        
        for i in tqdm(range(len(self.dataset["words"]))):
            if self.dataset["Y"][i] != class_num:
                continue
            
            tmp = []

            for j in range(len(self.dataset["words"][i])):
                
                try:
                    vec = self.glove_vectors[self.dataset["words"][i][j]]

                    if self.dataset["words"][i][j] not in words:
                        words.append(self.dataset["words"][i][j])
                except:
                    pass
        print("len(words)", len(words))
        a_list = []

        for i in tqdm(range(len(self.dataset["BOW_X"]))):

            bow = np.zeros(len(words))

            for j in range(len(self.dataset["BOW_X"][i])):

                if self.dataset["words"][i][j] not in words:
                    continue

                idx = words.index(self.dataset["words"][i][j])
                bow[idx] = self.dataset["BOW_X"][i][j]

            if self.dataset["Y"][i] == class_num:
                a_list.append(bow / bow.sum())

        return a_list, words

    
    def get_support(self, words):
        support = np.zeros((len(words), 50)) 

        for i in range(len(words)):
            support[i] = self.glove_vectors[words[i]]

        return support
    

    def compute_M(self, words, device=0):
        M_list = []

        for sub_words1 in [words[:5000], words[5000:10000], words[10000:15000], words[15000:]]:
            M_tmp_list = []
            for sub_words2 in [words[:5000], words[5000:10000], words[10000:15000], words[15000:]]:

                tmp1 = np.zeros((len(sub_words1), 50))

                for i in range(len(sub_words1)):
                    tmp1[i] = self.glove_vectors[sub_words1[i]]

                tmp2 = np.zeros((len(sub_words2), 50))

                for i in range(len(sub_words2)):
                    tmp2[i] = self.glove_vectors[sub_words2[i]]

                tmp1 = torch.tensor(tmp1).half().cuda(device)
                tmp2 = torch.tensor(tmp2).half().cuda(device)

                M_tmp = (torch.norm(tmp1.unsqueeze(1) - tmp2.unsqueeze(0), dim=2).permute(1,0).T)
                M_tmp_list.append(M_tmp.detach().cpu().numpy())
            M_list.append(M_tmp_list)
        M = np.concatenate([np.concatenate(M_list[i], axis=1) for i in range(len(M_list))]).astype(np.float32)
        return M / M.max()

    
class MNIST():
    def __init__(self, n_pixels=28):
        self.n_pixels = n_pixels
        transform = transforms.Compose(
            [transforms.Resize(n_pixels), transforms.ToTensor()])

        self.train_data = torchvision.datasets.MNIST(root = './data', train=True, transform=transform)
        self.zero_idx_list = []
        self.one_idx_list = []
        self.two_idx_list = []
        self.three_idx_list = []
        self.four_idx_list = []
        self.five_idx_list = []
        self.six_idx_list = []
        self.seven_idx_list = []
        self.eight_idx_list = []
        self.nine_idx_list = []

        self.idx_list = [[] for i in range(10)]
        
        for i in range(len(self.train_data)):
            number = self.train_data[i][1]
            self.idx_list[number].append(i)
         
    def get_data(self, n_class):
        A = torch.cat([self.train_data[self.idx_list[n_class][i]][0].reshape(1, -1) for i in range(len(self.idx_list[n_class]))], axis=0)
        
        for i in range(A.shape[0]):
            A[i] /= A[i].sum()
        
        return [A[i].numpy() for i in range(A.shape[0])]
    
    def get_support(self):
        size = self.n_pixels
        d = size*size
        support = np.zeros((d, 2))

        count = 0
        for i in range(size):
            for j in range(size):
                support[count, 0] = i
                support[count, 1] = j
                count += 1
                
        return support

    def compute_M(self):
        basic_size = 28
        size = self.n_pixels
        d = size*size

        M = torch.zeros((d, d))
    
        tmp = torch.zeros(d, 2)
        for i in tqdm(range(d)):
            tmp[i, 0] = (i // size) * ((basic_size-1) / (size-1))
            tmp[i, 1] = (i % size)  * ((basic_size-1) / (size-1))
    
        for i in tqdm(range(d)):
            M[i] = torch.sqrt((tmp - tmp[i]).mm((tmp - tmp[i]).T).diag())

        return (M/M.max()).numpy().astype(np.float32)
