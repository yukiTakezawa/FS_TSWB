import numpy as np
import matplotlib.pyplot as plt
from wordcloud import *
import random
import sinkhorn
import torch
from tqdm import tqdm
import ot
import ot.gpu


def color_func(barycenter, words, top_k):
    sorted_idx = (np.argsort(barycenter))[-top_k:].tolist()
    cmap = plt.get_cmap("Greys")
    
    def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        idx = sorted_idx.index(words.index(word))
        #print(idx)
        #r, g, b, a = cmap(idx/top_k)
        #return (int(r*255), int(g*255), int(b*255))
        if idx < top_k - 5:
            return (0,128,0)
        else:
            return (165,42,42)
        
    return _color_func


def visualize(barycenter_list, words, save_flag=False, save_path=None, top_k=50):
    freq = []
    for i in range(len(barycenter_list)):
        freq.append({})

    for i in range(len(freq)):
        thr = np.sort(barycenter_list[i])[-top_k]

        for j in range(len(words)):

            if barycenter_list[i][j] >= thr:
                freq[i][words[j]] = barycenter_list[i][j]
            #else:
            #    freq[i][words[j]] = 0.0

    x, y = np.ogrid[:300, :300]

    mask = (x - 150) ** 2 + (y - 150) ** 2 > 150 ** 2
    mask = 255 * mask.astype(int)

    fig = plt.figure(figsize=(20, 200))
    
    for i in range(len(freq)):
        #wc = WordCloud(background_color="white", mask=mask, color_func=lambda *args, **kwargs: (0,128,0), prefer_horizontal=1.0).fit_words(freq[i])
        _color_func = color_func(barycenter_list[i], words, top_k)
        wc = WordCloud(background_color="white", mask=mask, prefer_horizontal=1.0, color_func=_color_func).fit_words(freq[i])
        
        
        ax = fig.add_subplot(1, len(freq), i+1)
        ax.axis("off")
        plt.imshow(wc)
        #ax.set_title(dataset["C"][i], fontsize=30)
        #plt.legend()
    #plt.show()
    
    if save_flag:
        plt.savefig(save_path, bbox_inches='tight',dpi=300)
        #plt.savefig(save_path, bbox_inches='tight')
        #plt.savefig("pic/amazon_quadtree_glove.png", bbox_inches='tight',dpi=300)


def visualize2(barycenter_list, words_list, save_flag=False, save_path=None, top_k=50):
    freq = []
    for i in range(len(barycenter_list)):
        freq.append({})

    for i in range(len(freq)):
        thr = np.sort(barycenter_list[i])[-top_k]

        for j in range(len(words_list[i])):

            if barycenter_list[i][j] >= thr:
                freq[i][words_list[i][j]] = barycenter_list[i][j]
            #else:
            #    freq[i][words_list[i][j]] = 0.0

    x, y = np.ogrid[:300, :300]

    mask = (x - 150) ** 2 + (y - 150) ** 2 > 150 ** 2
    mask = 255 * mask.astype(int)

    fig = plt.figure(figsize=(20, 200))
    
    for i in range(len(freq)):
        #wc = WordCloud(background_color="white", mask=mask, color_func=lambda *args, **kwargs: (0,128,0), prefer_horizontal=1.0).fit_words_list[i](freq[i])
        _color_func = color_func(barycenter_list[i], words_list[i], top_k)
        wc = WordCloud(background_color="white", mask=mask, prefer_horizontal=1.0, color_func=_color_func).fit_words(freq[i])
        
        
        ax = fig.add_subplot(1, len(freq), i+1)
        ax.axis("off")
        plt.imshow(wc)
        #ax.set_title(dataset["C"][i], fontsize=30)
        #plt.legend()
    #plt.show()
    
    if save_flag:
        plt.savefig(save_path, bbox_inches='tight',dpi=300)
        #plt.savefig(save_path, bbox_inches='tight')
        #plt.savefig("pic/amazon_quadtree_glove.png", bbox_inches='tight',dpi=300)

def calc_wb_loss(barycenter, a_list, M, n_iter=1000, device=0, reg=0.01):
    """
    Return entropic Waserstein Barycenter Loss.
    """
    
    sinkhorn_layer = sinkhorn.SinkhornLayer(reg)
    """
    for a in tqdm(a_list):
        loss += sinkhorn_layer(torch.tensor(M.astype(np.float32)).cuda(0), torch.tensor(barycenter.astype(np.float32)).cuda(0), torch.tensor(a.astype(np.float32)).cuda(0), 10000)
    """
    a = torch.tensor(a_list).float().cuda(device)

    #distance2 = ot.gpu.sinkhorn(barycenter, np.array(a_list).T, M, 0.01)
    #distance = ot.sinkhorn2(barycenter, np.array(a_list).T, M, 0.01)
    #print(distance2, distance)

    #print(distance)
    distance = sinkhorn_layer(torch.tensor(M).float().cuda(device), torch.tensor(barycenter).float().cuda(device), a, n_iter, device=device)
    loss = sum(distance)
    return loss / len(a_list)
